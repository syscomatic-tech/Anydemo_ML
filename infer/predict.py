
from logging import getLogger
from pathlib import Path
from typing import Literal

import librosa
import numpy as np
import soundfile
import torch
from cm_time import timer
from pydub import AudioSegment
from so_vits_svc_fork.inference.core import RealtimeVC, RealtimeVC2, Svc, LOG
from so_vits_svc_fork.utils import get_optimal_device
from scipy.io import wavfile
import numpy as np
from demucs.pretrained import get_model, DEFAULT_MODEL
from demucs.apply import apply_model
import soundfile as sf
from moviepy.editor import concatenate_audioclips, AudioFileClip
import threading
from functools import partial
import json
from torch import cuda
import os
from flask import stream_with_context
from .rvcp import *
import subprocess
import shutil

LOG = getLogger(__name__)

def split_audio_into_chunks(input_file, chunk_duration):
    # Read the audio file
    data, samplerate = sf.read(input_file)

    # Calculate the number of samples for each chunk
    chunk_samples = int(chunk_duration * samplerate)

    # Split the audio data into chunks
    num_chunks = len(data) // chunk_samples
    chunks = [data[i * chunk_samples:(i + 1) * chunk_samples] for i in range(num_chunks)]

    # Save each chunk as a separate audio file
    return chunks

def extract_vocal_demucs(model, filename, out_filename,sck, sr=44100, device=None, shifts=1, split=True, overlap=0.25, jobs=0):
    wav, sr = librosa.load("/home/paperspace/project/audio/"+filename, mono=False, sr=sr)
    wav = torch.tensor(wav)
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / ref.std()
    sources = apply_model(
        model,
        wav[None],
        sck_func=sck,
        device=device,
        shifts=shifts,
        split=split,
        overlap=overlap,
        progress=True,
        num_workers=jobs
    )[0]
    sources = sources * ref.std() + ref.mean()
    fnames = []
    for name,src in zip(model.sources,sources):
        wav = src
        wav = wav / max(1.01 * wav.abs().max(), 1)
        file = "/home/paperspace/project/audio/" + name + "." + out_filename 
        fnames.append(file)
        soundfile.write(file, wav.numpy().T, sr)
    return fnames

def infer(
    *,
    # paths
    fnames,
    music = None,
    sck = None,
    # svc config
    speaker = 0,
    cluster_model_path = None,
    s_name : str = "",
    transpose: int = 0,
    auto_predict_f0: bool = False,
    cluster_infer_ratio: float = 0,
    noise_scale: float = 0.4,
    f0_method: Literal["crepe", "crepe-tiny", "parselmouth", "dio", "harvest"] = "dio",
    # slice config
    db_thresh: int = -40,
    pad_seconds: float = 0.1,
    chunk_seconds: float = 0.1,
    absolute_thresh: bool = False,
    device = get_optimal_device(),
):
    model_path = Path("/home/paperspace/project/codes/models/" + s_name +"/mdl/G.pth")
    output_path = Path(fnames[-1])
    input_path = Path(fnames[-1])
    LOG.info(f"processing file {input_path}")
    config_path = Path("/home/paperspace/project/codes/models/" + s_name +"/mdl/config.json")
    cluster_model_path = Path(cluster_model_path) if cluster_model_path else None
    svc_model = Svc(
        net_g_path=model_path.as_posix(),
        config_path=config_path.as_posix(),
        cluster_model_path=cluster_model_path.as_posix()
        if cluster_model_path
        else None,
        device=device,
    )
    LOG.info("using device " + torch.cuda.get_device_name(device))
    audio, _ = librosa.load(input_path, sr=svc_model.target_sample)
    
    try:
        audio = svc_model.infer_silence(
            audio.astype(np.float32),
            sock_func = sck,
            speaker=speaker,
            transpose=transpose,
            auto_predict_f0=auto_predict_f0,
            cluster_infer_ratio=cluster_infer_ratio,
            noise_scale=noise_scale,
            f0_method=f0_method,
            db_thresh=db_thresh,
            pad_seconds=pad_seconds,
            chunk_seconds=chunk_seconds,
            absolute_thresh=absolute_thresh
        )
        with torch.cuda.device(device):
            LOG.info(f"allocated memory {torch.cuda.memory_allocated()}")


    except Exception as err:
        del svc_model
        print("with error ",torch.cuda.memory_allocated())
        raise err
    soundfile.write(output_path, audio, svc_model.target_sample)
    del svc_model
def overlay_audio(audio_clip_paths, output_path, sck):
    """Concatenates several audio files into one audio file using MoviePy
    and save it to `output_path`. Note that extension (mp3, etc.) must be added to `output_path`"""
    clips = [AudioSegment.from_file(c) for c in audio_clip_paths]
    final_clip = clips[0]
    prg = 95
    LOG.info("Starting combining vocals")
    for i in range(len(clips)):
        prg += 1
        sck(str(prg))
        if i > 0:
            final_clip = final_clip.overlay(clips[i], position=0)
        LOG.info(f"combining and deleting {audio_clip_paths[i]}")
        os.system(f"rm -r {audio_clip_paths[i]}")
    final_clip.export(output_path)
    sck(str(prg + 1))

def get_less_used_gpu(gpus=None, debug=False):
    """Inspect cached/reserved and allocated memory on specified gpus and return the id of the less used device"""
    if gpus is None:
        warn = 'Falling back to default: all gpus'
        gpus = range(cuda.device_count())
    elif isinstance(gpus, str):
        gpus = [int(el) for el in gpus.split(',')]

    # check gpus arg VS available gpus
    sys_gpus = list(range(cuda.device_count()))
    if len(gpus) > len(sys_gpus):
        gpus = sys_gpus
        warn = f'WARNING: Specified {len(gpus)} gpus, but only {cuda.device_count()} available. Falling back to default: all gpus.\nIDs:\t{list(gpus)}'
    elif set(gpus).difference(sys_gpus):
        # take correctly specified and add as much bad specifications as unused system gpus
        available_gpus = set(gpus).intersection(sys_gpus)
        unavailable_gpus = set(gpus).difference(sys_gpus)
        unused_gpus = set(sys_gpus).difference(gpus)
        gpus = list(available_gpus) + list(unused_gpus)[:len(unavailable_gpus)]
        warn = f'GPU ids {unavailable_gpus} not available. Falling back to {len(gpus)} device(s).\nIDs:\t{list(gpus)}'

    cur_allocated_mem = {}
    cur_cached_mem = {}
    max_allocated_mem = {}
    max_cached_mem = {}
    for i in gpus:
        cur_allocated_mem[i] = cuda.memory_allocated(i)
        cur_cached_mem[i] = cuda.memory_reserved(i)
        max_allocated_mem[i] = cuda.max_memory_allocated(i)
        max_cached_mem[i] = cuda.max_memory_reserved(i)
    min_allocated = min(cur_allocated_mem, key=cur_allocated_mem.get)
    if debug:
        print(warn)
        print('Current allocated memory:', {f'cuda:{k}': v for k, v in cur_allocated_mem.items()})
        print('Current reserved memory:', {f'cuda:{k}': v for k, v in cur_cached_mem.items()})
        print('Maximum allocated memory:', {f'cuda:{k}': v for k, v in max_allocated_mem.items()})
        print('Maximum reserved memory:', {f'cuda:{k}': v for k, v in max_cached_mem.items()})
        print('Suggested GPU:', min_allocated)
    return min_allocated


def free_memory(to_delete: list,gpu, debug=False):
    import gc
    import inspect
    calling_namespace = inspect.currentframe().f_back
    if debug:
        print('Before:')
        get_less_used_gpu(debug=True)

    for _var in to_delete:
        calling_namespace.f_locals.pop(_var, None)
        with torch.cuda.device(f'cuda:{gpu}'):
            gc.collect()
            cuda.empty_cache()
    if debug:
        print('After:')
        get_less_used_gpu(debug=True)
def sse_progress(gpu):
    with open(f'progress{gpu}.json', 'w') as file:
        data = json.loads(file)
    return f"{data}"

def predict(outfile_path, spk,gpu, sck=None, only_vocal=False, t=0):
    with torch.cuda.device(f'cuda:{gpu}'):
        demucs_model = get_model(DEFAULT_MODEL)
        LOG.info(f"demucs model sources {demucs_model.sources}")
        fname = extract_vocal_demucs(demucs_model, outfile_path, outfile_path,sck, device="cuda:"+str(gpu))
        del demucs_model
        infer(fnames=fname,music="",sck=sck,s_name=spk,transpose=t,device="cuda:"+str(gpu))
        if not only_vocal:
            overlay_audio(fname,"/home/paperspace/project/audio/"+outfile_path, sck)
            LOG.info("Freeing up gpu memory")
            torch.cuda.empty_cache()
            LOG.info("Done.............")
            return outfile_path
        else:
            os.system("rm -r "+"/home/paperspace/project/audio/"+outfile_path)
            overlay_audio(fname[0:-1],"/home/paperspace/project/audio/"+"music."+outfile_path,sck)
            LOG.info("Freeing up gpu memory")
            torch.cuda.empty_cache()
            LOG.info("Done.............")
            return [fname[-1][len("/home/paperspace/project/audio/"):],"music."+outfile_path]

def remix(infile,outfile,gpu,sck=None,in_has_vocs=True):
    demucs_model = get_model(DEFAULT_MODEL)
    if in_has_vocs:        
        fname = extract_vocal_demucs(demucs_model, outfile, outfile,sck, device="cuda:"+str(gpu))
        fname.remove(fname[-1])
        fname.append("/home/paperspace/project/audio/"+infile)
        overlay_audio(fname,"/home/paperspace/project/audio/"+"remix."+infile, sck)
        del demucs_model
        return "remix."+infile
    else:
        fname = extract_vocal_demucs(demucs_model, infile, infile,sck, device="cuda:"+str(gpu))
        for i in fname[0:-1]:
            os.system("rm -r "+i)
        fname1 = extract_vocal_demucs(demucs_model, outfile, outfile,sck, device="cuda:"+str(gpu))
        fname1.remove(fname1[-1])
        fname1.append(fname[-1])
        overlay_audio(fname1,"/home/paperspace/project/audio/"+"remix."+infile, sck)
        del demucs_model
        return "remix."+infile


def predict_rvc(outfile_path, spk,gpu, sck=None, only_vocal=False):
    demucs_model = get_model(DEFAULT_MODEL)
    LOG.info(f"demucs model sources {demucs_model.sources}")
    fname = extract_vocal_demucs(demucs_model, outfile_path, outfile_path,sck, device="cuda:"+str(gpu))
    del demucs_model
    model_path = "/home/paperspace/project/codes/r_mdl/"+spk+"/mdl/G.onnx"
    infer_onnx(model_path,fname[-1],fname[-1][0:-4]+".wav","vec-256-layer-9",d_id=gpu)
    os.system("rm -r "+fname[-1])
    fname[-1] = fname[-1][0:-4]+".wav"
    if not only_vocal:
        overlay_audio(fname,"/home/paperspace/project/audio/"+outfile_path, sck)
        LOG.info("Freeing up gpu memory")
        torch.cuda.empty_cache()
        LOG.info("Done.............")
        return fname[-1][len("/home/paperspace/project/audio/"):]
    else:
        os.system("rm -r "+"/home/paperspace/project/audio/"+outfile_path)
        overlay_audio(fname[0:-1],"/home/paperspace/project/audio/"+"music."+outfile_path,sck)
        LOG.info("Freeing up gpu memory")
        torch.cuda.empty_cache()
        LOG.info("Done.............")
        return [fname[-1][len("/home/paperspace/project/audio/"):],"music."+outfile_path]

def add_voice(url, speaker_name, link_type, mdl):
    if mdl == "rvc":
        dir_path = "/home/paperspace/project/codes/r_mdl/" + speaker_name
    elif mdl == "svc":
        dir_path = "/home/paperspace/project/codes/models/" + speaker_name
    os.makedirs(dir_path, exist_ok=True)  # creates the directory if not exist

    if link_type == 'gdrive':
        try:
            subprocess.run(['gdown', url, '-O', os.path.join(dir_path, 'file.zip')], check=True)
        except subprocess.CalledProcessError:
            print('Error in downloading from Google Drive. Check your url or try again later.')
            return
    else:
        try:
            subprocess.run(['wget', url, '-O', os.path.join(dir_path, 'file.zip')], check=True)
        except subprocess.CalledProcessError:
            print('Error in downloading with wget. Check your url or try again later.')
            return

    try:
        shutil.unpack_archive(os.path.join(dir_path, 'file.zip'), dir_path)
        os.system("rm -r " + os.path.join(dir_path, 'file.zip'))
        fl_nm = os.path.join(dir_path,os.listdir(dir_path)[0])
        mdl_dir = os.path.join(dir_path, "mdl")
        os.rename(fl_nm,mdl_dir)
        dirs = os.listdir(mdl_dir)
        for root, dirs, files in os.walk(mdl_dir):
            for file in files:
                source = os.path.join(root, file)
                destination = os.path.join(mdl_dir, file)
                shutil.move(source, destination)
        if mdl == "svc":
            for filename in os.listdir(mdl_dir):
                # check if filename starts with 'G' and ends with '.pth'
                if filename.startswith('G') and filename.endswith(".pth"):
                    # construct full file path
                    source = os.path.join(mdl_dir, filename)
                    destination = os.path.join(mdl_dir, "G.pth")

                    # rename the file
                    os.rename(source, destination)
        elif mdl == "rvc":
            for filename in os.listdir(mdl_dir):
                # check if filename starts with 'G' and ends with '.pth'
                if filename.endswith(".pth"):
                    # construct full file path
                    source = os.path.join(mdl_dir, filename)
                    destination = os.path.join(mdl_dir, "G.pth")

                    # rename the file
                    os.rename(source, destination)
            convert_to_onnx(os.path.join(mdl_dir, "G.pth"),os.path.join(mdl_dir, "G.onnx"))
            os.system("rm -r "+ os.path.join(mdl_dir, "G.pth"))
        return {"success":"model added"}
    except Exception as e:
        return {"error":str(e)}

if __name__ == "__main__":
    def send_sock(data, file_name, gpu):
        with open(f'progress{gpu}.json', 'w') as file:
            json.dump({'progress':data, "file":file_name}, file)
    link = "https://huggingface.co/QuickWick/Music-AI-Voices/resolve/main/XXXTENTACION%20(RVC)%20150%20Epoch%2014k%20Steps/XXXTENTACION%20(RVC)%20150%20Epoch%2014k%20Steps.zip"
    demucs_model = get_model(DEFAULT_MODEL)
    LOG.info(f"demucs model sources {demucs_model.sources}")
    fname = extract_vocal_demucs(demucs_model, "arab_commy.mp3", "arab_commy.mp3",partial(send_sock,file_name="arab_commy.mp3"), device="cuda:"+str(0))