import subprocess
import os
import torch
import librosa
import soundfile
from random import shuffle
from demucs.pretrained import get_model, DEFAULT_MODEL
from demucs.apply import apply_model
import noisereduce as nr
import requests
from urllib.parse import quote

def file_list(now_dir, exp_dir1, if_f0_3,version19, spk_id5=0, sr2="40k"):
    exp_dir = "%s/logs/%s" % (now_dir, exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    gt_wavs_dir = "%s/0_gt_wavs" % (exp_dir)
    feature_dir = (
        "%s/3_feature256" % (exp_dir)
        if version19 == "v1"
        else "%s/3_feature768" % (exp_dir)
    )
    if if_f0_3:
        f0_dir = "%s/2a_f0" % (exp_dir)
        f0nsf_dir = "%s/2b-f0nsf" % (exp_dir)
        names = (
            set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
            & set([name.split(".")[0] for name in os.listdir(feature_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
        )
    else:
        names = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & set(
            [name.split(".")[0] for name in os.listdir(feature_dir)]
        )
    opt = []
    for name in names:
        if if_f0_3:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    f0_dir.replace("\\", "\\\\"),
                    name,
                    f0nsf_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
        else:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
    fea_dim = 256 if version19 == "v1" else 768
    if if_f0_3:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, now_dir, now_dir, spk_id5)
            )
    else:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, spk_id5)
            )
    shuffle(opt)
    with open("%s/filelist.txt" % exp_dir, "w") as f:
        f.write("\n".join(opt))

def extract_vocal_demucs(filename, out_filename, sr=44100, device=None, shifts=1, split=True, overlap=0.25, jobs=0):
    model = get_model(DEFAULT_MODEL)
    wav, sr = librosa.load("/home/paperspace/project/dataset/"+filename, mono=False, sr=sr)
    wav = torch.tensor(wav)
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / ref.std()
    sources = apply_model(
        model,
        wav[None],
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
        file = "/home/paperspace/project/dataset/" + name + "." + out_filename 
        fnames.append(file)
        soundfile.write(file, wav.numpy().T, sr)
    for i in fnames[0:-1]:
        os.system("rm -r "+i)
    os.system("rm -r "+"/home/paperspace/project/dataset/"+filename)
    return fnames[-1]
# create a new directory
def train(
    MODELNAME = "giga",
    BITRATE = "40000",
    THREADCOUNT = "8",
    ALGO = "dio",
    MODELSAMPLE = "40k",
    BATCHSIZE = "4",
    USEGPU = "0",
    MODELEPOCH = "20",
    EPOCHSAVE = "5",
    ONLYLATEST = "0",
    CACHEDATA = "1",
    VERSION = "v2",
    do_voc = False
):
    os.chdir("/home/paperspace/project/Retrieval-based-Voice-Conversion-WebUI/")
    #if not os.path.exists("/home/paperspace/project/Retrieval-based-Voice-Conversion-WebUI/logs/{MODELNAME}"):
    subprocess.run(["mkdir", "-p", f"/home/paperspace/project/Retrieval-based-Voice-Conversion-WebUI/logs/{MODELNAME}"], check=True)
    if do_voc:
        for data in os.listdir("/home/paperspace/project/dataset"):
            audio, rate = soundfile.read("/home/paperspace/project/dataset/"+data)
            reduced_noise = nr.reduce_noise(audio_clip=audio, noise_clip=audio, verbose=True)
            soundfile.write("/home/paperspace/project/dataset/"+data, rate, reduced_noise)
            print(extract_vocal_demucs(data,data,device="cuda:0"))
    # preprocess pipeline
    subprocess.run(["python3", "trainset_preprocess_pipeline_print.py", "/home/paperspace/project/dataset", BITRATE, THREADCOUNT, f"logs/{MODELNAME}", "True"], check=True)

    # extract f0
    subprocess.run(["python3", "extract_f0_print.py", f"logs/{MODELNAME}", THREADCOUNT, ALGO], check=True)

    # extract feature
    subprocess.run(["python3", "extract_feature_print.py", "cpu", "1", "0", f"logs/{MODELNAME}", VERSION], check=True)

    file_list("/home/paperspace/project/Retrieval-based-Voice-Conversion-WebUI",MODELNAME,True,VERSION)
    # train nsf
    subprocess.run([
        "python3", "train_nsf_sim_cache_sid_load_pretrain.py", "-v", VERSION, "-e", MODELNAME, "-sr", MODELSAMPLE, "-f0", "1", 
        "-bs", BATCHSIZE, "-g", USEGPU, "-te", MODELEPOCH, "-se", EPOCHSAVE, 
        "-pg", f"pretrained_v2/f0G{MODELSAMPLE}.pth", "-pd", f"pretrained_v2/f0D{MODELSAMPLE}.pth", 
        "-l", ONLYLATEST, "-c", CACHEDATA
    ], check=True)
    os.system("rm -r /home/paperspace/project/dataset/*")
    requests.get("http://184.105.3.254/add_custom/"+MODELNAME+"/"+quote("http://184.105.4.243/download/"+MODELNAME+".pth")+"?v="+VERSION)
if __name__ == "__main__":
    train(MODELNAME="gigs")