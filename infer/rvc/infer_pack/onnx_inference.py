import onnxruntime
import librosa
import numpy as np
from typing import Any, Callable, Iterable, Literal
from numpy import dtype,ndarray,float32
import soundfile


class Chunk:
    is_speech: bool
    audio: ndarray[Any, dtype[float32]]
    start: int
    end: int

    @property
    def duration(self) -> float32:
        # return self.end - self.start
        return float32(self.audio.shape[0])

    def __repr__(self) -> str:
        return f"Chunk(Speech: {self.is_speech}, {self.duration})"


def split_silence(
    audio: ndarray[Any, dtype[float32]],
    top_db: int = 40,
    ref: Callable[[ndarray[Any, dtype[float32]]], float] or float= 1,
    frame_length: int = 2048,
    hop_length: int = 512,
    aggregate: Callable[[ndarray[Any, dtype[float32]]], float] = np.mean,
) -> Iterable[Chunk]:
    non_silence_indices = librosa.effects.split(
        audio,
        top_db=top_db,
        ref=ref,
        frame_length=frame_length,
        hop_length=hop_length,
        aggregate=aggregate,
    )
    last_end = 0
    for start, end in non_silence_indices:
        if start != last_end:
            yield Chunk(
                is_speech=False, audio=audio[last_end:start], start=last_end, end=start
            )
        yield Chunk(is_speech=True, audio=audio[start:end], start=start, end=end)
        last_end = end
    if last_end != len(audio):
        yield Chunk(
            is_speech=False, audio=audio[last_end:], start=last_end, end=len(audio)
        )

class ContentVec:
    def __init__(self, vec_path="pretrained/vec-768-layer-12.onnx", device=None):
        print("load model(s) from {}".format(vec_path))
        if device == "cpu" or device is None:
            providers = ["CPUExecutionProvider"]
        elif device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif device == "dml":
            providers = ["DmlExecutionProvider"]
        else:
            raise RuntimeError("Unsportted Device")
        self.model = onnxruntime.InferenceSession(vec_path, providers=providers)

    def __call__(self, wav):
        return self.forward(wav)

    def forward(self, wav):
        feats = wav
        if feats.ndim == 2:  # double channels
            feats = feats.mean(-1)
        assert feats.ndim == 1, feats.ndim
        feats = np.expand_dims(np.expand_dims(feats, 0), 0)
        onnx_input = {self.model.get_inputs()[0].name: feats}
        logits = self.model.run(None, onnx_input)[0]
        return logits.transpose(0, 2, 1)


def get_f0_predictor(f0_predictor, hop_length, sampling_rate, **kargs):
    if f0_predictor == "pm":
        from rvc.infer_pack.modules.F0Predictor.PMF0Predictor import PMF0Predictor

        f0_predictor_object = PMF0Predictor(
            hop_length=hop_length, sampling_rate=sampling_rate
        )
    elif f0_predictor == "harvest":
        from rvc.infer_pack.modules.F0Predictor.HarvestF0Predictor import HarvestF0Predictor

        f0_predictor_object = HarvestF0Predictor(
            hop_length=hop_length, sampling_rate=sampling_rate
        )
    elif f0_predictor == "dio":
        from rvc.infer_pack.F0Predictor.DioF0Predictor import DioF0Predictor

        f0_predictor_object = DioF0Predictor(
            hop_length=hop_length, sampling_rate=sampling_rate
        )
    else:
        raise Exception("Unknown f0 predictor")
    return f0_predictor_object


class OnnxRVC:
    def __init__(
        self,
        model_path,
        sr=40000,
        hop_size=512,
        vec_path="vec-768-layer-12",
        device="cuda",
        d_id=0
    ):
        vec_path = f"/home/paperspace/project/codes/r_mdl/pretrained/{vec_path}.onnx"
        self.vec_model = ContentVec(vec_path, device)
        if device == "cpu" or device is None:
            providers = ["CPUExecutionProvider"]
        elif device == "cuda":
            # providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            providers = [
                ('CUDAExecutionProvider', {
                    'device_id': d_id,
                }),
                'CPUExecutionProvider',
            ]

        elif device == "dml":
            providers = ["DmlExecutionProvider"]
        else:
            raise RuntimeError("Unsportted Device")
        self.model = onnxruntime.InferenceSession(model_path, providers=providers)
        self.sampling_rate = sr
        self.hop_size = hop_size

    def forward(self, hubert, hubert_length, pitch, pitchf, ds, rnd):
        onnx_input = {
            self.model.get_inputs()[0].name: hubert,
            self.model.get_inputs()[1].name: hubert_length,
            self.model.get_inputs()[2].name: pitch,
            self.model.get_inputs()[3].name: pitchf,
            self.model.get_inputs()[4].name: ds,
            self.model.get_inputs()[5].name: rnd,
        }
        return (self.model.run(None, onnx_input)[0] * 32767).astype(np.int16)
    def inference2(
        self,
        raw_path,
        sid,
        f0_method="dio",
        f0_up_key=0,
        pad_time=0.5,
        cr_threshold=0.02,
    ):
        f0_min = 50
        f0_max = 1100
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)
        f0_predictor = get_f0_predictor(
            f0_method,
            hop_length=self.hop_size,
            sampling_rate=self.sampling_rate,
            threshold=cr_threshold,
        )
        wy, sr = librosa.load(raw_path, sr=self.sampling_rate)
        org_length = len(wy)

        total_time = org_length / sr
        start_time = 0  # Start time in seconds
        end_time = 30  # End time in seconds
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        wav = wy[start_sample:end_sample]
        # if org_length / sr > 50.0:
        #     raise RuntimeError("Reached Max Length")
        wav16k = librosa.resample(wav, orig_sr=self.sampling_rate, target_sr=16000)
        wav16k = wav16k

        hubert = self.vec_model(wav16k)
        hubert = np.repeat(hubert, 2, axis=2).transpose(0, 2, 1).astype(np.float32)
        hubert_length = hubert.shape[1]

        pitchf = f0_predictor.compute_f0(wav, hubert_length)
        pitchf = pitchf * 2 ** (f0_up_key / 12)
        pitch = pitchf.copy()
        f0_mel = 1127 * np.log(1 + pitch / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (
            f0_mel_max - f0_mel_min
        ) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        pitch = np.rint(f0_mel).astype(np.int64)

        pitchf = pitchf.reshape(1, len(pitchf)).astype(np.float32)
        pitch = pitch.reshape(1, len(pitch))
        ds = np.array([sid]).astype(np.int64)

        rnd = np.random.randn(1, 192, hubert_length).astype(np.float32)
        hubert_length = np.array([hubert_length]).astype(np.int64)

        out_wav = self.forward(hubert, hubert_length, pitch, pitchf, ds, rnd).squeeze()
        out_wav = np.pad(out_wav, (0, 2 * self.hop_size), "constant")
        print(out_wav.shape)

        return out_wav[0:org_length]
    def inference(
        self,
        raw_path,
        sid,
        f0_method="dio",
        f0_up_key=0,
        pad_time=0.5,
        cr_threshold=0.02,
    ):
        f0_min = 50
        f0_max = 1100
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)
        f0_predictor = get_f0_predictor(
            f0_method,
            hop_length=self.hop_size,
            sampling_rate=self.sampling_rate,
            threshold=cr_threshold,
        )
        wy, sr = librosa.load(raw_path, sr=self.sampling_rate)
        org_length = len(wy)

        total_time = org_length / sr
        start_time = 0  # Start time in seconds
        end_time = 30  # End time in seconds
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        wav = wy[start_sample:end_sample]
        wavs = None
        # if org_length / sr > 50.0:
        #     raise RuntimeError("Reached Max Length")
        while True:
            wav16k = librosa.resample(wav, orig_sr=self.sampling_rate, target_sr=16000)
            wav16k = wav16k

            hubert = self.vec_model(wav16k)
            hubert = np.repeat(hubert, 2, axis=2).transpose(0, 2, 1).astype(np.float32)
            hubert_length = hubert.shape[1]

            pitchf = f0_predictor.compute_f0(wav, hubert_length)
            pitchf = pitchf * 2 ** (f0_up_key / 12)
            pitch = pitchf.copy()
            f0_mel = 1127 * np.log(1 + pitch / 700)
            f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (
                f0_mel_max - f0_mel_min
            ) + 1
            f0_mel[f0_mel <= 1] = 1
            f0_mel[f0_mel > 255] = 255
            pitch = np.rint(f0_mel).astype(np.int64)

            pitchf = pitchf.reshape(1, len(pitchf)).astype(np.float32)
            pitch = pitch.reshape(1, len(pitch))
            ds = np.array([sid]).astype(np.int64)

            rnd = np.random.randn(1, 192, hubert_length).astype(np.float32)
            hubert_length = np.array([hubert_length]).astype(np.int64)

            out_wav = self.forward(hubert, hubert_length, pitch, pitchf, ds, rnd).squeeze()
            out_wav = np.pad(out_wav, (0, 2 * self.hop_size), "constant")
            if wavs is None:
                wavs = out_wav
            else:
                wavs = np.append(wavs,out_wav)
            if not total_time - end_time < 30:
                start_time = end_time
                end_time += 30
            elif end_time == total_time:
                break
            else:
                start_time = end_time
                end_time = total_time
            wav = wy[int(start_time * sr):int(end_time * sr)]
            print([start_time,end_time])
        return wavs[0:org_length]

