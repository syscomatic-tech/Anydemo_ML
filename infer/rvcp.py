import torch
import soundfile
from rvc.infer_pack.onnx_inference import OnnxRVC
from rvc.infer_pack.models_onnx import SynthesizerTrnMsNSFsidM

def convert_to_onnx(ModelPath, ExportedPath, version="v1"):
    MoeVS = True  # 模型是否为MoeVoiceStudio（原MoeSS）使用
    if version == "v1":
        hidden_channels = 256  # hidden_channels，为768Vec做准备
    else:
        hidden_channels = 768
    cpt = torch.load(ModelPath, map_location="cpu")
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
    print(*cpt["config"])

    test_phone = torch.rand(1, 200, hidden_channels)  # hidden unit
    test_phone_lengths = torch.tensor([200]).long()  # hidden unit 长度（貌似没啥用）
    test_pitch = torch.randint(size=(1, 200), low=5, high=255)  # 基频（单位赫兹）
    test_pitchf = torch.rand(1, 200)  # nsf基频
    test_ds = torch.LongTensor([0])  # 说话人ID
    test_rnd = torch.rand(1, 192, 200)  # 噪声（加入随机因子）

    device = "cpu"  # 导出时设备（不影响使用模型）

    net_g = SynthesizerTrnMsNSFsidM(
        *cpt["config"], version, is_half=False
    )  # fp32导出（C++要支持fp16必须手动将内存重新排列所以暂时不用fp16）
    net_g.load_state_dict(cpt["weight"], strict=False)
    input_names = ["phone", "phone_lengths", "pitch", "pitchf", "ds", "rnd"]
    output_names = [
        "audio",
    ]
    # net_g.construct_spkmixmap(n_speaker) 多角色混合轨道导出
    torch.onnx.export(
        net_g,
        (
            test_phone.to(device),
            test_phone_lengths.to(device),
            test_pitch.to(device),
            test_pitchf.to(device),
            test_ds.to(device),
            test_rnd.to(device),
        ),
        ExportedPath,
        dynamic_axes={
            "phone": [1],
            "pitch": [1],
            "pitchf": [1],
            "rnd": [2],
        },
        do_constant_folding=False,
        opset_version=16,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
    )

def infer_onnx(model_path, wav_path, out_path, vec_name, sampling_rate = 40000, hop_size = 512, f0_method = "dio", d_id = 0): 
    f0_up_key = 0  # 升降调
    sid = 0  # 角色ID
    model = OnnxRVC(
        model_path, vec_path=vec_name, sr=sampling_rate, hop_size=hop_size, device="cuda", d_id = d_id
    )

    audio = model.inference2(wav_path, sid, f0_method=f0_method, f0_up_key=f0_up_key)
    soundfile.write(out_path, audio, sampling_rate)

if __name__ == "__main__":
    convert_to_onnx("/home/paperspace/project/codes/r_mdl/gig/mdl/gigs.pth","/home/paperspace/project/codes/r_mdl/gig/mdl/G.onnx",version="v2")
    infer_onnx("/home/paperspace/project/codes/r_mdl/gig/mdl/G.onnx","/home/paperspace/project/audio/tts.mp3","/home/paperspace/project/audio/aout.wav","vec-768-layer-12", d_id=0)