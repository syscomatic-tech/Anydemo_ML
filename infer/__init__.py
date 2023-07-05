from flask import Flask, request, send_file, after_this_request, Response, stream_with_context
from flask_restful import Resource, Api
import os
import json
from .predict import *
import time
import torch
import requests
from urllib.parse import unquote
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import datetime
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 999 * 1024 * 1024


with open('data.json', 'r') as file:
    data = json.load(file)
GPU = data["use_gpu"]

if data["use_gpu"] == 0:
    data["use_gpu"] = 1
elif data["use_gpu"] == 1:
    data["use_gpu"] = 0
with open('data.json', 'w') as file:
    json.dump(data, file)


def list_directories(folder_path):
    directories = []

    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            directories.append(item)
    return directories

def send_sock(data, file_name):
    global GPU
    with open(f'progress{GPU}.json', 'w') as file:
        json.dump({'progress':data, "file":file_name}, file)

@app.route("/download/<string:file_name>", methods=["GET"])
def download(file_name):
    try:
        resp = send_file("/home/paperspace/project/audio/"+file_name)
        return resp
    except Exception as e:
        return {"messege":str(e)},400

@app.route("/delete/<string:file_name>", methods=["GET"])
def delete(file_name):
    try:
        os.system(f"rm -r {'/home/paperspace/project/audio/' + file_name}")
        return {"messege":f"Successfully deleted {file_name}"},200
    except Exception as e:
        return {"messege":str(e)},400



@app.route('/predict/<string:speaker>/<path:url>',methods=["GET"])
def audio_api(speaker, url):
    global GPU
    gpu = GPU
    path = unquote(url)
    transp = int(request.args["transpose"])
    print(f"transpose is {transp}")
    print(path)
    audio_url = path
    file_name = 'audio.mp3'
    response = requests.get(audio_url)
    response.raise_for_status()
            # Open the JSON file
    with open('data.json', 'r') as file:
        data = json.load(file)

    file_name = str(data['total_reqs']) + "." +file_name
    data['total_reqs'] = data['total_reqs'] + 1
    ## whether using main gpu or not ?

    ## save the data
    with open('data.json', 'w') as file:
        json.dump(data, file)
    
    with open("/home/paperspace/project/audio/" + file_name, 'wb') as file:
        file.write(response.content)
    
    try:
        out = predict(file_name,speaker, gpu,sck=partial(send_sock,file_name=file_name), t=transp)
        resp = send_file("/home/paperspace/project/audio/"+file_name)
        os.system(f"rm -r {'/home/paperspace/project/audio/' + file_name}")
        return resp
        # return Response(,mimetype='text/event-stream')
    except Exception as e:
        return {"messege":str(e)},400

@app.route('/upload/<path:url>',methods=["GET"])
def upload(url):
    path = unquote(url)
    print(path)
    audio_url = path
    file_name = 'audio.mp3'
    response = requests.get(audio_url)
    response.raise_for_status()
            # Open the JSON file
    with open('data.json', 'r') as file:
        data = json.load(file)

    file_name = str(data['total_reqs']) + "." +file_name
    data['total_reqs'] = data['total_reqs'] + 1
    ## whether using main gpu or not ?

    ## save the data
    with open('data.json', 'w') as file:
        json.dump(data, file)
    
    with open("/home/paperspace/project/audio/" + file_name, 'wb') as file:
        file.write(response.content)
    
    try:
        return {"messege":"uploaded successfully","file_name":file_name}
        # return Response(,mimetype='text/event-stream')
    except Exception as e:
        return {"messege":str(e)},400

@app.route('/predict2/<string:speaker>/<string:file_name>',methods=["GET"])
def audio_api2(speaker, file_name):
    global GPU
    gpu = GPU
    transp = int(request.args["transpose"])
    print(f"transpose is {transp}")
    
    try:
        out = predict(file_name,speaker, gpu,sck=partial(send_sock,file_name=file_name), t=transp)
        resp = send_file("/home/paperspace/project/audio/"+file_name)
        os.system(f"rm -r {'/home/paperspace/project/audio/' + file_name}")
        return resp
        # return Response(,mimetype='text/event-stream')
    except Exception as e:
        return {"messege":str(e)},400
@app.route("/progress/<string:file_name>")
def progress(file_name):
    prg = 0
    def func2():
        global prg
        with open('progress0.json', 'r') as file:
            data1 = json.load(file)
            with open('progress1.json', 'r') as file:
                data2 = json.load(file)
            for i in [data1,data2]:
                if i["file"] == file_name:
                    prg = i["progress"]
                    return 'progress: ' + prg
    try:
        return Response(func2(), mimetype='text/event-stream')
    except Exception as e:
        return {"messege":str(e)},400

@app.route("/list_models", methods=["GET"])
def lst_mdl():
    try:
        models = list_directories("/home/paperspace/project/codes/models")
        rvc_mdl = list_directories("/home/paperspace/project/codes/r_mdl")
        rvc_mdl.remove("pretrained")
        return {"RVC Models":rvc_mdl, "So_vits_svc models":models},200
    except Exception as e:
        return {"error":str(e)},400

@app.route("/predict3", methods=['POST'])
def prd3():
    data = request.json
    if data["convert"]:
        try:
            mdl_type = data["model_type"]
            speaker = data["speaker"]
            onl_voice = data["only_voice"]
            remx = data["remix"]
            transpose = data["transpose"]
            file_path = data["filename"]
            r_link = ""
            outp = ""
            if mdl_type == "rvc":
                if remx:
                    onl_voice = True
                outp = predict_rvc(file_path,speaker,GPU,sck=partial(send_sock,file_name=data["filename"]),only_vocal=onl_voice)
            elif mdl_type == "svc":
                if remx:
                    onl_voice = True
                outp = predict(file_path,speaker,GPU,sck=partial(send_sock,file_name=data["filename"]),only_vocal=onl_voice)
            if remx:
                r_file = data["remix_path"]
                op = remix(outp[0],r_file,GPU, sck=partial(send_sock,file_name=data["filename"]))
                return {"files":op},200
            return {"files":outp},200
        except Exception as e:
            return {"error":str(e)},400
    elif data["remix"]:
        try:
            r_link = data["remix_path"]
            op = remix(data["filename"],r_link,GPU, sck=partial(send_sock,file_name=data["filename"]))
            return {"files":op},200
        except Exception as e:
            return {"error":str(e)}
    elif data["only_voice"]:
        demucs_model = get_model(DEFAULT_MODEL)
        fname = extract_vocal_demucs(demucs_model, data["filename"], data["filename"],partial(send_sock,file_name=data["filename"]), device="cuda:"+str(GPU))
        overlay_audio(fname[0:-1],"/home/paperspace/project/audio/music."+data["filename"],partial(send_sock,file_name=data["filename"]))
        del demucs_model
        return {"files":[fname[-1][len("/home/paperspace/project/audio/"):],"music."+data["filename"]]},200
    else:
        return {"error":"either of convert, remix, only_voice needs to be true"}

@app.route("/add_mdl/<string:mdl_type>/<string:speaker>/<path:url>")
def add_mdl(mdl_type,speaker,url):
    lnk_tp = "any"
    if "drive.google.com" in url:
        lnk_tp = "gdrive"
    return add_voice(url,speaker,lnk_tp, mdl_type)

@app.route("/add_custom/<string:name>/<path:url>")
def add_custom(name,url):
    try:
        dir_path = "/home/paperspace/project/codes/r_mdl/" + name +"/mdl"
        os.makedirs(dir_path, exist_ok=True)
        subprocess.run(['wget', unquote(url), '-O', os.path.join(dir_path, 'G.pth')], check=True)
        convert_to_onnx(os.path.join(dir_path, "G.pth"),os.path.join(dir_path, "G.onnx"),version=request.args["v"])
        return {"success":"successfully added model "+name},200
    except Exception as e:
        return {"error":str(e)},400

if __name__ == '__main__':
    app.run(port=8000,debug=False,threaded=True)
