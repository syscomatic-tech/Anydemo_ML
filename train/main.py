from flask import Flask, request, send_file, after_this_request, Response, stream_with_context
from train import *
import concurrent.futures
from functools import partial
import requests
import glob
from urllib.parse import unquote

app = Flask(__name__)

executor = concurrent.futures.ThreadPoolExecutor()

mdl_name = "cln"

@app.route("/download/<string:file_name>", methods=["GET"])
def download(file_name):
    try:
        resp = send_file("/home/paperspace/project/Retrieval-based-Voice-Conversion-WebUI/weights/"+file_name)
        return resp
    except Exception as e:
        return {"messege":str(e)},400


@app.route("/train", methods=["POST"])
def trn():
    data = request.json
    global mdl_name
    
    MODELNAME = str(data["name"])
    mdl_name = MODELNAME
    VERSION = str(data["version"])
    MODELEPOCH = str(data["epoch"])
    EPOCHSAVE = str(data["epoch_save"])
    onl_vc = not data["has_vocals"]
    executor.submit(partial(train,MODELNAME=MODELNAME,VERSION=VERSION,MODELEPOCH=MODELEPOCH,EPOCHSAVE=EPOCHSAVE,do_voc=onl_vc))
    return {"messege":"Training started. Check logs endpoint for progress."}

@app.route('/logs/<string:filename>')
def get_logs(filename):
    try:
        with open("/home/paperspace/project/Retrieval-based-Voice-Conversion-WebUI/logs/"+mdl_name+"/"+filename, 'r') as log_file:
            logs = log_file.read()

        return logs, 200
    except FileNotFoundError:
        return 'Log file not found. wait and try again', 404

@app.route('/list_logs')
def get_logs_list():
    return {"log_files":[fl[len("/home/paperspace/project/Retrieval-based-Voice-Conversion-WebUI/logs/"+mdl_name+"/"):] for fl in glob.glob("/home/paperspace/project/Retrieval-based-Voice-Conversion-WebUI/logs/"+mdl_name+"/"+"*.log")]}
@app.route('/upload/<path:url>',methods=["GET"])
def upload(url):
    path = unquote(url)
    r_pth = "/home/paperspace/project/dataset/"
    print(path)
    audio_url = path
    response = requests.get(audio_url)
    file_name = str(len(os.listdir(r_pth))) + '.audio.mp3'
    with open("/home/paperspace/project/dataset/" + file_name, 'wb') as file:
        file.write(response.content)
    return {"messege":"successfully uploaded"}

@app.route('/delete',methods=["GET"])
def delete():
    os.system("rm -r /home/paperspace/project/dataset/*")
    return {"messege": "deleted"}

@app.route("/progress",methods=["GET"])
def progress():
    pass
if __name__ == "__main__":
    app.run(port=8000,debug=False)
