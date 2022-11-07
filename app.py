from flask import Flask, render_template, jsonify
import os
from flask import request

#from tensorflow.keras.preprocessing import image
#import numpy as np
#from keras import applications
#import tensorflow as tf

from PIL import Image
import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn

from datetime import datetime


project_dir = os.path.dirname(os.path.abspath(__file__))


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'images'

ALLOWED_EXTENSIONS = {'webp', 'tiff', 'png', 'jpg', 'jpeg'}


def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model_conv = torchvision.models.resnet50(pretrained=True)

    for param in model_conv.parameters():
        param.requires_grad = False

    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 2)

    # loading the weights
    model_conv.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model_conv.eval()
    return model_conv


def applyTransforms(inp):
    outp = transforms.functional.resize(inp, [224, 224])
    outp = transforms.functional.to_tensor(outp)
    outp = transforms.functional.normalize(
        outp, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return outp


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET'])
def home():
    return render_template("index.html")



@app.route("/predict", methods=["POST"])
def predict():
    if 'photo' not in request.files:
        response = {"status": 500,
                    "status_msg": "File is not uploaded", "message": ""}
        return jsonify(response)


    file = request.files['photo']
    if file.filename == '':
        response = {"status": 500,
                    "status_msg": "No image Uploaded", "message": ""}
        return jsonify(response)


    if file and not allowed_file(file.filename):
        response = {
            "status": 500, "status_msg": "File extension is not permitted", "message": ""}
        return jsonify(response)


    name = str(datetime.now().microsecond) + str(datetime.now().month) + '-' + str(datetime.now().day) + '.jpg'
    photo = request.files['photo']
    path = os.path.join(app.config['UPLOAD_FOLDER'], name)
    photo.save(path)


    model_conv = load_checkpoint("checkpoint.pth")


    img = Image.open(path)
    imageTensor = applyTransforms(img)
    minibatch = torch.stack([imageTensor])
    # model_conv(minibatch)
    softMax = nn.Softmax(dim=1)
    preds = softMax(model_conv(minibatch))


    result = "Fake"
    if preds[0,1].item() > preds[0,0].item():
        result = "Real"

    
    print("Fake : ",preds[0,0].item())
    print("Real : ",preds[0,1].item())

    os.unlink(path)

    response = {"status": 200, "status_msg": result}
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)