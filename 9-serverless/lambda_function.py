import onnxruntime as ort
from io import BytesIO
from urllib import request
from PIL import Image
import numpy as np

onnx_model_path = "hair_classifier_empty.onnx"
session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])

inputs = session.get_inputs()
outputs = session.get_outputs()

input_name = inputs[0].name
output_name = outputs[0].name



def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def preprocess(url):
    img = download_image(url)

    if img.mode != "RGB":
        img = img.convert("RGB")

    img = img.resize((200, 200), Image.BILINEAR)

    x = np.array(img).astype("float32") / 255.0
    x = np.transpose(x, (2, 0, 1))  # HWC → CHW

    mean = np.array([0.485, 0.456, 0.406], dtype="float32").reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype="float32").reshape(3, 1, 1)

    x = (x - mean) / std
    x = np.expand_dims(x, axis=0)
    x = x.astype("float32") 

    return x
def predict(url):
    X = preprocess(url)
    result = session.run([output_name], {input_name: X})
    return result


def lambda_handler(event, context):
    url = event["url"]
    result = predict(url)
    score = float(result[0][0][0])

    return {
        "prediction": score
    }
