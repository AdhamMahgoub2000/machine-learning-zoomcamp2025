import onnxruntime as ort
from io import BytesIO
from urllib import request
from torchvision import transforms
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


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img
def preprocess(url):

    transforms = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
        ) 
        ])
    img = download_image(url)
    img = prepare_image(img, (200, 200))
    X = transforms(img)
    X_np = X.numpy()
    X_np = np.expand_dims(X_np, axis=0)
    return X_np

def predict(url):
    X = preprocess(url)
    result = session.run([output_name], {input_name: X})
    return result


def lambda_handler(event, context):
    url = event["url"]
    result = predict(url)
    return result


