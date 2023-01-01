import numpy as np
import onnxruntime as ort
import os
import re
import torch
from PIL import Image
from a_dataloaders import base_transforms
import torchvision.transforms as transforms
import random
from pathlib import Path
import yaml


idx_label = {
    0: 'airplane', 
    1: 'automobile', 
    2: 'bird', 
    3: 'cat', 
    4: 'deer', 
    5: 'dog', 
    6: 'frog', 
    7: 'horse', 
    8: 'ship', 
    9: 'truck'}

input_dir = '/content/drive/MyDrive/Colab/MIPT/CV/HW_1/Bharani_Lightning/cifar10/cifar-10-batches-py/test_batch'
label_file = '/content/drive/MyDrive/Colab/MIPT/CV/HW_1/Bharani_Lightning/cifar10/cifar-10-batches-py/batches.meta'
to_infer_data_path = '/content/drive/MyDrive/Colab/MIPT/CV/HW_1/Bharani_Lightning/imgs'

def read_bin(path):
    import pickle
    with open(path, 'rb') as f:
        r = pickle.load(f, encoding='bytes')
    return r

def generate_random_10_images(input_dir, to_infer_data_path, label_file):
    try: os.makedirs(to_infer_data_path)
    except: pass

    images = read_bin(input_dir)
    files = images[b"filenames"]
    if b"labels" in images:
        image_labels = images[b"labels"]
    else:
        image_labels = images[b"coarse_labels"]
    image_data = images[b"data"]

    imgs_files = random.sample(files, 10)
    for i, _ in enumerate(imgs_files):
        img_file = imgs_files[i]
        image_label = image_labels[i]
        image = image_data[i]
        path = os.path.join(os.path.join(to_infer_data_path, img_file.decode('utf-8')))
        out_image_1 = np.reshape(image, [3, 32, 32]).transpose(1, 2, 0)
        out_image_2 = Image.fromarray(out_image_1)
        with open(path, mode='wb') as o:
            out_image_2.save(o)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def softmaxx(x):
    e = np.exp(x)
    return e / e.sum()

def infer_predict(model_onnx, data):
    ort_session = ort.InferenceSession(model_onnx)
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(data)}
    ort_outs = ort_session.run(None, ort_inputs)
    img_out_y = ort_outs[0]
    softmaxed_y = softmaxx(img_out_y)
    predictions_idx = np.argmax(softmaxx(img_out_y), 1)
    predictions_label = [idx_label.get(i) for i in predictions_idx]
    return predictions_idx, predictions_label

def get_onnx_model(path_to_model):
    onnx_model = sorted(Path(path_to_model).glob('*.onnx'))[0]
    model_onnx = Path(onnx_model).as_posix()
    return model_onnx

def write_to_yaml(path_to_file, predictions):
    with open(path_to_file, 'w') as f:
        yaml.dump(predictions, f)

def transform_image(imgs_list):
    to_infer_batch = []
    for i in imgs_list:
        image = Image.open(i)
        transform_basic = transforms.Compose(base_transforms)
        img_tensor = transform_basic(image)
        to_infer_batch.append(img_tensor)
    to_infer_batch = torch.stack(to_infer_batch)
    return to_infer_batch

if __name__ == "__main__":
    generate_random_10_images(input_dir, to_infer_data_path, label_file)
    imgs_list = sorted(Path(to_infer_data_path).glob('*.png'))
    photos = []
    for i in imgs_list:
        j = imgs_list.index(i)
        to_predict = Path(imgs_list[j]).as_posix()
        photos.append(to_predict)
    to_infer_batch = transform_image(imgs_list)
    data = transform_image(imgs_list)
    model_onnx = get_onnx_model(os.getcwd()) 
    predictions_idx, predictions_label = infer_predict(model_onnx, data)
    image_and_prediction = zip(predictions_idx, predictions_label)
    write_to_yaml('predictions.yaml', predictions_label)
    print(*image_and_prediction, sep='\n')
    print('predictions are written to predictions.yaml successfully!')
