import pandas as pd

pd.set_option('display.max_columns', None)
import joblib
import boto3
import json

s3 = boto3.client('s3')

import os
import io

import datetime
from argparse import ArgumentParser, Namespace
from importlib import import_module
from pathlib import Path
from typing import Optional, Tuple, Dict
import pickle
import numpy as np
from anomalib.deploy import Inferencer

DEFAULT_CONFIG_PATH = Path("app/config.yaml")
#DEFAULT_WEIGHT_PATH = Path("results/padim/mvtec/bottle/weights/model.ckpt")
DEFAULT_WEIGHT_PATH = Path("app/model.bin")
DEFAULT_META_DATA_path = Path("app/meta_data.json")

def get_inferencer(config_path = DEFAULT_CONFIG_PATH , weight_path = DEFAULT_WEIGHT_PATH, meta_data_path = DEFAULT_META_DATA_path):
    """Parse args and open inferencer.
    Args:
        config_path (Path): Path to model configuration file or the name of the model.
        weight_path (Path): Path to model weights.
        meta_data_path (Optional[Path], optional): Metadata is required for OpenVINO models. Defaults to None.
    Raises:
        ValueError: If unsupported model weight is passed.
    Returns:
        Inferencer: Torch or OpenVINO inferencer.
    """

    # Get the inferencer. We use .ckpt extension for Torch models and (onnx, bin)
    # for the openvino models.
    extension = weight_path.suffix
    module = import_module("anomalib.deploy")
    if extension in (".ckpt"):
        torch_inferencer = getattr(module, "TorchInferencer")
        inferencer = torch_inferencer(config=config_path, model_source=weight_path, meta_data_path=meta_data_path)

    elif extension in (".onnx", ".bin", ".xml"):
        openvino_inferencer = getattr(module, "OpenVINOInferencer")
        inferencer = openvino_inferencer(config=config_path, path=weight_path, meta_data_path=meta_data_path)

    else:
        raise ValueError(
            f"Model extension is not supported. Torch Inferencer exptects a .ckpt file,"
            f"OpenVINO Inferencer expects either .onnx, .bin or .xml file. Got {extension}"
        )
    return inferencer

def infer(image, inferencer):
    """Inference function, return anomaly map, score, heat map, prediction mask ans visualisation.
    Args:
        image (np.ndarray): image to compute
        inferencer (Inferencer): model inferencer
    Returns:
        Tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
        heat_map, pred_mask, segmentation result.
    """
    # Perform inference for the given image.
    predictions = inferencer.predict(image=image)
    return (predictions.heat_map, predictions.pred_mask, predictions.segmentations)


def read_s3_image(bucket,key):
    """Read image array in s3.
    bucket[str]:fsdl-anomalib
    key[str]:test_image_{key_name}.pkl
    """
    s3 = boto3.resource('s3')
    print(bucket,key)
    with io.BytesIO() as data:
        s3.Bucket(bucket).download_fileobj(key, data)
        data.seek(0)  # move back to the beginning after writing
        image = pickle.load(data)
    print(f'file loaded')
    return image

def save_s3_prediction(bucket,key,predictions):
    """SAVE prediction produced by AWS Lambda.
    Prediction[dict]: '
        - 'heat_map': predictions.heat_map,
        - 'red_mask': predictions.pred_mask
        - 'segmentations': predictions.segmentations
    bucket[str]:fsdl-anomalib-prediction
    key[str]:test_image_prediction_{key_name}.pkl
    """
    s3 = boto3.resource('s3')
    
    with io.BytesIO() as pred_io:
        pickle.dump(predictions, pred_io)
        pred_io.seek(0)
        s3.Bucket(bucket).put_object(Key=key, Body=pred_io)
    #pickle.dump(predictions, pred_io)
    #pred_io.seek(0)
    #s3.Bucket(bucket).upload_fileobj(pred_io, key)
    print(f'file uploaded')
    return True


def handler(event, context):
    """
    AWS Lambda handler function: https://docs.aws.amazon.com/lambda/latest/dg/python-handler.html
    Triggered by s3 file upload 
    https://stackoverflow.com/questions/43987732/aws-lambda-and-multipart-upload-to-from-s3
    """
    print("-- Running ML --")

    # s3 bucket
    bucket = event['Records'][0]['s3']['bucket']['name']
    # key = filename = s3 path
    key = event['Records'][0]['s3']['object']['key']
    print(bucket,key)
    # load the data
    image = read_s3_image(bucket,key)
    print('Image Loaded')
    # setup inferencer
    print('../:',os.listdir('../'))
    print('.:',os.listdir('.'))
    print('app/',os.listdir('app/'))
    gradio_inferencer = get_inferencer(DEFAULT_CONFIG_PATH, DEFAULT_WEIGHT_PATH, DEFAULT_META_DATA_path)

    heat_map, pred_mask, segmentations = infer(image, gradio_inferencer)
    print('Set up inferencer')
    
    # save the results
    key_pred = key.replace('test_image','test_image_prediction')
    predictions = dict(zip(['heat_map','red_mask','segmentations'],[heat_map, pred_mask, segmentations]))
    save_s3_prediction('fsdl-anomalib-prediction',key_pred,predictions)
    print('Prediction Saved')
    return True

if __name__ == '__main__':
    print('Running locally')


