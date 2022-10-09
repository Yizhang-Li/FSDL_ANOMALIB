"""
Gradio Anomaly Inference application.
This module provides a gradio-based web application
for the anomaly detection project.
"""
import json
import boto3
s3 = boto3.client('s3')
import gradio as gr
import io
from argparse import ArgumentParser, Namespace
from importlib import import_module
from pathlib import Path
from typing import Optional, Tuple
import gradio.inputs
import gradio.outputs
import datetime
import numpy as np
import pickle
import uuid
import time
#from anomalib.deploy import Inferencer
from botocore.errorfactory import ClientError
from datetime import timedelta

title = "Anomaly Detection"
description = """
<h2> Description </h2>
Anomalib is 
The positivity is measured in five categories:
- Extremely negative
- Negative
- Neutral
- Positive
- Extremely positive
The application is based on a PADIM model fine tuned on the following:
[dataset](https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification).
"""

article = "Check out this \
[repository](https://github.com/hectorLop/Twitter_Positivity_Analyzer) \
with a lot more details about this method and implementation."

def save_s3_image(image_array, bucket='fsdl-anomalib'):
    """Save Image Array to AWS Lambda.
    Trigger Prefix: [TODO]
    Suffix: .pkl
    Prediction Name: test_image_prediction_{uuid}.pkl
    """
    key_name = uuid.uuid4().hex
    key = f'test_image_{key_name}.pkl'
    pred_key = f'test_image_prediction_{key_name}.pkl'
    s3_client = boto3.client('s3')
    np_buffer = io.BytesIO()
    pickle.dump(image_array, np_buffer)
    np_buffer.seek(0)
    s3_client.upload_fileobj(np_buffer, bucket, key)
    #s3_client.put_object(Bucket=bucket, Key=key, Body=csv_buffer.getvalue())
    print(f'file written to {bucket} --{key}')
    return pred_key

def read_s3_prediction(bucket,key,timeout = 120):
    """Read prediction produced by AWS Lambda.
    Prediction[dict]: '
        - 'heat_map': predictions.heat_map,
        - 'red_mask': predictions.pred_mask
        - 'segmentations': predictions.segmentations
    Trigger Prefix: image-new-test-prediction/
    Suffix: .pkl
    bucket[str]:fsdl-anomalib-prediction
    key[str]:test_image_prediction_{key_name}.pkl
    """
    s3 = boto3.client('s3')
    exist_flag = False
    wait_until = datetime.now() + timedelta(seconds = timeout)
    while not exist_flag:
        try:
            s3.head_object(bucket, key)
            exist_flag = True
        except ClientError:
            time.sleep(1)
            pass

        if wait_until < datetime.now():
            exist_flag = True
            return {'heat_map':np.empty([16,16]),
                    'red_mask':np.empty([16,16]),
                   'segmentations': np.empty([16,16])}

    print(f'file found!')
    with io.BytesIO() as data:
        s3.Bucket(bucket).download_fileobj(key, data)
        data.seek(0)  # move back to the beginning after writing
        predictions = pickle.load(data)
    print(f'file loaded')
    return predictions


def infer_image(image):
    """Send image to AWS Lambda's Inference function, return anomaly map, score, heat map, prediction mask ans visualisation.
    Args:
        image (np.ndarray): image to compute
    Returns:
        Tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
        heat_map, pred_mask, segmentation result.
    """
    #text = None

    #payload = {"text": text}

    #session = boto3.Session()
    #lambda_client = session.client("lambda")
    #response = lambda_client.invoke(
    #    FunctionName="twitter-analyzer-lambda",
    #    InvocationType="RequestResponse",
    #    Payload=json.dumps(payload),
    #)

    #response = json.loads(response["Payload"].read().decode())
    #response = json.loads(response["body"])
    #outcome = response["label"]

    # Upload image to s3 bucket
    pred_key = save_s3_image(image, bucket='fsdl-anomalib')

    # trigger AWS Lambda

    # Weight until prediction is made
    predictions = read_s3_prediction('fsdl-anomalib-prediction', pred_key, timeout=120)

    return (predictions.heat_map, predictions.pred_mask, predictions.segmentations)

if __name__ == "__main__":
    gradio.close_all()
    app = gr.Interface(
        fn = lambda image: infer_image(image),
        inputs=[
            gradio.inputs.Image(
                shape=None, image_mode="RGB", source="upload", tool="editor", type="numpy", label="Image"
            ),
        ],
        outputs=[
            gradio.outputs.Image(type="numpy", label="Predicted Heat Map"),
            gradio.outputs.Image(type="numpy", label="Predicted Mask"),
            gradio.outputs.Image(type="numpy", label="Segmentation Result"),
        ],
        title=title,
        description=description,
        article=article,
    )
    app, local_url, share_url = app.launch(server_port=8500, server_name="0.0.0.0")
    print(share_url)
    # fuser 8500/tcp