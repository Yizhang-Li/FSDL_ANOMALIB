# Deploy Anomalib to AWS (Model)

## Setup

Make sure you have [conda](https://docs.conda.io/en/latest/miniconda.html) installed. Then, install the requirements with `bash install_requirements.sh`.

## Prepare MVTEC Data
`python datasets/prepare_data.py`

## Download pre-trained Padim models
Unzip Padim models (including meta-data) to the folder `results/`.

You need to put the following files in the folder `results/`:
- `results/padim/mvtec/bottle/openvino/model.bin`
- `results/padim/mvtec/bottle/openvino/meta_data.json`

## Gradio Inference (Local)
`python tools/inference/gradio_inference.py --config tools/config/padim/config.yaml --weights results/padim/mvtec/bottle/openvino/model.bin --meta_data results/padim/mvtec/bottle/openvino/meta_data.json`

## Deployment

General steps to deploy your code include

1.  Build your machine learning model and pipeline
2.  Create/setup a AWS account
3.  Package your code in a Docker container
4.  Upload your Docker image to AWS Elastic Container Registry (ECR)
5.  Create your AWS Lambda to run the ECR image
6.  Run/test/configure your AWS Lambda
7.  Deliver your results to others who may need the results


![Deployment Process](images/diagram2.png)