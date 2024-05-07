# COCO 8 - Instance Segmentation

## <ins>Introduction</ins>
This repository focuses on performing instance segmentation on a subset of the COCO dataset, specifically known as COCO 8. The COCO 8 dataset is chosen to expedite the development of other important features. 

## <ins>Technology/ Implementation</ins>
- The instance segmentation model is built using Torchvision's Mask RCNN FPN v2, leveraging pretrained weights.
- MLflow and Prefect will be used to serve for experiment tracking and orchestration.

## <ins>Leverage MLflow</ins>
<ins>Command-line Interface</ins>
<br>
Execute below command to interacting with MLflow.
<br>
`mlflow ui --backend-store-uri sqlite:///mlflow.db`

<ins>Log/ Save Model</ins>
<br>
The difference between `mlflow.pytorch.log_model()` and `mlflow.pytorch.save_model()` lies in their usage. `save_model()` is used to save a PyTorch model to a specified path, while `log_model()` is used to log a PyTorch model to the MLflow tracking server for further tracking and versioning.

## <ins>Issue/ Challenge</ins>
`mlflow==2.12.1` currently support `torch==2.1.2+cu118` and `torchvision==0.16.2+cu118`.

`mlflow.pytorch` appears to have compatibility with PyTorch Lightning instead of Torchvision, additional code for MLflow need to be worked with Torchvision.