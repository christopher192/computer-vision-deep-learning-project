# COCO 8 - Instance Segmentation

## <ins>Introduction</ins>
This repository focuses on performing instance segmentation on a subset of the COCO dataset, specifically known as COCO 8. The COCO 8 dataset is chosen to expedite the development of other important features. Based on current state of implementation, the MLOps maturity of this repository stands at level 2. To delve deeper into the concept of MLOps maturity, please visit [Microsoft's Guide on MLOps Maturity Model](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/mlops-maturity-model).

## <ins>Technology/ Implementation</ins>
- The instance segmentation model is built using Torchvision's Mask RCNN FPN v2, leveraging pretrained weight.
- MLflow will be used to serve for experiment tracking. 
- Prefect for orchestration.

## <ins>MLflow</ins>
<ins>Start interface</ins>
<br>
Execute below command to interact with MLflow server.
<br>
```
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

<ins>Log/ save model</ins>
<br>
The difference between `mlflow.pytorch.log_model()` and `mlflow.pytorch.save_model()` lies in their usage. `save_model()` is used to save a PyTorch model to a specified local path, while `log_model()` is used to log a PyTorch model to the MLflow tracking server for further tracking and versioning.

<ins>System metric</ins>
<br>
```
pip install psutil
pip install pynvml # for GPU metric
```
```
os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
```

## <ins>Prefect</ins>
<ins>Start interface</ins>
<br>
Execute below command to interact with Prefect server.
<br>
```
prefect server start
```

<ins>Follow these steps to start orchestration</ins>
<br>
Create work pool.
<br>
```
prefect work-pool create --type process inst-seg
```

Initialize a local project.
<br>
```
prefect init --recipe local
```

Deploy workflow.
<br>
```
prefect deploy training.py:init_flow --name inst-seg-run --pool inst-seg
```

Workflow deployment.
<br>
```
prefect deployment run 'init-flow/inst-seg-run'
```

Start work pool.
<br>
```
prefect worker start --pool "inst-seg"
```

## <ins>Prefect Schedule Type</ins>
When deploying the flow, there are 3 types of schedules to choose from:

- `Cron`: This is a time-based job scheduler that uses cron expressions to schedule flows. For instance, a cron string of `"0 0 * * *"` schedules the flow to run daily at midnight.

- `Interval`: This allows for running a flow at regular intervals. By specifying the interval as a duration, Prefect runs the flow each time that duration passes. For instance, an interval schedule with a duration of one hour runs the flow every hour.

- `RRule`: This stands for "Recurrence Rule" and is a format for specifying recurring events. RRules can create more complex schedules such as calendar logic for simple recurring schedules, irregular intervals, exclusions, or day-of-month adjustments. For instance, an RRule can schedule a flow to run at 9am on every weekday.

## <ins>Instruction</ins>

## <ins>Result</ins>

## <ins>Issue/ Challenge</ins>
`mlflow==2.12.1` currently support `torch==2.1.2+cu118` and `torchvision==0.16.2+cu118`.

`mlflow.pytorch` appears to have compatibility with PyTorch Lightning instead of Torchvision, additional code for MLflow need to be worked with Torchvision.

## <ins>To-Do List</ins>
| No | Task                                              | Complete |
| --- | ------------------------------------------------- | ---- |
| 1 | Validation loss | &#10004; |
| 2 | COCO evaluation | &#10004; |
| 3 | Confusion matrix/ metric | &cross; |
| 4 | Precision recall curve | &cross; |
| 5 | Precision confidence curve | &cross; |
| 6 | Recall confidence curve | &cross; |
| 7 | F1 confidence curve | &cross; |
| 8 | Early stopping | &cross; |
| 9 | Fitness evaluation metric | &cross; |
| 10 | `Production` transition by comparing `Stagging` models | &cross; |
| 11 | Soft-dice Loss | &cross; |
| 12 | Dice coefficient metric | &cross; |