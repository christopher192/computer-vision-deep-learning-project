import os
import gc
import torch
import shutil
import mlflow
import datetime
from utils import training_loader, validation_loader
from utils import load_maskrcnn_resnet50_fpn_v2, get_iou_types
from utils import evaluation_loss_map, save_loss_curve
from prefect import flow, task

os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"

@task
def load_dataset():
    load_training_loader, load_training_dataset = training_loader(
        rootPath = "../../../dataset/coco8-seg/train/images", 
        annFilePath = "../../../dataset/coco8-seg/train/annotation/_annotations.coco.json"
    )
    load_validation_loader, load_validation_dataset = validation_loader(
        rootPath = "../../../dataset/coco8-seg/valid/images", 
        annFilePath = "../../../dataset/coco8-seg/valid/annotation/_annotations.coco.json"
    )
    return load_training_loader, load_training_dataset, load_validation_loader, load_validation_dataset

@task
def train_model(l_train_loader, l_train_dataset, l_val_loader, l_val_dataset):
    mlflow_tracking_url = "sqlite:///mlflow.db"
    mlflow.set_tracking_uri(mlflow_tracking_url)
    mlflow.set_experiment("coco8-instance-seg")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_maskrcnn_resnet50_fpn_v2()
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    learning_rate, momentum, weight_decay = 0.005, 0.9, 0.0005
    step_size, gamma = 3, 0.1

    unfreeze_all_layer = True

    optimizer = torch.optim.SGD(params, lr = learning_rate, momentum = momentum, weight_decay = weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = step_size, gamma = gamma)

    if unfreeze_all_layer:
        for param in model.parameters():
            param.requires_grad = True

    num_epochs = 15
    selected_keys = ['boxes', 'labels', 'masks']
    iou_types = get_iou_types(model)
    cpu_device = torch.device("cpu")

    result = {
        "total_train_loss": [],
        "total_val_loss": [],
        "total_train_loss_classifier": [],
        "total_val_loss_classifier": [],
        "total_train_loss_box_reg": [],
        "total_val_loss_box_reg": [],
        "total_train_loss_mask": [],
        "total_val_loss_mask": [],
        "total_train_loss_objectness": [],
        "total_val_loss_objectness": [],
        "total_train_loss_rpn_box_reg": [],
        "total_val_loss_rpn_box_reg": []
    }

    pip_requirements = [
        'torch==2.1.2+cu118',
        'torchvision==0.16.2+cu118'
    ]

    mlflow.pytorch.autolog(disable = False)

    with mlflow.start_run() as run:
        mlflow.set_tag("developer", "christopher")
        mlflow.set_tag("model", "mask_rcnn_resnet50_fpn_v2")
        mlflow.set_tag("version", mlflow.__version__)

        mlflow.log_param("train-data-path", l_train_dataset.root)
        mlflow.log_param("valid-data-path", l_val_dataset.root)
        mlflow.log_param("train-annotation-path", l_train_dataset.annFile)
        mlflow.log_param("valid-annotation-path", l_val_dataset.annFile)

        mlflow.log_param("optimizer", type(optimizer).__name__)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("momentum", momentum)
        mlflow.log_param("weight_decay", weight_decay)

        mlflow.log_param("scheduler", type(lr_scheduler).__name__)
        mlflow.log_param("step_size", step_size)
        mlflow.log_param("gamma", gamma)

        mlflow.log_param("unfreeze_all_layer", unfreeze_all_layer)

        for epoch in range(num_epochs):
            model.train()

            train_loss = 0.0
            train_loss_classifier = 0.0
            train_loss_box_reg = 0.0
            train_loss_mask = 0.0
            train_loss_objectness = 0.0
            train_loss_rpn_box_reg = 0.0

            for batch_idx, (images, targets) in enumerate(l_train_loader):
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items() if k in selected_keys} for t in targets]

                loss_dict = model(images, targets)
                loss = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_loss_classifier += loss_dict["loss_classifier"].item()
                train_loss_box_reg += loss_dict["loss_box_reg"].item()
                train_loss_mask += loss_dict["loss_mask"].item()
                train_loss_objectness += loss_dict["loss_objectness"].item()
                train_loss_rpn_box_reg += loss_dict["loss_rpn_box_reg"].item()

                del images, targets, loss_dict
                gc.collect()

            val_loss_result = evaluation_loss_map(model, l_val_dataset, l_val_loader, 
                selected_keys, device, iou_types, cpu_device)

            torch.cuda.empty_cache()
            gc.collect()

            total_train_loss = train_loss / len(l_train_loader)
            total_train_loss_classifier = train_loss_classifier / len(l_train_loader)
            total_train_loss_box_reg = train_loss_box_reg / len(l_train_loader)
            total_train_loss_mask = train_loss_mask / len(l_train_loader)
            total_train_loss_objectness = train_loss_objectness / len(l_train_loader)
            total_train_loss_rpn_box_reg = train_loss_rpn_box_reg / len(l_train_loader)

            # log training loss
            mlflow.log_metric("total_train_loss", total_train_loss, step = epoch)
            mlflow.log_metric("train_loss_classifier", total_train_loss_classifier, step = epoch)
            mlflow.log_metric("train_loss_box_reg", total_train_loss_box_reg, step = epoch)
            mlflow.log_metric("train_loss_mask", total_train_loss_mask, step = epoch)
            mlflow.log_metric("train_loss_objectness", total_train_loss_objectness, step = epoch)
            mlflow.log_metric("train_loss_rpn_box_reg", total_train_loss_rpn_box_reg, step = epoch)

            # log validation loss
            mlflow.log_metric("total_val_loss", val_loss_result["total_val_loss"], step = epoch)
            mlflow.log_metric("val_loss_classifier", val_loss_result["total_val_loss_classifier"], step = epoch)
            mlflow.log_metric("val_loss_box_reg", val_loss_result["total_val_loss_box_reg"], step = epoch)
            mlflow.log_metric("val_loss_mask", val_loss_result["total_val_loss_mask"], step = epoch)
            mlflow.log_metric("val_loss_objectness", val_loss_result["total_val_loss_objectness"], step = epoch)
            mlflow.log_metric("val_loss_rpn_box_reg", val_loss_result["total_val_loss_rpn_box_reg"], step = epoch)

            result["total_train_loss"].append(total_train_loss)
            result["total_train_loss_classifier"].append(total_train_loss_classifier)
            result["total_train_loss_box_reg"].append(total_train_loss_box_reg)
            result["total_train_loss_mask"].append(total_train_loss_mask)
            result["total_train_loss_objectness"].append(total_train_loss_objectness)
            result["total_train_loss_rpn_box_reg"].append(total_train_loss_rpn_box_reg)

            result["total_val_loss"].append(val_loss_result["total_val_loss"])
            result["total_val_loss_classifier"].append(val_loss_result["total_val_loss_classifier"])
            result["total_val_loss_box_reg"].append(val_loss_result["total_val_loss_box_reg"])
            result["total_val_loss_mask"].append(val_loss_result["total_val_loss_mask"])
            result["total_val_loss_objectness"].append(val_loss_result["total_val_loss_objectness"])
            result["total_val_loss_rpn_box_reg"].append(val_loss_result["total_val_loss_rpn_box_reg"])

            base_dir = "model_checkpoint"
            checkpoint_path = f"{base_dir}/epoch_{epoch + 1}"
            mlflow.pytorch.log_model(model, checkpoint_path, pip_requirements = pip_requirements)

            print(f'Epoch {epoch + 1}, Loss: {total_train_loss}')
            print(f'Max allocated memory: {torch.cuda.max_memory_allocated(device) / 1024 ** 2} MB, Allocated memory: {torch.cuda.memory_allocated(device) / 1024 ** 2} MB')
            print(f'Max reserved memory: {torch.cuda.max_memory_reserved(device) / 1024 ** 2} MB, Reserved memory: {torch.cuda.memory_reserved(device) / 1024 ** 2} MB')
            print("")

        model_path = "model"
        scripted_model_path = "scripted_model"

        for path in [model_path, scripted_model_path]:
            shutil.rmtree(path, ignore_errors = True)

        mlflow.pytorch.log_model(model, model_path, pip_requirements = pip_requirements)
        mlflow.pytorch.save_model(model, model_path, pip_requirements = pip_requirements)

        scripted_model = torch.jit.script(model)
        mlflow.pytorch.log_model(scripted_model, scripted_model_path, pip_requirements = pip_requirements)
        mlflow.pytorch.save_model(scripted_model, scripted_model_path, pip_requirements = pip_requirements)

        now = datetime.now()
        dir_name = now.strftime("%Y-%m-%d_%H-%M-%S")
        save_loss_curve(result, dir_name)

        # log artifact for loss curve
        mlflow.log_artifact(f"result/{dir_name}/loss_plot.png", artifact_path = "loss_plot")

@flow
def init_flow():
    l_train_loader, l_train_dataset, l_val_loader, l_val_dataset = load_dataset()
    train_model(l_train_loader, l_train_dataset, l_val_loader, l_val_dataset)

if __name__ == "__main__":
    init_flow()