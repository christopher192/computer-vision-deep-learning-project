import gc
import torch
import shutil
import mlflow
from utils import training_loader, validation_loader
from utils import load_maskrcnn_resnet50_fpn_v2
from prefect import flow, task

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

    selected_keys = ['boxes', 'labels', 'masks']
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

        for epoch in range(5):
            model.train()
            train_loss = 0.0

            for batch_idx, (images, targets) in enumerate(l_train_loader):
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items() if k in selected_keys} for t in targets]

                loss_dict = model(images, targets)
                loss = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                del images, targets, loss_dict
                gc.collect()

            torch.cuda.empty_cache()
            gc.collect()

            total_train_loss = train_loss / len(l_train_loader)

            mlflow.log_metric("total_train_loss", total_train_loss, step = (epoch + 1))

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

@flow
def init_flow():
    l_train_loader, l_train_dataset, l_val_loader, l_val_dataset = load_dataset()
    train_model(l_train_loader, l_train_dataset, l_val_loader, l_val_dataset)

if __name__ == "__main__":
    init_flow()