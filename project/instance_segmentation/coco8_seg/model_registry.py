import mlflow
from datetime import datetime
from prefect import flow, task
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

def get_experiment_id(client, name):
    experiments = client.search_experiments()
    for exp in experiments:
        if exp.name == name:
            return exp.experiment_id

@task
def register_model(experiment_name, mlflow_tracking_url, model_register_name):
    client = MlflowClient(tracking_uri = mlflow_tracking_url)
    experiment_id = get_experiment_id(client, experiment_name)

    top_3_lowest_training_loss = client.search_runs(
        experiment_ids = experiment_id,
        filter_string = "metrics.total_train_loss < 0.5 and tags.model = 'mask_rcnn_resnet50_fpn_v2' and attributes.status = 'FINISHED'",
        run_view_type = ViewType.ACTIVE_ONLY,
        max_results = 3,
        order_by = ["metrics.total_train_loss ASC"] 
    )

    run_id = top_3_lowest_training_loss[0].info.run_id
    model_uri = f"runs:/{run_id}/model"

    all_registered_run_ids = [mv.run_id for mv in client.search_model_versions(f"name='{model_register_name}'")]

    if run_id in all_registered_run_ids:
        print(f"{run_id} is already registered at {model_register_name}")
    else:
        # register
        # mlflow.set_tracking_uri(mlflow_tracking_url)
        # mlflow.set_experiment(experiment_name)

        # mlflow.register_model(
        #     model_uri = model_uri, 
        #     name = model_register_name
        # )
        client.create_model_version(
            name = model_register_name,
            source = model_uri, 
            run_id = run_id
        )

@task
def staging_transition(mlflow_tracking_url, model_register_name):
    client = MlflowClient(tracking_uri = mlflow_tracking_url)

    info_for_transition = [{'run_id': mv.run_id, 'name': mv.name, 'version': mv.version}
        for mv in client.search_model_versions(f"name='{model_register_name}'")
        if mv.current_stage == 'None']

    for info in info_for_transition:
        date = datetime.today().date()
        new_stage = "Staging"

        # transition the model version to the "Staging" stage
        client.transition_model_version_stage(
            name = info["name"],
            version = info["version"],
            stage = new_stage,
            archive_existing_versions = False
        )

        # update the model version description
        client.update_model_version(
            name = info["name"],
            version = info["version"],
            description = f"The model version {info['version']} was transitioned to {new_stage} on {date}"
        )

@flow
def init_flow():
    mlflow_tracking_url = "sqlite:///mlflow.db"
    experiment_name = "coco8-instance-seg"
    model_register_name = "coco8-inst-detection"

    register_model(experiment_name, mlflow_tracking_url, model_register_name)
    # transition 'None' to 'Staging' for experiment
    staging_transition(mlflow_tracking_url, model_register_name)

if __name__ == "__main__":
    init_flow()