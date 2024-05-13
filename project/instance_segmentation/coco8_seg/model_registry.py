from mlflow.tracking import MlflowClient

def get_experiment_id(client, name):
    experiments = client.search_experiments()

    for exp in experiments:
        if exp.name == name:
            return exp.experiment_id

if __name__ == "__main__":
    mlflow_tracking_url = mlflow_tracking_url = "sqlite:///mlflow.db"

    client = MlflowClient(tracking_uri = mlflow_tracking_url)
    experiment_id = get_experiment_id(client, 'coco8-instance-seg')

    print(experiment_id)