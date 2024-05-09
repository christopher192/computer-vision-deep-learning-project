import mlflow
from prefect import flow, task

@flow
def start_training():
    print("Hello World")

if __name__ == "__main__":
    start_training()