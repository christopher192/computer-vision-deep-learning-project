import os
import json
import click
import time
import mlflow
import random
import psycopg
import base64
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from datetime import datetime
from torchvision.transforms import v2
from mlflow.tracking import MlflowClient

# create_table_statement = """
#     DROP TABLE IF EXISTS cv_metrics;
#     CREATE TABLE cv_metrics(
#         timestamp timestamp,
#         image_name VARCHAR(255),
#         image_base64 TEXT,
#         image_metadata JSONB
#     )
# """

create_table_statement = """
    CREATE TABLE IF NOT EXISTS cv_metrics (
        timestamp timestamp NOT NULL,
        image_name VARCHAR(255),
        image_base64 TEXT,
        image_metadata JSONB,
        inference_time FLOAT,
        prediction_result JSON
    )
"""
create_timescaledb_table_statement = """
    CREATE TABLE IF NOT EXISTS cv_metrics_timescaledb (
        timestamp TIMESTAMPTZ NOT NULL,
        image_name VARCHAR(255),
        image_base64 TEXT,
        image_metadata JSONB,
        inference_time FLOAT,
        prediction_result JSON
    )
"""

host = "localhost"
port = "5432"
user = "postgres"
password = "Password123"
database = "cvops"

mlflow_tracking_url = "sqlite:///mlflow.db"
model_register_name = "coco8-inst-detection"
stage = "Production"

def prep_database():
    # connect to the default database (postgres) to check for the cvops database
    with psycopg.connect(f"host = {host} port = {port} user = {user} password = {password}", autocommit = True) as conn:
        res = conn.execute(f"SELECT 1 FROM pg_database WHERE datname = '{database}'")
        if len(res.fetchall()) == 0:
            conn.execute(f"CREATE DATABASE {database};")
        # connect to the cvops database to create the timescaledb extension
        with psycopg.connect(f"host = {host} port = {port} dbname = '{database}' user = {user} password = {password}", autocommit = True) as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")
        # connect to the cvops database to create the cv_metrics and cv_metrics_timescaledb table
        with psycopg.connect(f"host = {host} port = {port} dbname = {database} user = {user} password = {password}") as conn:
            # cv_metrics table
            conn.execute(create_table_statement)
            # cv_metrics_timescaledb table
            conn.execute(create_timescaledb_table_statement)

def get_image_metadata(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
        img_format = img.format
        img = img.convert('RGB')
        img_data = np.array(img)

        # convert the image to base64
        buffered = BytesIO()
        img.save(buffered, format = img_format)
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

    ratio = width / height
    area = width * height

    return {"width": width, "height": height, "ratio": ratio, "area": area, "image_base64": img_base64}

def get_production_run_id():
    client = MlflowClient(tracking_uri = mlflow_tracking_url)
    run_id = None

    for rm in client.search_model_versions(f"name='{model_register_name}'"):
        registered_model = dict(rm)

        if registered_model["current_stage"] == stage:
            run_id = registered_model["run_id"]
            return run_id

    return run_id

def predict_image(image_path, run_id):
    mlflow.set_tracking_uri(mlflow_tracking_url)
    model_path = f"runs:/{run_id}/model"
    loaded_model = mlflow.pytorch.load_model(model_path)

    device = next(loaded_model.parameters()).device
    device = torch.device(device)

    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale = True),
        v2.ToDtype(torch.float32, scale = True),
    ])

    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)

    result = loaded_model(image)

    return result

def insert_data(image_name, image_base64, image_metadata, inference_time, prediction_result):
    with psycopg.connect(f"host = {host} port = {port}  dbname = {database} user = {user} password = {password}", autocommit = True) as conn:
        with conn.cursor() as curr:
            timestamp = datetime.now()
            curr.execute("INSERT INTO cv_metrics (timestamp, image_name, image_base64, image_metadata, inference_time, prediction_result) VALUES (%s, %s, %s, %s, %s, %s)", (timestamp, image_name, image_base64, json.dumps(image_metadata), inference_time, json.dumps(prediction_result)))
            curr.execute("INSERT INTO cv_metrics_timescaledb (timestamp, image_name, image_base64, image_metadata, inference_time, prediction_result) VALUES (%s, %s, %s, %s, %s, %s)", (timestamp, image_name, image_base64, json.dumps(image_metadata), inference_time, json.dumps(prediction_result)))

@click.command()
@click.option("--inference_datapath", default = "../../../dataset/coco8-seg/valid/images", help = "location where inference data is stored")
def initialize(inference_datapath):
    prep_database()

    for i in range(10):
        random_image = random.sample(os.listdir(inference_datapath), 1)
        image_name = random_image[0]
        image_metadata = get_image_metadata(image_path = os.path.join(inference_datapath, random_image[0]))

        run_id = get_production_run_id()

        if run_id is None:
            raise Exception("No model in production")

        start_time = time.time()
        result = predict_image(image_path = os.path.join(inference_datapath, random_image[0]), run_id = run_id)
        inference_time = time.time() - start_time

        # convert prediction to json for data insertion
        boxes_json =  json.dumps(result[0]["boxes"].tolist())
        masks_json =  json.dumps(result[0]["masks"].tolist())
        labels_json =  json.dumps(result[0]["labels"].tolist())
        scores_json =  json.dumps(result[0]["scores"].tolist())

        # combine into a single dictionary
        combined_data = {
            "boxes": boxes_json,
            "masks": masks_json,
            "labels": labels_json,
            "scores": scores_json
        }

        insert_data(image_name = image_name, image_base64 = image_metadata["image_base64"], 
            image_metadata = {k: v for k, v in image_metadata.items() if k != 'image_base64'},
            inference_time = inference_time, prediction_result = combined_data
        )

if __name__ == '__main__':
    initialize()