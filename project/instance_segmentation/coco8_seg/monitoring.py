import os
import json
import click
import random
import psycopg
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from datetime import datetime

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
        timestamp timestamp,
        image_name VARCHAR(255),
        image_base64 TEXT,
        image_metadata JSONB
    )
"""
host = "localhost"
port = "5432"
user = "postgres"
password = "Password123"
database = "cvops"

def prep_database():
    with psycopg.connect(f"host = {host} port = {port} user = {user} password = {password}", autocommit = True) as conn:
        res = conn.execute(f"SELECT 1 FROM pg_database WHERE datname = '{database}'")
        if len(res.fetchall()) == 0:
            conn.execute("CREATE DATABASE cvops;")
        with psycopg.connect(f"host = {host} port = {port} dbname = {database} user = {user} password = {password}") as conn:
            conn.execute(create_table_statement)

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

def insert_data(image_name, image_base64, image_metadata):
    with psycopg.connect(f"host = {host} port = {port}  dbname = {database} user = {user} password = {password}", autocommit = True) as conn:
        with conn.cursor() as curr:
            curr.execute("INSERT INTO cv_metrics (timestamp, image_name, image_base64, image_metadata) VALUES (%s, %s, %s, %s)", (datetime.now(), image_name, image_base64, json.dumps(image_metadata)))

@click.command()
@click.option("--inference_datapath", default = "../../../dataset/coco8-seg/valid/images", help = "location where inference data is stored")
def initialize(inference_datapath):
    prep_database()
    random_image = random.sample(os.listdir(inference_datapath), 1)
    image_name = random_image[0]
    image_metadata = get_image_metadata(image_path = os.path.join(inference_datapath, random_image[0]))
    insert_data(image_name = image_name, image_base64 = image_metadata["image_base64"], image_metadata = {k: v for k, v in image_metadata.items() if k != 'image_base64'})

if __name__ == '__main__':
    initialize()