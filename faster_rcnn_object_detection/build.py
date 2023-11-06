import os
import sys
sys.path.append("..")
from config import config
from sklearn.model_selection import train_test_split
from PIL import Image
import tensorflow as tf
from library.utils.tf_annotation import TFAnnotation

f = open(config.classes_file, "w")
print(f)