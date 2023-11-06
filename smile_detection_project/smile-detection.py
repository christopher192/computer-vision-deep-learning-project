import os
import sys
sys.path.append("..")
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import cv2
import imutils
from tensorflow.keras.preprocessing.image import img_to_array
from imutils import paths
import numpy as np
from tensorflow.keras.utils import to_categorical
from library.neural_network.conv_neural_network import LeNet
import matplotlib.pyplot as plt

image_data = []
image_label = []

for image_path in sorted(list(paths.list_images("../dataset/smile/SMILEs"))):
    image = cv2.imread(image_path)

    ### convert image channel from 3 to 1
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ### resize image data from (64, 64) to (28, 28)
    image = imutils.resize(image, width = 28)

    ### convert image data from (28, 28) to (28, 28, 1) 
    image = img_to_array(image)
    
    image_data.append(image)
    
    label = image_path.split(os.path.sep)[-3]
    label = "smiling" if label == "positives" else "not_smiling"
    
    image_label.append(label)

image_data = np.array(image_data, dtype = "float") / 255.0
image_label = np.array(image_label)

le = LabelEncoder().fit(image_label)
image_label = to_categorical(le.transform(image_label), num_classes = 2)

### initialize class weight for imbalanced dataset
total_class = image_label.sum(axis = 0)
class_weight = total_class.max() / total_class
class_weight = {i: w for i, w in enumerate(class_weight)}

### using stratify to deal with imbalanced dataset
### also ensure that the class distribution is preserved in both sets
(x_train, x_test, y_train, y_test) = train_test_split(image_data, image_label, test_size = 0.20, 
    stratify = image_label, random_state = 42)

print("INFO - compiling model...")

model = LeNet.build(28, 28, 1, 2)
model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])

print("INFO - model training...")

history = model.fit(x_train, y_train, validation_data = (x_test, y_test), class_weight = class_weight, batch_size = 64, 
    epochs = 15, verbose = 1)

print("INFO - model evaluation...")

prediction = model.predict(x_test, batch_size = 64)

print(classification_report(y_test.argmax(axis = 1), prediction.argmax(axis = 1), target_names = le.classes_))

### save model
print("INFO - model saving...")

model.save("saved_model/lenet-smile-detection.keras")

### training and validation accuracy/ loss plotting
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 15), history.history["loss"], label = "train loss")
plt.plot(np.arange(0, 15), history.history["val_loss"], label = "val loss")
plt.plot(np.arange(0, 15), history.history["accuracy"], label = "train accuracy")
plt.plot(np.arange(0, 15), history.history["val_accuracy"], label = "val accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Loss/ Accuracy")
plt.legend()
plt.show()