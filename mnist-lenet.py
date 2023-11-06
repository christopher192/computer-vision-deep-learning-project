import tensorflow as tf
from library.neural_network.conv_neural_network import LeNet
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras import backend
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.optimizers import SGD
import numpy as np
from sklearn.metrics import classification_report

### data loading and preprocessing
print("INFO - accessing mnist dataset...")

data = mnist.load_data()

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)

### initializing the optimizer and model
print("INFO - compiling model...")

width, height, channel = 28, 28, 1
class_number = 10 

optimizer = SGD(learning_rate = 0.01)

model = LeNet.build(width, height, channel, class_number)
model.compile(loss = "categorical_crossentropy", optimizer = optimizer, metrics = ["accuracy"])

### model training
history = model.fit(x_train, y_train, validation_data = (x_test, y_test), batch_size = 128, epochs = 20, verbose = 1)

### model evaluation
print("INFO - evaluating model...")

prediction = model.predict(x_test, batch_size = 128)

print(classification_report(y_test.argmax(axis = 1), prediction.argmax(axis = 1), 
    target_names = [str(x) for x in lb.classes_]))

### training and validation accuracy/ loss plotting
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 20), history.history["loss"], label = "train loss")
plt.plot(np.arange(0, 20), history.history["val_loss"], label = "val loss")
plt.plot(np.arange(0, 20), history.history["accuracy"], label = "train accuracy")
plt.plot(np.arange(0, 20), history.history["val_accuracy"], label = "val accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Loss/ Accuracy")
plt.legend()
plt.show()