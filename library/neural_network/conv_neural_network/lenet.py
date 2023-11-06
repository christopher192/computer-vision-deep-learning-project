#import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense
from tensorflow.keras import backend

class LeNet:
    @staticmethod
    def build(width, height, channel, class_number):
        model = Sequential()
        input_shape = (height, width, channel)

        if backend.image_data_format() == "channels_first":
            input_shape = (channel, height, width)

        ### first layer conv layer => relu layer => pool layer
        model.add(Conv2D(20, (5, 5), padding = "same", input_shape = input_shape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

        ### second layer conv layer => relu layer => pool layer
        model.add(Conv2D(50, (5, 5), padding = "same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

        ### fully connected layer with with 500 units => relu layer
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        ### output layer with softmax activation
        model.add(Dense(class_number))
        model.add(Activation("softmax"))

        return model