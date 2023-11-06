#import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras import backend

class MiniVGGNet:
    @staticmethod
    def build(width, height, channel, class_number):
        model = Sequential()
        input_shape = (height, width, channel)
        channel_dimension = -1

        if backend.image_data_format() == "channels_first":
            input_shape = (channel, height, width)
            channel_dimension = 1
        
        ### first layer conv layer => relu layer => conv layer => relu layer => pool layer
        model.add(Conv2D(32, (3, 3), padding="same", input_shape = input_shape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis = channel_dimension))
        model.add(Conv2D(32, (3, 3), padding = "same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis = channel_dimension))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Dropout(0.25))

        ### second layer conv layer => relu layer => conv layer => relu layer => pool layer
        model.add(Conv2D(64, (3, 3), padding = "same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis = channel_dimension))
        model.add(Conv2D(64, (3, 3), padding = "same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis = channel_dimension))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Dropout(0.25))

        ### fully connected layer with with 512 units => relu layer
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        ### output layer with softmax activation
        model.add(Dense(class_number))
        model.add(Activation("softmax"))

        return model