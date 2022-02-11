from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import Flatten, Dense, Reshape, Dropout, Activation
from keras.layers import SpatialDropout2D
from keras.regularizers import l1
from keras.models import Model

def create_model():
    _input_img = Input(shape=(28, 28, 1))
    _x = Conv2D(10, (5, 5))(_input_img)
    _x = MaxPooling2D((2, 2))(_x)
    _x = Activation('relu')(_x)
    _x = Conv2D(20, (5, 5))(_x)
    _x = SpatialDropout2D(0.5)(_x)
    _x = MaxPooling2D((2, 2))(_x)
    _x = Activation('relu')(_x)
    _x = Flatten()(_x)
    _x = Dense(50)(_x)
    _x = Activation('relu')(_x)
    _x = Dropout(0.5)(_x)
    _x = Dense(10)(_x)
    _x = Activation('softmax')(_x)

    return Model(_input_img, _x)
