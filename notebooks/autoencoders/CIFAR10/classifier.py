from keras.layers import Input, Conv2D, MaxPool2D, Dense
from keras.layers import Activation, Flatten, Dropout, BatchNormalization
from keras import regularizers
from keras.models import Model

def create_model():
    image = Input(shape=(32, 32, 3))

    x = Conv2D(64, 3, padding='same', kernel_regularizer=regularizers.l2(5e-4))(image)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Conv2D(64, 3, padding='same', kernel_regularizer=regularizers.l2(5e-4))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(padding='same')(x)

    x = Conv2D(128, 3, padding='same', kernel_regularizer=regularizers.l2(5e-4))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Conv2D(128, 3, padding='same', kernel_regularizer=regularizers.l2(5e-4))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(padding='same')(x)

    x = Conv2D(256, 3, padding='same', kernel_regularizer=regularizers.l2(5e-4))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Conv2D(256, 3, padding='same', kernel_regularizer=regularizers.l2(5e-4))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(padding='same')(x)

    x = Conv2D(512, 3, padding='same', kernel_regularizer=regularizers.l2(5e-4))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Conv2D(512, 3, padding='same', kernel_regularizer=regularizers.l2(5e-4))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(padding='same')(x)

    x = Conv2D(512, 3, padding='same', kernel_regularizer=regularizers.l2(5e-4))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Conv2D(512, 3, padding='same', kernel_regularizer=regularizers.l2(5e-4))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(padding='same')(x)

    x = Flatten()(x)

    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = BatchNormalization()(x)

    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = BatchNormalization()(x)

    x = Dense(10)(x)

    x = Activation('softmax')(x)

    classifier = Model(inputs=image, outputs=x)

    return Model(image, x)
