from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, concatenate, BatchNormalization
from config.config import IMG_SIZE

def create_model():
    input_l = Input(shape= IMG_SIZE+(1,))

    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_l)
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same', strides=2)(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c1)
    c2 = BatchNormalization()(c2)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same', strides=2)(c2)

    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c2)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same', strides=2)(c3)

    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c3)
    c4 = BatchNormalization()(c4)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same', strides=2)(c4)  # Bottleneck

    u1 = UpSampling2D((2, 2))(c4)
    u1 = concatenate([u1, c3])
    u1 = Conv2D(256, (3, 3), activation='relu', padding='same')(u1)
    u1 = BatchNormalization()(u1)

    u2 = UpSampling2D((2, 2))(u1)
    u2 = concatenate([u2, c2])
    u2 = Conv2D(128, (3, 3), activation='relu', padding='same')(u2)
    u2 = BatchNormalization()(u2)

    u3 = UpSampling2D((2, 2))(u2)
    u3 = concatenate([u3, c1])
    u3 = Conv2D(64, (3, 3), activation='relu', padding='same')(u3)
    u3 = BatchNormalization()(u3)

    u4 = UpSampling2D((2, 2))(u3)
    u4 = Conv2D(32, (3, 3), activation='relu', padding='same')(u4)
    u4 = BatchNormalization()(u4)

    output_ab = Conv2D(2, (3, 3), activation='tanh', padding='same')(u4)

    model = Model(inputs=input_l, outputs=output_ab, name="colorize_model")
    return model
