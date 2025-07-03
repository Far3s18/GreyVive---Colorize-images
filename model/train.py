from tensorflow.keras.callbacks import ModelCheckpoint
from model.model import create_model
from data.preprocessing import PreprocessingSequence
from config.config import IMG_SIZE, BATCH_SIZE, RAW_DATA
import os
import numpy as np

image_paths = [os.path.join(RAW_DATA, i) for i in os.listdir(RAW_DATA)][:50000]
np.random.shuffle(image_paths)

train_paths = image_paths[:45000]
val_paths = image_paths[45000:]

train_gen = PreprocessingSequence(train_paths, BATCH_SIZE, IMG_SIZE)
val_gen = PreprocessingSequence(val_paths, BATCH_SIZE, IMG_SIZE)

model = create_model()
model.compile(optimizer='adam', loss='mse')

model.fit(train_gen, validation_data=val_gen, epochs=10)

model.save('/home/fares-fadi/Desktop/Fares/Portfolio/greyVive/saved_models/colorize_model.keras')

