import numpy as np
import cv2
from skimage.color import lab2rgb, rgb2lab
from config.config import IMG_SIZE
from tensorflow.keras.models import load_model

model = load_model("/home/fares-fadi/Desktop/Fares/Portfolio/greyVive/saved_models/colorize_model.keras")

def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0

    lab = rgb2lab(img)
    L = lab[:, :, 0] / 100.0
    L_input = L.reshape(1, 128, 128, 1)

    pred_AB = model.predict(L_input)[0]
    pred_AB_rescaled = pred_AB * 128

    lab_pred = np.zeros((128, 128, 3))
    lab_pred[:, :, 0] = L * 100
    lab_pred[:, :, 1:] = pred_AB_rescaled

    rgb_result = (lab2rgb(lab_pred) * 255).astype(np.uint8)
    return rgb_result