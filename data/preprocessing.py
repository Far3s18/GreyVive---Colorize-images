import cv2
import numpy as np
from tensorflow.keras.utils import Sequence
from skimage.color import rgb2lab

class PreprocessingSequence(Sequence):
    def __init__(self, image_paths, batch_size, img_size):
        self.image_paths = image_paths
        self.batch_size = batch_size
        self.img_size = img_size

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, idx):
        batch_paths = self.image_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        L_images, AB_images = [], []

        for img_path in batch_paths:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.img_size)
            img = img / 255.0

            lab = rgb2lab(img)
            L = lab[:, :, 0] / 100.0
            AB = lab[:, :, 1:] / 128.0

            L_images.append(L.reshape(self.img_size + (1,)))
            AB_images.append(AB)

        return np.array(L_images), np.array(AB_images)
