import os
import math
import cv2
import numpy as np
from tensorflow.keras.utils import Sequence
import functions.image.image_funct as im


class DataGenerator(Sequence):
    
    def __init__(self, df, batch_size, pix, path):
        self.df = df.sample(frac=1).reset_index(drop=True)
        self.batch_size = batch_size
        self.pix = pix
        self.path = path
        self.errors = 0
        self.errors_location = []

    def __len__(self):
        return math.ceil(len(self.df['normal']) / self.batch_size)

    def __getitem__(self, idx):
        # idx: numero de batch
        # batch 0: idx = 0 -> [0*batch_size:1*batch_size]
        # batch 1: idx = 1 -> [1*batch_size:2*batch_size]
        # Lo que hago es recorrer el indice
        batch_df = self.df.iloc[idx * self.batch_size:(idx + 1) * self.batch_size].reset_index(drop = True)
        batch_y = np.array(batch_df[['normal']])
        imgs = [cv2.imread(os.path.join(self.path, batch_df['img_names'].iloc[i])) for i in range(len(batch_df))]
        imgs = [im.normalize(img) for img in imgs]
        batch_x = np.stack(imgs, axis = 0)
        return batch_x, batch_y

