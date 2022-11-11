import os
import math
import cv2
import numpy as np
from tensorflow.keras.utils import Sequence
import classification_funct.image.prepare_img as fu
import image_functions.image_funct as im


class DataGenerator(Sequence):
    
    def __init__(self, df, batch_size, pix, mask, path):
        self.df = df.sample(frac=1).reset_index(drop=True)
        self.batch_size = batch_size
        self.pix = pix
        self.mask = mask
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
        batch_x = np.zeros((len(batch_df), self.pix, self.pix, 1))
        batch_y = np.array(batch_df[['normal']])
        for i in range(len(batch_df)):
            try:
                img = cv2.imread(os.path.join(self.path, batch_df['Image Index'].iloc[i]))
                batch_x[i,...] = fu.get_prepared_img(img, self.pix, self.mask, clahe_bool=True)
            except Exception as e:
                print(e)
                self.errors = self.errors +1
                self.errors_location.append(idx*self.batch_size + i)
                img = np.random.randint(0,255,self.pix*self.pix).reshape((self.pix, self.pix, 1))
                batch_x[i,...] = im.normalize(img)
                
        # batch_x = fu.augment_tensor(batch_x)
        return batch_x, batch_y


