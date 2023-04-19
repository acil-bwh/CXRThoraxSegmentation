import os
import math
import cv2
import numpy as np
from tensorflow.keras.utils import Sequence
import functions.image.image_funct as im


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
                img = cv2.imread(os.path.join(self.path, batch_df['img_names'].iloc[i]))
                batch_x[i,...] = im.get_prepared_img(img, self.pix, self.mask, clahe_bool=True)
            except Exception as e:
                print(e)
                self.errors = self.errors +1
                self.errors_location.append(idx*self.batch_size + i)
                img = np.random.randint(0,255,self.pix*self.pix).reshape((self.pix, self.pix, 1))
                batch_x[i,...] = im.normalize(img)
                
        return batch_x, batch_y



def apply_mask(img, model, remove=True):
    pix1 = img.shape[0]
    pix2 = img.shape[1]
    # To grayscale
    img = im.recolor(img)
    # New image with mask_model input size
    img_2 = im.normalize(im.recolor_resize(img, 256))[np.newaxis,...]
    # Generate the mask
    mask = model.predict(img_2, verbose = 0)[0,...]
    # Transform the mask into the original image size
    mask = cv2.resize(mask, (pix2, pix1))
    # If it is a thorax model clean the mask
    if remove:
        mask = remove_parts(mask > 0.5)
    return img*mask



def get_prepared_img(img, pix, mask_model, clahe_bool = False):
    if mask_model != 'none':
        model_instance = mo_class.model_class(os.path.join('./models', mask_model + '.h5'))
        # Check if it is a thorax model to remove parts in segmentation process
        thorax_model = bool(re.search('thorax', mask_model))
        img = msk.des_normalize(msk.apply_mask(img,model_instance.model,thorax_model))
    img = recolor_resize(img, pix)
    if clahe_bool:
        img = clahe(img)
    img = normalize(img)
    return img
