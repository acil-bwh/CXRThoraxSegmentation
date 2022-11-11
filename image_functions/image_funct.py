import os
import math
import cv2
import numpy as np
import albumentations as A
from tensorflow.keras.utils import Sequence


def recolor_resize(img, pix=256):
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        print('', end = '')
    img = cv2.resize(img, (pix, pix))
    img = np.expand_dims(img, axis=-1)
    return img


def read_img(path, folder, img, pix = 256):
    img = cv2.imread(os.path.join(path, folder, img))
    img = recolor_resize(img, pix)
    return img


def clahe(img):
    clahe = cv2.createCLAHE()
    img = np.uint8(img)
    final_img = clahe.apply(img)
    final_img = np.expand_dims(final_img, axis=-1)
    return final_img


def normalize(img):
    return (img - np.mean(img))/ np.std(img)


def binarize(img):
    img[img>0] = 1
    return img


def norm_clahe(img):
    img = clahe(img)
    img = normalize(img)
    return img


def create_tensor(path, folder, names, func, pixels=256):
    tensor = np.zeros((len(names), pixels,pixels,1))
    for i in range(len(names)):
        tensor[i, ...] = func(read_img(path, folder, names[i], pixels))
    return tensor
    

def albumentation(input_image, input_mask):
    transform = A.Compose([
        A.Rotate(limit=90, border_mode = None, interpolation=2, p=1),
        A.OneOf([
            A.RandomCrop(p= 1, width=230, height=230),
            A.Downscale(scale_min=0.5, scale_max=0.8, interpolation=0, always_apply=False, p=0.5),
            A.GridDistortion (num_steps=5, distort_limit=0.3, interpolation=2, border_mode=None, p=1),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), always_apply=False, p=0.5),
            A.MotionBlur(blur_limit=7, always_apply=False, p=0.5),
            A.ElasticTransform(alpha=0.5, sigma=50, alpha_affine=50, interpolation=1, border_mode=None, always_apply=False, p=1)
        ], p=0.8),
    ])
    transformed = transform(image=input_image.astype(np.float32), mask=input_mask.astype(np.float32))
    input_image = normalize(recolor_resize(transformed['image']))
    input_mask = binarize(recolor_resize(transformed['mask']))
    return input_image, input_mask


def augment_tensor(images_tensor, masks_tensor, n=2):
    new_n = images_tensor.shape[0]
    pixels = images_tensor.shape[1]
    for _ in range(n):
        new_img = np.zeros((new_n, pixels,pixels,1))
        new_mask = np.zeros((new_n, pixels,pixels,1))
        for j in range(new_n):
            img, mask = albumentation(images_tensor[j], masks_tensor[j])
            new_img[j, ...] = img
            new_mask[j,...] = mask
        images_tensor = np.concatenate((new_img, images_tensor), axis = 0)
        masks_tensor = np.concatenate((new_mask, masks_tensor), axis = 0)
    return images_tensor, masks_tensor


class DataGenerator(Sequence):
    
    def __init__(self, path, names, batch_size = 8, pix = 256):
        self.path = path
        self.names = names
        self.batch_size = batch_size
        self.pix = pix
        self.errors = 0
        self.errors_location = []

    def __len__(self):
        # numero de batches
        return math.ceil(len(self.names) / self.batch_size)

    def __getitem__(self, idx):
        # idx: numero de batch
        # batch 0: idx = 0 -> [0*batch_size:1*batch_size]
        # batch 1: idx = 1 -> [1*batch_size:2*batch_size]
        # Lo que hago es recorrer el indice
        batch_images = self.names[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = np.zeros((len(batch_images), self.pix, self.pix, 1))
        batch_y = np.zeros((len(batch_images), self.pix, self.pix, 1))
        for i, image in enumerate(batch_images):
            try:
                batch_x[i,...] = normalize(read_img(self.path, 'images', image, self.pix))
                batch_y[i,...] = binarize(read_img(self.path, 'masks', image, self.pix))
            except:
                self.errors = self.errors +1
                self.errors_location.append(idx*self.batch_size + i)
                img = np.random.randint(0,255,self.pix*self.pix).reshape((self.pix, self.pix, 1))
                batch_x[i,...] = normalize(img)
                batch_y[i,...] = binarize(img)
                
        # batch_x = fu.augment_tensor(batch_x)
        return batch_x, batch_y

