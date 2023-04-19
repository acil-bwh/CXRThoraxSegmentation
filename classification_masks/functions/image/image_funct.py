import os
import cv2
import numpy as np
import re
import functions.principal.model_class as mo_class
import functions.image.mask_funct as msk


def recolor_resize(img, pix=256):
    img = recolor(img)
    try:
        img = cv2.resize(img, (pix, pix))
    except:
        print("resize exception")
        img = np.random.rand(pix, pix)
    img = np.expand_dims(img, axis=-1)
    return img


def recolor(img):
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        print("", end = "")
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
    norm = (img - np.mean(img))/ np.std(img)
    norm = np.nan_to_num(norm)
    return norm


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
    

def get_prepared_imgs(imgs, pix, mask_model, clahe_bool = False):
    imgs = [recolor_resize(img, pix) for img in imgs]
    if mask_model != 'none':
        model_instance = mo_class.model_class(os.path.join('./models', mask_model + '.h5'))
        # Check if it is a thorax model to remove parts in segmentation process
        thorax_model = bool(re.search('thorax', mask_model))
        masked_imgs = msk.apply_masks(imgs,model_instance.model,thorax_model)
        imgs = [msk.des_normalize(masked) for masked in masked_imgs]
    if clahe_bool:
        imgs = [clahe(img) for img in imgs]
    return imgs