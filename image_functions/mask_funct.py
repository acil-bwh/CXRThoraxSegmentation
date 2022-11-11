import cv2
import numpy as np
from skimage import measure
from scipy import ndimage
import image_functions.image_funct as im


def remove_parts(mask):
    mask = measure.label(mask)
    ntotal = {k: (k==mask).sum() for k in np.unique(mask) if k >0}
    k = list(ntotal.keys())[np.argmax(list(ntotal.values()))]
    mask = k==mask
    mask = ndimage.binary_fill_holes(mask, structure=np.ones((5,5)))
    return mask


def recolor(img):
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        print('', end = '')
    return img


def apply_mask(img, model, remove=True):
    pix1 = img.shape[0]
    pix2 = img.shape[1]
    # To grayscale
    img = recolor(img)
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


def des_normalize(img):
    return cv2.normalize(img, None, alpha = 0, beta = 255,
                         norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)

