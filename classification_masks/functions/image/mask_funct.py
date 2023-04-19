import cv2
import numpy as np
from skimage import measure
from scipy import ndimage
import functions.image.image_funct as im


def remove_parts(mask):
    try:
        mask = measure.label(mask)
        ntotal = {k: (k==mask).sum() for k in np.unique(mask) if k >0}
        k = list(ntotal.keys())[np.argmax(list(ntotal.values()))]
        mask = k==mask
        mask = ndimage.binary_fill_holes(mask, structure=np.ones((5,5)))
    except Exception as e:
        print(e)
    return mask


def des_normalize(img):
    return cv2.normalize(img, None, alpha = 0, beta = 255,
                         norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)



def apply_masks(imgs, model, remove=True):
    pix = imgs[0].shape[1]
    imgs = [im.recolor(img) for img in imgs]
    imgs2 = [im.normalize(im.recolor_resize(img, 256)) for img in imgs]
    imgs2 = np.stack(imgs2, axis=0)
    # Generate the mask
    masks = model.predict(imgs2, verbose = 0)[...,0]
    # Transform the mask into the original image siz
    masks = [cv2.resize(masks[i,...], (pix, pix)) for i in range(masks.shape[0])]    
    # If it is a thorax model clean the mask
    if remove:
        masks = [remove_parts(mask > 0.5) for mask in masks]
    return [masks[i]*imgs[i][...,0] for i in range(len(imgs))]