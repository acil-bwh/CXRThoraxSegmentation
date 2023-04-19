import os
import re
import classification_funct.image.model_class as mo_class
import image_functions.mask_funct as msk
import image_functions.image_funct as im


def get_prepared_img(img, pix, mask_model, clahe_bool = False):
    model_instance = mo_class.model_class(os.path.join('./models', mask_model + '.h5'))
    # Check if it is a thorax model to remove parts in segmentation process
    thorax_model = bool(re.search('thorax', mask_model))
    img = msk.des_normalize(msk.apply_mask(img,model_instance.model,thorax_model))
    img = im.recolor_resize(img, pix)
    if clahe_bool:
        img = im.clahe(img)
    img = im.normalize(img)
    return img
