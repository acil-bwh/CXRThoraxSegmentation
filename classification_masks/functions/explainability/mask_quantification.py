import image_functions.mask_funct as msk
import explainability.grad_cam as gc
import numpy as np


def apply_mask(img, model):
    # To grayscale
    img = msk.recolor(img)
    # New image with mask model input size
    img_2 = msk.normalize(msk.recolor_resize(img, 256))[np.newaxis,...]
    # Mask generation
    mask = model.predict(img_2, verbose = 0)[0,...,0]
    mask = msk.remove_pieces(mask > 0.5)
    return mask


def extract_proportion(heatmap, mask, th = 0.1):
    binary_hm = (heatmap > th) *1
    suma = binary_hm + mask
    external_activation = binary_hm - (suma==2)*1
    external_area = (mask == 0)*1
    proportion = len(np.where(external_activation.flatten() == 1)[0])/len(np.where(external_area.flatten() == 1)[0])
    return proportion

 
def list_proportions(image_list, model, mask):
    import image_functions.mask_model as mask_model
    proportions = []
    for image in image_list:
        mask_img = apply_mask(image, mask_model.model)
        _, heatmap = gc.apply_grandcam(model, mask, image)
        proportions.append(extract_proportion(heatmap, mask_img))
    return proportions