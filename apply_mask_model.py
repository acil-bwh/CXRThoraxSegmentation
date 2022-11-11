import os
import re
import argparse
import cv2
from tqdm import tqdm
import tensorflow.keras as keras


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--device',
                        help="GPU device",
                        type=str,
                        default=2)
    parser.add_argument('-m',
                        '--model',
                        type=str,
                        default='thorax_segmentation_model',
                        help="nombre del modelo (incluyendo extension)")
    parser.add_argument('-p',
                        '--path',
                        type=str,
                        default='/home/mr1142/Documents/Data/prueba',
                        help="path with the images over you want the model to be applied")

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    model_name = args.model + '.h5'
    path = args.path

    # Get images names in the folder
    images = [im for im in os.listdir(path) if 
                            im.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]
    
    # Check if the images path exists and contain images
    if os.path.exists(path) and len(images) > 0:

        # Load packages
        import image_functions.mask_funct as msk
        import image_functions.extra_functions as ex
        import model_functions.unet_doble_loss as u_loss

        #Load model
        model = os.path.join('./models', model_name)
        model = keras.models.load_model(model, 
                                            custom_objects={"MyLoss": u_loss.MyLoss, 
                                                            "loss_mask": u_loss.loss_mask, 
                                                            "dice_coef_loss": ex.dice_coef_loss,
                                                            "dice_coef": ex.dice_coef})                                                          

        # Create masks path
        if not os.path.exists(os.path.join(path, 'masks')):
            os.makedirs(os.path.join(path, 'masks'))

        # Check if it is a thorax model to remove parts in segmentation process
        thorax_model = bool(re.search('thorax', model_name))
        
        # Create mask per image
        for image in tqdm(images):
            img = cv2.imread(os.path.join(path, image))
            segmented = msk.des_normalize(msk.apply_mask(img, model, thorax_model))    
            cv2.imwrite(os.path.join(path,'masks', image), segmented)
    else:
        print("\n Images path does not exist or it does not have any image!! \n Introduce a new path: python apply_model.py -p path")
