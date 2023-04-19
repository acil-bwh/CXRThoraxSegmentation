import os
import argparse
import cv2
from tqdm import tqdm
import functions.image.image_funct as im


def batches(images_path, batch_size = 20):
    division = len(images_path)/batch_size
    if int(division) == division:
        batches = int(division)
    else:
        batches = int(division) +1
    return batches



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--device',
                        help="GPU device",
                        type=str,
                        default=0)
    parser.add_argument('-m',
                        '--model',
                        type=str,
                        default='thorax_segmentation_model',
                        help="nombre del modelo")
    parser.add_argument('-p',
                        '--path',
                        type=str,
                        default='/home/mr1142/Documents/Data/folder_PADCHEST',
                        help="path with the images over you want the model to be applied")

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    mask = args.model
    path = args.path
    batch_size = 200
    pix = 512

    # Get images names in the folder
    images = [im for im in os.listdir(path) if 
                            im.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]
    
    # Create masks path
    folder = os.path.join(path, 'processed_' + args.model)
    if not os.path.exists(folder):
        os.makedirs(folder)

    for batch in tqdm(range(batches(images,batch_size))):
        batch_paths = images[batch*batch_size:(batch+1)*batch_size]
        imgs = [cv2.imread(os.path.join(path, img)) for img in batch_paths]
        imgs = im.get_prepared_imgs(imgs, pix, mask, True)
        [cv2.imwrite(os.path.join(folder, batch_paths[i]), imgs[i]) for i in range(len(imgs))]


        
