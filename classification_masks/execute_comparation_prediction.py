import os
import argparse
import pandas as pd
import numpy as np
from tensorflow import keras
import re


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--device',
                        help="GPU device",
                        type=str,
                        default=0)
    parser.add_argument('-p',
                        '--path',
                        type=str,
                        default='/home/mr1142/Documents/Data/folder_PADCHEST/',
                        help="validation images path")
    parser.add_argument('-df',
                        '--df_path',
                        type=str,
                        default="./results/classification/PADCHEST_PA.csv",
                        help="df for results")     

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    pix = 512 
    path = args.path
    df_path = args.df_path
    models = os.listdir("./models/classification_models")
    df = pd.read_csv(df_path)

    
    for i, model in enumerate(models):
        print("\n \n "+ str(i) + " " + model)
        model_name = model[:-3]
        mask = re.split("_", model)[1:-1]
        mask = "_".join(mask)

        # VALIDATION
        import functions.evaluation.prediction as pre

        # DATA
        path_preprocessed = os.path.join(path, "processed_" + mask)
        img_paths = [os.path.join(path_preprocessed, img) for img in df.img_names]
     
        # Load model
        model = keras.models.load_model(os.path.join('./models/classification_models', model))

        y_pred = pre.prediction_tensor(model, img_paths, batch_size = 50)

        df[model_name] = y_pred
        df.to_csv(df_path, index = False)

        print("\n \n END... " + str(i))
