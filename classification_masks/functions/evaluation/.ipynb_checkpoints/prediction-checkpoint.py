import cv2
from tqdm import tqdm
import numpy as np
import pandas as pd
import classification_funct.image.prepare_img as fu


def img_prepare(img, mask, pix = 512):
    try:
        img = fu.get_prepared_img(img, pix, mask)
    except Exception as e:
        print(e)
        img = np.random.randint(0,255,512*512).reshape((512,512, 1))
    return img[np.newaxis,:]


def prediction_tensor(model, images_path, mask, pix = 512, batch_size = 20):
    division = len(images_path)/batch_size
    if int(division) == division:
        batches = int(division)
    else:
        batches = int(division) +1
    y_pred = []
    for batch in tqdm(range(batches)):
        batch_names = images_path[batch*batch_size:(batch+1)*batch_size]
        images = list(map(lambda x: img_prepare(cv2.imread(x),
                                    mask, pix), batch_names))
        images = np.concatenate(images)
        y_pred.append(model.predict(images, verbose=0, batch_size=batch_size))
    y_pred = np.concatenate(y_pred)
    return y_pred


def save(name, results, path):
    df = pd.read_csv(path)
    save = [name] + results
    try:
        # Si ya existe el modelo, se sobreescriben las métricas
        i = df[df['nombre'] == name].index
        df.loc[i[0]] = save
    except:
        df.loc[len(df.index)] = save
    df.reset_index(drop=True)
    df.to_csv(path, index = False)
