import cv2
from tqdm import tqdm
import numpy as np
import pandas as pd
import functions.image.image_funct as im


def prediction_tensor(model, images_path, batch_size = 20):
    division = len(images_path)/batch_size
    if int(division) == division:
        batches = int(division)
    else:
        batches = int(division) +1
    y_pred = []
    for batch in tqdm(range(batches)):
        batch_paths = images_path[batch*batch_size:(batch+1)*batch_size]
        imgs = [im.recolor(cv2.imread(img))[..., np.newaxis] for img in batch_paths]
        imgs = [im.normalize(img) for img in imgs]
        imgs = np.stack(imgs, axis = 0)
        y_pred.append(model.predict(imgs, verbose=0, batch_size=batch_size))
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
