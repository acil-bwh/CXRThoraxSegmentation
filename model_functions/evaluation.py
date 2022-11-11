import os
import image_functions.image_funct as im
import image_functions.extra_functions as ex
import pandas as pd


def evaluate(model, path):
    file_names = ex.list_files(os.path.join(path, 'masks'))
    masks = im.create_tensor(path, 'masks', file_names, im.binarize)
    images = im.create_tensor(path, 'images', file_names, im.normalize)
    results = model.evaluate(images, masks, batch_size=8)
    print(results)
    return results


def save_eval(model_type, name, results, path):
    df = pd.read_csv(path)
    save = [model_type, name] + results
    df.loc[len(df.index)] = save
    df.to_csv(path, index = False)
