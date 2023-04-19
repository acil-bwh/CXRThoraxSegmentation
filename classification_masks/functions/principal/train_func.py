import pandas as pd
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.applications import Xception
import tensorflow as tf


def crear_modelo(pix):
    backbone = Xception(weights="imagenet", include_top=False, input_shape=(pix,pix,3))
    model = models.Sequential()
    model.add(layers.Conv2D(3,3,padding="same", input_shape=(pix,pix,1), activation='elu', name = 'conv_inicial'))
    model.add(backbone)
    model.add(layers.Conv2D(3000,3,padding="same", activation='elu', name = 'conv_salida'))
    model.add(layers.GlobalMaxPooling2D(name="general_max_pooling"))
    model.add(layers.Dropout(0.2, name="dropout_out_1"))
    model.add(layers.Dense(768, activation="elu"))
    model.add(layers.Dense(128, activation="elu"))
    model.add(layers.Dropout(0.2, name="dropout_out_2"))
    model.add(layers.Dense(32, activation="elu"))
    model.add(layers.Dense(1, activation="sigmoid", name="fc_out"))
    # Se coge una proporción del modelo que dejar fija
    fine_tune_at = int(len(backbone.layers)*0.5)
    backbone.trainable = True
    for layer in backbone.layers[:fine_tune_at]:
        layer.trainable = False
    return model


def tensorboard(name):
    log_dir = "./results/logs/" + name
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                        update_freq='batch',
                                                        histogram_freq=1)
    return tensorboard_callback

                                        
def early_stop(patient):
    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", restore_best_weights=True, patience = patient)
    return early_stop


def save_train_in_table(datos, mask_type, name, path):
    df = pd.read_csv(path)
    values = [mask_type, name]
    for v in datos.values():
        values.append(max(v))
    df.loc[len(df)] = values
    df.reset_index(drop = True)
    df.to_csv(path, index = False)