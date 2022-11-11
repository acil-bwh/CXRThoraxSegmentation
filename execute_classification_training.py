import os
import argparse
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.applications import Xception
from sklearn.model_selection import train_test_split
import model_functions.logs as logs


def crear_modelo(input_shape):
    backbone = Xception(weights="imagenet", include_top=False, input_shape=input_shape)
    model = models.Sequential()
    model.add(layers.Conv2D(3,3,padding="same", input_shape=(pix,pix,1), activation='elu', name = 'conv_inicial'))
    model.add(backbone)
    # model.add(layers.Conv2D(3000,3,padding="same", input_shape=(pix,pix,1), activation='elu', name = 'conv_salida'))
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


def save_train_in_table(datos, mask_type, name, path):
    df = pd.read_csv(path)
    values = [mask_type, name]
    for v in datos.values():
        values.append(max(v))
    df.loc[len(df)] = values
    df.reset_index(drop = True)
    df.to_csv(path, index = False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--device',
                        help="GPU device",
                        type=str,
                        default=0)
    parser.add_argument('-n',
                        '--name',
                        type=str,
                        default='new',
                        help="name of the model")
    parser.add_argument('-mt',
                        '--model_thorax',
                        type=str,
                        default='thorax_segmentation_model',
                        help="name of the model")
    parser.add_argument('-ml',
                        '--model_lung',
                        type=str,
                        default='lung_segmentation_model',
                        help="name of the model")
    parser.add_argument('-p',
                        '--path',
                        type=str,
                        default='./data/pneumonia_classification/training_data',
                        help="training_images_path")    

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    path = args.path
    lung_mask_name = args.model_lung
    thorax_mask_name = args.model_thorax
    trainprop = 0.8
    epoch = 100
    pix = 512
    batch = 8
    lr = 1e-4

    # DATA ----------------------------------------------------
    df = pd.read_csv(os.path.join(path, 'data.csv'))
    train, val = train_test_split(df, test_size=0.2, stratify = df.normal)

    for mask in [thorax_mask_name, lung_mask_name]:
        name = 'classification_' + mask + '_' + args.name
        print('\n ...Training {} \n'.format(name))

        # Data generators
        import classification_funct.image.data_generator as gen
        traingen = gen.DataGenerator(train, batch, pix, mask, path)
        valgen = gen.DataGenerator(val, batch, pix, mask, path)

        # MODEL
        input_shape = (pix,pix,3)
        model = crear_modelo(input_shape)    

        # Compile
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = lr), 
                        loss = 'binary_crossentropy',
                        metrics = ['BinaryAccuracy', 'Precision', 'AUC'])

        # CALLBACK
        callb = [logs.early_stop(5)]

        # TRAIN
        history = model.fit(traingen, 
                            validation_data = valgen,
                            batch_size = batch,
                            callbacks = callb,
                            epochs = epoch,
                            shuffle = True)
        
        # Save model
        model.save('./models/classification_models/' + name + '.h5')
        print('MODEL SAVED')

        # Save metrics
        data = history.history
        save_train_in_table(data, mask, args.name, './results/classification/training_data.csv')





