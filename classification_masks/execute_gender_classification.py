import os
import argparse
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import functions.principal.train_func as tr
import re


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--device',
                        help="GPU device",
                        type=str,
                        default=1)
    parser.add_argument('-n',
                        '--name',
                        type=str,
                        default='new',
                        help="name of the model") 
 

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    path = '/datagpu/datasets/mr1142/folder_NIH'
    trainprop = 0.8
    epoch = 100
    pix = 512
    batch = 16
    lr = 1e-4

    # TRAIN con cada género
    for df_name in ["NIH_male_train.csv", "NIH_female_train.csv"]:
        male_female = re.split("_",df_name)[1]

        # DATA ----------------------------------------------------
        df = pd.read_csv(os.path.join("./data", df_name))
        train, val = train_test_split(df, test_size=0.2, stratify = df.normal)

        # TRAIN con cada máscara
        masks = ['lung_segmentation_model', 'thorax_segmentation_model', "none"]
        for mask in masks:
            path_preprocessed = os.path.join(path, "processed_" + mask)
            name = 'gender_classification_' + male_female + "_" + mask + '_' + args.name
            print('\n ...Training {} \n'.format(name))

            # Data generators
            import functions.principal.data_generator as gen
            traingen = gen.DataGenerator(train, batch, pix, path_preprocessed)
            valgen = gen.DataGenerator(val, batch, pix, path_preprocessed)

            # MODEL
            model = tr.crear_modelo(pix)    

            # Compile
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = lr), 
                            loss = 'binary_crossentropy',
                            metrics = ['BinaryAccuracy', 'Precision', 'AUC'])

            # CALLBACK
            callb = [tr.early_stop(3)]

            # TRAIN
            history = model.fit(traingen, 
                                validation_data = valgen,
                                batch_size = batch,
                                callbacks = callb,
                                epochs = epoch,
                                shuffle = True)
            
            # Save model
            model.save('./models/gender_models/' + name + '.h5')
            print('MODEL SAVED')

            # Save metrics
            data = history.history
            tr.save_train_in_table(data, mask, "gender_classification_" + male_female + "_" + args.name, './results/classification/training_data.csv')

            # Save prediction
            import functions.evaluation.gender_evaluation as gender
            gender.evaluation(model, args.name, mask, df_name, male_female)







