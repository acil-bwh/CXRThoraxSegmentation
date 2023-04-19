import os
import argparse
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import functions.principal.train_func as tr


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
    parser.add_argument('-df',
                        '--df',
                        type=str,
                        default='NIH_train.csv',
                        help="training dataset")    
    parser.add_argument('-p',
                        '--path',
                        type=str,
                        default='/datagpu/datasets/mr1142/folder_NIH',
                        help="training_images_path")    

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    path = args.path
    df_name = args.df

    #os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    #path = '/home/mr1142/Documents/Data/NIH_folder'
    #df_name = 'NIH_train.csv'
    trainprop = 0.8
    epoch = 100
    pix = 512
    batch = 8
    lr = 1e-4

    # DATA ----------------------------------------------------
    df = pd.read_csv(os.path.join("./data", df_name))
    train, val = train_test_split(df, test_size=0.2, stratify = df.normal)

    masks = ['lung_segmentation_model', 'thorax_segmentation_model', "none"]
    for mask in masks:
        name = 'classification_' + mask + '_' + args.name
        print('\n ...Training {} \n'.format(name))
        path_preprocessed = os.path.join(path, "processed_" + mask)

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
        callb = [tr.early_stop(5)]

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
        tr.save_train_in_table(data, mask, args.name, './results/classification/training_data.csv')





