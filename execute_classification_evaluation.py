import os
import argparse
import pandas as pd
import numpy as np
from tensorflow import keras


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--device',
                        help="GPU device",
                        type=str,
                        default=0)
    parser.add_argument('-mt',
                        '--model_thorax',
                        type=str,
                        default='lung_thorax_comparation_thorax_model',
                        help="name of the model")
    parser.add_argument('-ml',
                        '--model_lung',
                        type=str,
                        default='lung_thorax_comparation_lung_model',
                        help="name of the model")
    parser.add_argument('-ip',
                        '--internal_path',
                        type=str,
                        default='./data/pneumonia_classification/internal_validation',
                        help="training_images_path")    
    parser.add_argument('-ep',
                        '--external_path',
                        type=str,
                        default='./data/pneumonia_classification/external_validation',
                        help="training_images_path")    

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    lung_model_name = args.model_lung
    thorax_model_name = args.model_thorax
    pix = 512 
    internal_path = args.internal_path
    external_path = args.external_path

    # VALIDATION
    import classification_funct.evaluation.metrics_and_plots as met
    import classification_funct.evaluation.prediction as pre
    import classification_funct.image.data_generator as gen

    # DATA
    # Internal
    internal_val_df = pd.read_csv(os.path.join(internal_path, 'data.csv'))
    internal_val_imgs = [os.path.join(internal_path, p) for p in internal_val_df['Image Index']]
    internal_y_true = np.array(list(internal_val_df['normal']))
    # External
    external_val_df = pd.read_csv(os.path.join(external_path, 'data.csv'))
    external_val_imgs = [os.path.join(external_path, external_val_df.folder[i], external_val_df.img_name[i]) for i in external_val_df.index]
    external_y_true = np.array(list(external_val_df['normal']))

    # One validation is made for each model
    for i, model_name in enumerate([thorax_model_name, lung_model_name]):
        print('\n {} VALIDATION \n'.format(model_name))
        if i == 0:
            mask = 'thorax_segmentation_model'
        elif i == 1:
            mask = 'lung_segmentation_model'
    
        # Load model
        model = keras.models.load_model(os.path.join('./models/classification_models', model_name + '.h5'))

        # Internal validation
        print('\n...Internal validation')
        internal_valgen = gen.DataGenerator(internal_val_df, 20, 512, mask, internal_path)
        internal_results = model.evaluate(internal_valgen, batch_size=20)
        # Delong
        internal_y_pred = pre.prediction_tensor(model, internal_val_imgs, mask, pix)
        internal_aucci = met.AUC_CI(internal_y_true,internal_y_pred,100)

        # External validation
        print('\n...External validation')
        external_y_pred = pre.prediction_tensor(model, external_val_imgs, mask, pix)
        mets, plots = met.metricas_dict(external_y_true, external_y_pred)
        for k, v in plots.items():
            met.save_plot(v, os.path.join('./results/classification', model_name), k +'external')
        # Delong
        external_aucci = met.AUC_CI(external_y_true,external_y_pred,100)
        
        # Save results
        p = './results/classification/validation.csv'
        results = [mask] + [internal_aucci[1]] + internal_results + [external_aucci[1]] + list(mets.values())
        pre.save(model_name, results, p)