import argparse
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
import model_functions.evaluation as ev
import model_functions.logs as logs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--device',
                        help="GPU device",
                        type=str,
                        default=2)
    parser.add_argument('-n',
                        '--name',
                        type=str,
                        default='new',
                        help="name of the model")
    parser.add_argument('-p',
                        '--path',
                        type=str,
                        default='/home/mr1142/Documents/ACIL_data_repo/CXRThoraxSegmentation/data/lung_segmentation',
                        help="segmentated images path")

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    name = args.name
    path = args.path
    train_path = os.path.join(path, 'train')
    val_path = os.path.join(path, 'test')
    batch = 8
    epoch = 200
    pixels = 256

    #----------------------------------------------------
    import image_functions.image_funct as im
    import model_functions.unet_funct as u_net
    import image_functions.extra_functions as ex

    # MODEL FUNCTIONS
    metrics = [ex.dice_coef_loss, 'accuracy', 'AUC',
                tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives()]

    def unet():
        unet_model = u_net.build_unet_model(256,1)
        unet_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                        loss=ex.dice_coef_loss,
                        metrics=metrics)
        return unet_model

    #----------------------------------------------------

    # DATA
    # Get images and masks names
    masks_name = ex.list_files(os.path.join(train_path, 'masks'))

    # Split in train and test
    train, test = train_test_split(masks_name, test_size=0.2, random_state=42)

    # Data generators
    traingen = im.DataGenerator(train_path, train, batch, pixels)
    testgen = im.DataGenerator(train_path, test, batch, pixels)

    # CALLBACK
    callb = [logs.tensorboard('lung_' + name), logs.early_stop(5)]

    # MODEL
    model = unet()

    # Train model
    history = model.fit(traingen,
                            validation_data = testgen,
                            batch_size = batch,
                            epochs = epoch,
                            callbacks= callb)

    model.save('./models/lung_unet_' +  name + '.h5')
    
    # VALIDATION
    results = ev.evaluate(model, val_path)
    ev.save_eval('unet', name, results, './results/lung_segmentation/validation_results.csv')
