import os
import argparse
import model_functions.logs as logs
import tensorflow as tf
import model_functions.evaluation as ev


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
    parser.add_argument('-m',
                        '--model',
                        type=str,
                        default="unet",
                        help="type of model")
    parser.add_argument('-p',
                        '--path',
                        type=str,
                        default='/home/mr1142/Documents/ACIL_data_repo/CXRThoraxSegmentation/data/thorax_segmentation',
                        help="type of model") 

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    name = args.name
    model_type = args.model
    path = args.path
    train_path = os.path.join(path, 'train')
    val_path = os.path.join(path, 'test')
    batch = 8
    epoch = 200
    pixels = 256
    #----------------------------------------------------
    import image_functions.image_funct as im
    import model_functions.unet_doble_loss as u_loss
    import model_functions.eff_unet as u_eff
    import model_functions.unet_funct as u_net
    import image_functions.extra_functions as ex

    # MODELS FUNCTIONS
    metrics = [ex.dice_coef_loss, u_loss.loss_mask, 'accuracy', 'AUC',
                tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives()]

    def unet():
        unet_model = u_net.build_unet_model(256,1)
        unet_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                        loss=ex.dice_coef_loss,
                        metrics=metrics)
        return unet_model

    def uloss():
        unet_model = u_loss.build_unet_model(256)
        unet_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 1e-4),
                            loss=u_loss.MyLoss,
                            metrics =metrics)
        return unet_model
    
    def ueff():
        unet_model = u_eff.build_unet_model(256)
        unet_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 1e-4),
                            loss=ex.dice_coef_loss,
                            metrics =metrics)
        return unet_model
    #----------------------------------------------------

    # DATA
    # Get images and masks names
    masks_name = ex.list_files(os.path.join(train_path, 'masks'))

    # Create images and masks tensors
    masks = im.create_tensor(train_path, 'masks', masks_name, im.binarize, pixels)
    images = im.create_tensor(train_path, 'images', masks_name, im.normalize, pixels)

    # Agument data
    images, masks = im.augment_tensor(images,masks)

    # CALLBACK
    if model_type == 'ueff':
        callb = [logs.early_stop(10)]
    else:
        callb = [logs.tensorboard('thorax_' + model_type + '_' + name), logs.early_stop(10)]

    # MODEL
    # Select model
    if model_type == 'unet':
        model = unet()
    elif model_type == 'uloss':
        model = uloss()
    elif model_type == 'ueff':
        model = ueff()
    else:
        model = None
        print('\n INCORRECT MODEL \n')

    # Train model
    history = model.fit(images,masks,
                            batch_size = batch,
                            epochs = epoch,
                            callbacks= callb,
                            shuffle = True,
                            validation_split = 0.2)

    model.save('./models/thorax_' + model_type + '_' + name + '.h5')
    
    # VALIDATION
    # Over the normal dataset
    results = ev.evaluate(model, val_path)
    ev.save_eval(model_type, name, results,'./results/thorax_segmentation/validation_results.csv')
    # Over the patologic dataset
    pathologic_results = ev.evaluate(model, '/home/mr1142/Documents/ACIL_data_repo/CXRThoraxSegmentation/data/thorax_pathologic_segmentation')
    ev.save_eval(model_type, name, results,'./results/thorax_segmentation/pathologic_validation_results.csv')
