import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB4


def encoder_subblock(efNet_conection_output):
    x = layers.Conv2D(512,3, padding = 'same')(efNet_conection_output)
    x = residual_block(x)
    x = residual_block(x)
    x = layers.LeakyReLU()(x)
    return x


def efNet_conection(efNet_output):
    x1 = layers.LeakyReLU()(efNet_output)
    x2 = layers.MaxPool2D((2,2), padding = 'same')(x1)
    x3 = layers.Dropout(0.3)(x2)
    return x3


def decoder_subblock(encoder_output, prev_layer):
    # https://www.nature.com/articles/s41598-022-12743-y/figures/3
    unification = layers.concatenate([encoder_output, prev_layer])
    x1 = layers.Dropout(0.3)(unification)
    x2 = layers.Conv2D(3,3,padding = 'same')(x1)
    x3 = residual_block(x2)
    x3 = residual_block(x3)
    x4 = layers.LeakyReLU()(x3)
    return x4


def residual_block(prev_layer):
    path_1 = layers.LeakyReLU()(prev_layer)
    path_1 = layers.BatchNormalization()(path_1)
    path_1 = layers.Conv2D(3,3,padding = 'same')(path_1)
    path_1 = layers.BatchNormalization()(path_1)
    path_1 = layers.LeakyReLU()(path_1)
    path_1 = layers.Conv2D(3,3,padding = 'same')(path_1)
    path_1 = layers.BatchNormalization()(path_1)
    path_2 = layers.BatchNormalization()(prev_layer)
    return layers.concatenate([path_1, path_2])

    
def up_sampling(prev_layer, deep, kernel):
    x = layers.Conv2DTranspose(deep, kernel)(prev_layer)
    return x


def last_up_sampling(prev_layer):
    x = layers.Conv2D(1,3, padding='same', activation='sigmoid')(prev_layer)
    return x


def build_unet(pixels):
     # inputs
    inputs = layers.Input(shape=(pixels,pixels,3))
    
    # adaptation = layers.Conv2D(3, 3, padding="same", activation = "elu")(inputs)
        
    #EfficientNet
    efficienNet = EfficientNetB4(weights="imagenet",
                                 include_top=False,
                                 input_shape=(256,256,3),
                                 input_tensor=inputs)
    
   
    # ENCONDING
    # (128,128,144)
    dw1 = efficienNet.layers[31].output
    # (64,64,192)
    dw2 = efficienNet.layers[90].output
    # (32,32,336)
    dw3 = efficienNet.layers[149].output
    # (16,16,960)
    dw4 = efficienNet.layers[326].output

    # (8,8,960)
    middle = efNet_conection(dw4)
    # Encoder sub block (no la he añadido porque me da error)
    middle = encoder_subblock(middle)
    
    # DECODING
    # (16,16,960)
    uc1 = up_sampling(middle, dw4.shape[3],9)
    up1 = decoder_subblock(dw4, uc1)
    
    # (32,32,336)
    uc2 = up_sampling(up1, dw3.shape[3],17)
    up2 = decoder_subblock(dw3, uc2)
    
    # (64,64,192)
    uc3 = up_sampling(up2, dw2.shape[3],33)
    up3 = decoder_subblock(dw2, uc3)
    
    # (128,128,144)
    uc4 = up_sampling(up3,dw3.shape[3],65)
    up4 = decoder_subblock(dw1, uc4)

    uc5 = up_sampling(up4,16,129)

    # outputs
    outputs = last_up_sampling(uc5)

    # unet model with Keras Functional API
    unet_model = tf.keras.Model(inputs, outputs, name="U-Net")
    
    print('\ntrainable variables of unet model: {}'.format(len(unet_model.trainable_variables)))

    return unet_model


def build_unet_model(pixels, fine_tune = False, fine_tune_at = 326): 
    model = tf.keras.models.Sequential()
    model.add(layers.Conv2D(3,3,padding="same", input_shape=(pixels,pixels,1), activation='elu', name = 'conv_inicial'))
    unet_model = build_unet(pixels)
    model.add(unet_model)

    unet_model.trainable = True
    print('\ntrainable variables: {}'.format(len(model.trainable_variables)))

    if fine_tune:
        for layer in unet_model.layers[:fine_tune_at]:
            layer.trainable = False
    
    print('trainable variables: {}'.format(len(model.trainable_variables)))
    
    return model