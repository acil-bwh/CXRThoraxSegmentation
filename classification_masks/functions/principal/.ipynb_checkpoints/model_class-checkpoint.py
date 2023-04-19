import tensorflow.keras as keras

class model_class():
    def __init__(self, model_path):
        self.model = keras.models.load_model(model_path, 
                                    custom_objects={"loss_mask": keras.losses.BinaryCrossentropy, 
                                                    "dice_coef_loss": keras.losses.BinaryCrossentropy,
                                                    "dice_coef": keras.losses.BinaryCrossentropy})
    print('\n\n MASK MODEL LOADED \n\n')