import tensorflow.keras as keras
import classification_funct.image.losses as ex

class model_class():
    def __init__(self, model_path):
        self.model = keras.models.load_model(model_path, 
                                    custom_objects={"loss_mask": keras.losses.BinaryCrossentropy, 
                                                    "dice_coef_loss": ex.dice_coef_loss,
                                                    "dice_coef": ex.dice_coef})
    print('\n\n MASK MODEL LOADED \n\n')