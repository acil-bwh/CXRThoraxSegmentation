import tensorflow as tf


def tensorboard(name):
    log_dir = "./results/logs/" + name
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                        update_freq='batch',
                                                        histogram_freq=1)
    return tensorboard_callback

                                        
def early_stop(patient):
    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", restore_best_weights=True, patience = patient)
    return early_stop
