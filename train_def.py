

import os
from keras.callbacks import ModelCheckpoint, TensorBoard

def train_model(model, X, X_test, Y, Y_test):
    checkpoints = []
    if not os.path.exists('Datadefect/Checkpoints/'):
        os.makedirs('Datadefect/Checkpoints/')
    checkpoints.append(ModelCheckpoint('Datadefect/Checkpoints/best_weights.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1))
    checkpoints.append(TensorBoard(log_dir='Datadefect/Checkpoints/./logs', histogram_freq=0, write_graph=True, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None))



    from keras.preprocessing.image import ImageDataGenerator
    generated_data = ImageDataGenerator(featurewise_center=False, samplewise_center=False, featurewise_std_normalization=False, samplewise_std_normalization=False, zca_whitening=False, rotation_range=0,  width_shift_range=0.1, height_shift_range=0.1, horizontal_flip = True, vertical_flip = False)
    generated_data.fit(X)
    import numpy
    model.fit_generator(generated_data.flow(X, Y, batch_size=8), steps_per_epoch=X.shape[0], epochs=35,
                        validation_data=(X_test, Y_test), callbacks=checkpoints)

    return model

def main():
    from get_dataset_def import get_dataset_def
    X, X_test, Y, Y_test = get_dataset_def()
    from get_model_def import get_model_def, save_model_def
    model = get_model_def(len(Y[0]))
    import numpy
    model_def = train_model(model, X, X_test, Y, Y_test)
    save_model_def(model_def)
    return model_def

if __name__ == '__main__':
    main()
