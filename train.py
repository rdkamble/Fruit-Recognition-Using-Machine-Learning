

import os
from keras.callbacks import ModelCheckpoint, TensorBoard

def train_model(model, X, X_test, Y, Y_test):
    checkpoints = []
    if not os.path.exists('Data/Checkpoints/'):
        os.makedirs('Data/Checkpoints/')
    checkpoints.append(ModelCheckpoint('Data/Checkpoints/best_weights.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1))
    checkpoints.append(TensorBoard(log_dir='Data/Checkpoints/./logs', histogram_freq=1, write_graph=True))



    from keras.preprocessing.image import ImageDataGenerator
    generated_data = ImageDataGenerator(featurewise_center=False, samplewise_center=False, featurewise_std_normalization=False, samplewise_std_normalization=False, zca_whitening=False, rotation_range=0,  width_shift_range=0.1, height_shift_range=0.1, horizontal_flip = True, vertical_flip = False)
    generated_data.fit(X)
    import numpy
    model.fit_generator(generated_data.flow(X, Y, batch_size=32), steps_per_epoch=X.shape[0]//32, epochs=5,
                        validation_data=(X_test, Y_test), callbacks=checkpoints)

    return model

def main():
    from get_dataset import get_dataset
    X, X_test, Y, Y_test = get_dataset()
    from get_model import get_model, save_model
    model = get_model(len(Y[0]))
    import numpy
    model = train_model(model, X, X_test, Y, Y_test)
    save_model(model)
    return model

if __name__ == '__main__':
    main()
