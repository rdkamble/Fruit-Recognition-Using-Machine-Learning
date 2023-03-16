

from keras.models import Sequential
from keras.models import model_from_json

import sys

def predict(model_def, X):
    return model_def.predict(X)

if __name__ == '__main__':

    img_dir = "Datadefect/Train_Data/rottenapples/rotated_by_15_Screen Shot 2018-06-07 at 2.15.34 PM.png";
    #sys.argv[1]
    from get_dataset_def import get_img
    img = get_img(img_dir)
    import numpy as np
    X = np.zeros((1, 64, 64, 3), dtype='float64')
    X[0] = img
    # Getting model:
    model_file = open('Datadefect/Model/model.json', 'r')
    model_def = model_file.read()
    model_file.close()
    model = model_from_json(model_def)
    # Getting weights
    model.load_weights("Datadefect/Model/weights1.h5")
    print('Possibilities:\n[[ <rotten apple>  <rotten banana> <rotten orange> ]]\n' + str(predict(model, X)))
    xarr = predict(model, X)
    myarr = np.asarray(xarr)

    val = np.amax(myarr)
    index = np.argmax(myarr)
    print('The index of a fruit is', index)
    if index == 0:
        print("Predicted fruit is an apple")
        return "apple"
    elif index == 1:
        print("Predicted fruit is a Banana")
    elif index == 2:
        print("Predicted fruit is a Orange")
