

from keras.models import Sequential
from keras.models import model_from_json

import sys

def predict(model, X):
    #return model.predict(X)
    return preds

if __name__ == '__main__':

    img_dir = 'Data/Train_Data/Cocos/40_100.jpg';
    #sys.argv[1]
    from get_dataset import get_img
    img = get_img(img_dir)
    import numpy as np
    X = np.zeros((1, 64, 64, 3), dtype='float64')
    X[0] = img
    # Getting model:
    model_file = open('Data/Model/model.json', 'r')
    model = model_file.read()
    model_file.close()
    model = model_from_json(model)
    # Getting weights
    model.load_weights("Data/Model/weights.h5")
    preds = model.predict(X)
    print('Possibilities:\n[[<Apple Red 1> <Apricot> <Banana> <Beetroot> <Blueberry> <Cactus fruit> <Cherry 1> <Cocos> <Grape Blue> <Guava> <Lemon> <Mango> <Orange> <Potato Red> <Strawberry> <Tomato 1> <Walnut>]]\n' + str(predict(model, X)))

    xarr = predict(model, X)
    myarr = np.asarray(xarr)

    val = np.amax(myarr)
    index = np.argmax(myarr)
    print('The index of a fruit is', index)

    if index == 0:
        print("Predicted fruit is an apple")
    elif index == 1:
        print("Predicted fruit is an Apricot")
    elif index == 2:
        print("Predicted fruit is a banana")
    elif index == 3:
        print("Predicted fruit is a Beetroot")
    elif index == 4:
        print("Predicted fruit is a Blueberry")
    elif index == 5:
        print("Predicted fruit is a Cactus fruit")
    elif index == 6:
        print("Predicted fruit is a Cherry 1")
    elif index == 7:
        print("Predicted fruit is Cocos")
    elif index == 8:
        print("Predicted fruit is Grape Blue")
    elif index == 9:
        print("Predicted fruit is Gauva")
    elif index == 10:
        print("It is a Lemon")
    elif index == 11:
        print("It is a Mango")
    elif index == 12:
        print("It is a Orange")
    elif index == 13:
        print("It is a Potato Red")
    elif index == 14:
        print("It is a Strawberry")
    elif index == 15:
        print("It is a Tomato 1")
    elif index == 16:
        print("It is a Walnut")
