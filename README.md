# Fruit-Recognition-Using-Machine-Learning

- There are total 9 classes of fruits used for training and validation purposes. These 9 
classes of fruits include apple, blueberry, lemon, mango, orange, pear, pineapple, 
pomegranate and walnut. 
- The required dataset of these fruits were taken from the Fruit-360 dataset of Kaggle.
- The first step is to provide an image dataset to the system. The dataset comprises of 
9 classes.
- The second step comprises of storing the dataset into its memory. This 
image dataset is used for pre-processing. Pre-processing of the image is done to 
remove the noise and get clear image.
- The third step is feature extraction from it. The features such as edge detection, texture and color are included. Edge detection can be 
done by detecting the edges of the image. Texture is used to detect the skin disease 
and defects present on the fruit. Color features are used to identify the color of the 
fruit including mean and variance of the RGB color model.
- Based on the features extracted, Convolution Neural Network (CNN) in Keras is trained.
- The model uses layers of CNN such as Conv2D & MaxPooling2D. Conv2D is also called as 2D 
Convolution Layer. This layer produces tensor of outputs by creating convolution 
kernel that is convolved with the input layer.
- MaxPooling2D is used for max pooling 
operation for spatial data. Selecting the maximum element from the region of the 
feature map covered by the filter is the operation performed by max pooling layer. To 
reduce the dimensions of the feature maps, pooling layers are used so. 
