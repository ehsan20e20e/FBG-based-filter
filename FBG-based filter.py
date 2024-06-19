"""
The code for the research presented in the paper titled "Inverse Design of FBG-based Optical Filters using Deep Learning: A Hybrid CNN-MLP Approach"

This code corresponds to the article's Convolutional Neural Network (CNN) section.
Please cite the paper in any publication using this code.
"""
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import os 
from imutils import paths
import cv2
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import Adam


Link = "D:/FBG-based filters" 

result = pd.read_csv("D:/FBG-based filters/DL-based_FBG_V.csv", header=None)
result = result.to_numpy()
#y = result[0:result.shape[0],10:310]

def Loading_Image_dataset(Link):
    Img_dataset = []
    Img_labels = []
    imagePaths = list(paths.list_images(Link))
    for imagePath in imagePaths:
       image = cv2.imread(imagePath, cv2.IMREAD_UNCHANGED)
       label = result[0:result.shape[0],10:310]
       image = image.astype('float32')
       Img_dataset.append(image)
       Img_labels.append(label)
    Img_dataset_array = np.asarray(Img_dataset)
    Img_labels_array = np.asarray(Img_labels)
    return Img_dataset_array, Img_labels_array 



X, L = Loading_Image_dataset(Link)

# determin channel
if K.image_data_format() == 'channels_first':
    X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
    input_shape = (1, X.shape[1], X.shape[2])
else:
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    input_shape = (X.shape[1], X.shape[2], 1)

# Normalizaion data    
X_train, X_test, L_train, L_test = train_test_split(X, L, test_size=0.3)
L_train = to_categorical(L_train, num_classes=10)
L_test = to_categorical(L_test, num_classes=10)
X_train = (X_train - X_train.min())/(X_train.max() - X_train.min())
X_test = (X_test - X_test.min())/(X_test.max() - X_test.min())



input_shape= (800, 100, 1)
# CNN model
Model = Sequential()
Model.add(Conv2D(32, (3,3), input_shape =input_shape, activation = 'relu' ))
Model.add(MaxPooling2D((2,2)))
Model.add(Conv2D(64, (3,3), activation = 'relu'))
Model.add(MaxPooling2D((2,2)))
Model.add(Dropout(0.25))
Model.add(Flatten())
Model.add(Dense(360, activation = 'relu'))



Model.summary()

Model.compile(Adam(),
              loss = 'categorical_crossentropy',
              metrics= ['accuracy'])

h = Model.fit(X_train, L_train, epochs= 500, batch_size = 64)

plt.plot(h.history['loss'])
plt.title('The loss of training model')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

plt.plot(h.history['accuracy'])
plt.title('The accuracy of training model')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()

Pre = Model.predict(X_test, batch_size=64)
Output_model = Model.predict_classes(X_test, batch_size=64)
Score = Model.evaluate(X_test, L_test, batch_size= 64)
print("Test Loss: ", Score[0])
print("Test accuracy: ", Score[1])




    
