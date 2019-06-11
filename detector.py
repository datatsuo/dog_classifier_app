# this part is based on the dog_app.ipynb
# here the functions for (1) detect human (2) detect dog
# and (3) detect dog breed are given

import cv2
import tensorflow as tf
from tqdm import tqdm
import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.models import model_from_json

from extract_bottleneck_features import *

# read the list of the dog names
f = open("./model/dog_names.txt","r")
dog_names = []
for name in f:
    dog_names.append(name.rstrip("\n"))
f.close()

# read the trained deep learning model for breed classification
json_file = open('./model/Resnet50_model_breed.json', 'r')
json_loaded = json_file.read()
json_file.close()
Resnet50_model_breed = model_from_json(json_loaded)
Resnet50_model_breed.load_weights('./model/weights.best.Resnet50.hdf5')
graph2 = tf.get_default_graph()

# read the model for dog detection
ResNet50_model_dog_detect = ResNet50(weights='imagenet')
graph1 = tf.get_default_graph()

# extract pre-trained face detector for face
cascade_path_face = './haarcascades/haarcascade_frontalface_default.xml'

# face detector
def face_detector(img_path):

    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cascade_path_face)
    faces = face_cascade.detectMultiScale(gray)

    return len(faces) > 0

# converting the data shape for dog detector
def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

# prediction of the label based on ResNet50 with ImageNet
def ResNet50_predict_labels(img_path):
    global graph1
    with graph1.as_default():
        img = preprocess_input(path_to_tensor(img_path))
        return np.argmax(ResNet50_model_dog_detect.predict(img))

# dog detector
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))

# predicting dog breed based on our model
def Resnet50_predict_breed(img_path):
    global graph2
    with graph2.as_default():
        # extract bottleneck features
        bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
        # obtain predicted vector
        predicted_vector = Resnet50_model_breed.predict(bottleneck_feature)
        # return dog breed that is predicted by the model
        return dog_names[np.argmax(predicted_vector)]

# human/dog detector
def dog_or_dog_like_human(img_path):
    global graph2
    with graph2.as_default():
        # for dog image return the breed
        if(dog_detector(img_path) == True):
            comment = "A dog is detected. The breed is:"
            label = "dog"
            dog_breed = Resnet50_predict_breed(img_path)
            return [label, comment, dog_breed]
        elif(face_detector(img_path) == True):
            # for human image return which dog breed she/he looks like
            label = "human"
            comment = "A human is detected. She/He looks like a dog with breed:"
            dog_breed = Resnet50_predict_breed(img_path)
            return [label, comment, dog_breed]
        else:
            # otherwise return error
            label = "error"
            comment = "This is neither dog nor human."
            dog_breed = "error"
            print("error")
            return [label, comment, dog_breed]
