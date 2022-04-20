import pathlib

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import idx2numpy
import cv2
import matplotlib
import matplotlib.pyplot as plt
import time
import PIL.Image as Image
from io import BytesIO

import matplotlib.pylab as plt
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import tf2onnx
import onnxruntime as ort
import subprocess
import onnx
import torch
from torchvision import datasets, transforms
from torch import nn
import torch.nn.functional as F
import face_recognition


#CLASSIFIER_URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
IMAGE_RES = 224

def export_to_onnx(model):
# https://github.com/onnx/tensorflow-onnx/blob/master/tutorials/keras-resnet50.ipynb
    spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
    output_path = model.name + ".onnx"

    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)
    output_names = [n.name for n in model_proto.graph.output]

def preprocess(img):
    img = img / 255.
    img = cv2.resize(img, (IMAGE_RES, IMAGE_RES))
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return img

def get_image(path, show=False):
    with Image.open(path) as img:
        img = np.array(img.convert('RGB'))
    if show:
        plt.imshow(img)
        plt.axis('off')
    return img




if __name__ == '__main__':
    device = "Cuda" if torch.cuda.is_available() else "CPU"
    print(f"Using {device} device")

    if device=="CPU":
        count_kernels = subprocess.run('face_recognition --cpus -1 ./KNOWN_PEOPLE_FOLDER/ ./IMAGE_TO_CHECK/'.split(),
                                   capture_output=True)
        print(str(count_kernels))

    pictureJ = face_recognition.load_image_file("/home/progforce/facerecognition/KNOWN_PEOPLE_FOLDER/J1.jpg")
    my_face_encoding = face_recognition.face_encodings(pictureJ)[0]

    # my_face_encoding now contains a universal 'encoding' of my facial features that can be compared to any other picture of a face!

    unknown_picture = face_recognition.load_image_file("/home/progforce/facerecognition/IMAGE_TO_CHECK/K1.jpg")
    unknown_face_encoding = face_recognition.face_encodings(unknown_picture)[0]

    # Now we can see the two face encodings are of the same person with `compare_faces`!

    results = face_recognition.compare_faces([my_face_encoding], unknown_face_encoding)

    if results[0] == True:
        print("It's the same person")
    else:
        print("it's NOT the same person")



















    """
    layer = hub.KerasLayer(CLASSIFIER_URL, input_shape=(IMAGE_RES, IMAGE_RES, 3), trainable=True)
    #import pdb; pdb.set_trace()
    model = tf.keras.Sequential([
        layer
    ])

    grace_hopper = tf.keras.utils.get_file('image.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg')
    #grace_hopper = Image.open('car3.jpg').resize((IMAGE_RES, IMAGE_RES))
    grace_hopper = Image.open(grace_hopper).resize((IMAGE_RES, IMAGE_RES))

    # If you need to see a picture
    # grace_hopper.save("grace.jpg")
    # picture = cv2.imread('grace.jpg')
    # cv2.imshow('grace_hopper', picture)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    grace_hopper_div = np.array(grace_hopper) / 255.0
    #print(np.shape(grace_hopper_div))
    #print(np.shape(grace_hopper_div[np.newaxis, ...]))   # models always want a batch of images to process. So here, we add a batch dimension, to pass the image to the model for prediction.


    result = model.predict(grace_hopper_div[np.newaxis, ...])
    print(result.shape)

    predicted_class = np.argmax(result[0], axis=-1)


    labels_path = tf.keras.utils.get_file('ImageNetLabels.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
    imagenet_labels = np.array(open(labels_path).read().splitlines())

    predicted_class_name = imagenet_labels[predicted_class]
    title = predicted_class_name.title()
    print('Data from TF-model ...  there must be military_uniform')
    print(predicted_class, predicted_class_name)

    # grace_hopper.save("grace.jpg")
    # picture = cv2.imread('grace.jpg')
    # cv2.imshow(title, picture)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

#================================ save model as ONNX and launch it ==========================================>
    proc = subprocess.run('python -m tf2onnx.convert --opset 13 --saved-model "home/progforce/TestModels" --output pf1.onnx'.split(),
                          capture_output=True)   # it creates  sequential.onnx   why?????

    onnx_model = onnx.load('sequential.onnx')
    session = ort.InferenceSession(onnx_model.SerializeToString())

    pathImg = pathlib.Path('cat2.jpg')
    img = get_image(pathImg)
    img = preprocess(img)
    ort_input = {session.get_inputs()[0].name: img}
    preds = session.run(None, ort_input)[0]

    print(preds.shape)
    onnx_predicted_class = np.argmax(preds)
    onnx_predicted_class_name = imagenet_labels[onnx_predicted_class]
    print('Data from ONNX-model... there must be a cat :)')
    print(onnx_predicted_class, onnx_predicted_class_name)


    #
    # for number in range (1000):
    #     class_name = imagenet_labels[number]
    #     title = class_name.title()
    #     print(number, ' ', title)


#===================================== full preservation of TF model ===============================>
    model.save('/home/progforce/TestModels/')
    restore_model = keras.models.load_model('/home/progforce/TestModels/')
    restore_model.summary()

#===================================== save and load model as Keras .h5 ============================>

    # t = time.time()
    # export_path_keras = "./{}.h5".format(int(t))
    model.save('myKeras.h5', save_format='h5')

    # `custom_objects` tells keras how to load a `hub.KerasLayer`
    reloaded_model = keras.models.load_model('myKeras.h5', custom_objects={'KerasLayer': hub.KerasLayer})
    print("It's from h5_model")
    reloaded_model.summary()

    print('Data from reloaded model.. it should be the same cat')
    predict_from_reload = reloaded_model.predict(img)
    predicted_class_from_reload = np.argmax(predict_from_reload[0], axis=-1)
    predicted_class_name_from_reload = imagenet_labels[predicted_class_from_reload]
    print(predicted_class_from_reload, predicted_class_name_from_reload)



    #====================== Display image ================================>
    # img = cv2.imread('/home/progforce/Pictures/inJava.jpg')
    # cv2.imshow('Java', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    """
