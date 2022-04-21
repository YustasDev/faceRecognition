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
from PIL import Image, ImageDraw
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

    # if device=="CPU":
    #     count_kernels = subprocess.run('face_recognition --cpus -1 ./KNOWN_PEOPLE_FOLDER/ ./IMAGE_TO_CHECK/'.split(),
    #                                capture_output=True)
    #     print(str(count_kernels))

    # Load the jpg files into numpy arrays
    j_image = face_recognition.load_image_file("./KNOWN_PEOPLE_FOLDER/J1.jpg")
    k_image = face_recognition.load_image_file("./IMAGE_TO_CHECK/K1.jpg")
    me_image = face_recognition.load_image_file("./V1/V3.jpg")
    unknown_image = face_recognition.load_image_file("./V1/V2.jpg")

    j_encoding = face_recognition.face_encodings(j_image)[0]
    k_encoding = face_recognition.face_encodings(k_image)[0]
    me_encoding = face_recognition.face_encodings(me_image)[0]

    known_face_encodings = [j_encoding, k_encoding, me_encoding]
    known_face_names = ["Jolie", "Katrin", "Me"]

    # Find all the faces and face encodings in the unknown image
    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

    # Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
    # See http://pillow.readthedocs.io/ for more about PIL/Pillow
    pil_image = Image.fromarray(unknown_image)
    # Create a Pillow ImageDraw Draw instance to draw with
    draw = ImageDraw.Draw(pil_image)

    # Loop through each face found in the unknown image
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # If a match was found in known_face_encodings, just use the first one.
        # if True in matches:
        #     first_match_index = matches.index(True)
        #     name = known_face_names[first_match_index]

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            print(name)

        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

    # Remove the drawing library from memory as per the Pillow docs
    del draw

    # Display the resulting image
    pil_image.show()






    # Get the face encodings for each face in each image file
    # Since there could be more than one face in each image, it returns a list of encodings.
    # But since I know each image only has one face, I only care about the first encoding in each image, so I grab index 0.
    # try:
    #     j_encoding = face_recognition.face_encodings(j_image)[0]
    #     k_encoding = face_recognition.face_encodings(k_image)[0]
    #     me_encoding = face_recognition.face_encodings(me_image)[0]
    #     unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]
    # except IndexError:
    #     print("I wasn't able to locate any faces in at least one of the images. Check the image files. Aborting...")
    #     quit()
    #
    # known_faces = [
    #     j_encoding,
    #     k_encoding,
    #     me_encoding
    # ]
    #
    # # results is an array of True/False telling if the unknown face matched anyone in the known_faces array
    # results = face_recognition.compare_faces(known_faces, unknown_face_encoding)
    #
    # print("Is the unknown face a picture of J? {}".format(results[0]))
    # print("Is the unknown face a picture of K? {}".format(results[1]))
    # print("Is the unknown face a picture of Me? {}".format(results[2]))
    # print("Is the unknown face a new person that we've never seen before? {}".format(not True in results))


    """"
    pictureJ = face_recognition.load_image_file("/home/progforce/facerecognition/V1/V3.jpg")
    my_face_encoding = face_recognition.face_encodings(pictureJ)[0]

    # my_face_encoding now contains a universal 'encoding' of my facial features that can be compared to any other picture of a face!

    unknown_picture = face_recognition.load_image_file("/home/progforce/facerecognition/V1/V10.jpg")
    unknown_face_encoding = face_recognition.face_encodings(unknown_picture)[0]

    # Now we can see the two face encodings are of the same person with `compare_faces`!

    results = face_recognition.compare_faces([my_face_encoding], unknown_face_encoding)

    if results[0] == True:
        print("It's the same person")
    else:
        print("it's NOT the same person")
    """


















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
