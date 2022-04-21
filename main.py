import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import matplotlib.pylab as plt
import torch
import face_recognition
import pyglet


IMAGE_RES = 224

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

    # Get a reference to webcam #0 (the default one)
    video_capture = cv2.VideoCapture(0)


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

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    countFrame = 0

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame (or one in ten, we need to research) of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)

        process_this_frame = False
        #process_this_frame = not process_this_frame
        countFrame += 1
        if countFrame > 9:
            process_this_frame = True
            countFrame = 0



        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

#=================== only for stationary images ==================================>
    """
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
            print("Found {} face(s) in this picture".format(len(face_locations)))
            print('Index №' + str(best_match_index) + '  -  this is  ' + name)

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
    """



