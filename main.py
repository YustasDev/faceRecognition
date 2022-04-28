import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import torch
import face_recognition
import pyglet
import threading
import time
import multiprocessing
import asyncio


# say a greeting for each name
def thread_sounding(list_of_names):
    for soundName in list_of_names:
        load_sounding = soundNames[soundName]
        if load_sounding is not None:
            song = pyglet.media.load(load_sounding)
            song.play()
            time.sleep(1)

def compare_face(name, dictionary, default="Not in the dictionary"):
    if name in dictionary:
        return dictionary[name]
    else:
        print(default)
        return None



class FaceRecognitionError(Exception): pass

# Singleton class
class Person:
    count_occurrence = 0
    was_voiced = 0

    def __init__(self, name):
        self.__name = name
        self._start_time = None
        self.init_time = time.perf_counter()

    @property
    def name(self):
        return self.__name

    def start(self):
        """Starting a new timer"""

        if self._start_time is not None:
            raise FaceRecognitionError("The timer is already running")

        self._start_time = time.perf_counter()

    def current_time(self):
        current_time = time.perf_counter() - self.init_time
        return current_time

    def stop(self):
        """Stop timer and return elapsed time"""

        if self._start_time is None:
            raise FaceRecognitionError("The timer doesn't work")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None

        return elapsed_time






if __name__ == '__main__':
    device = "Cuda" if torch.cuda.is_available() else "CPU"
    print(f"Using {device} device")

    # if device=="CPU":
    #     count_kernels = subprocess.run('face_recognition --cpus -1 ./KNOWN_PEOPLE_FOLDER/ ./IMAGE_TO_CHECK/'.split(),
    #                                capture_output=True)
    #     print(str(count_kernels))

    # Get a reference to webcam #0 (the default one)
    video_capture = cv2.VideoCapture(1)

    # Find OpenCV version
    (major_ver, _, _) = (cv2.__version__).split('.')

    # be sure of the fps
    if int(major_ver) < 3:
        fps = video_capture.get(cv2.cv.CV_CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    else:
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

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
    soundNames = {'Jolie':'./SoundNames/Jsound.mp3', 'Katrin':'./SoundNames/Ksound.mp3',
                  'Me':'./SoundNames/Vsound.mp3', 'Unknown': None}

    #createInstance_known_face = {"Me": Person("Me"), "Jolie": Person("Jolie"), "Katrin": Person("Katrin")}


    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    already_said_hello = set()
    process_this_frame = True
    sound_launch = True
    countFrame = 0
    threadNew = None
    frequency_of_greeting = 10    # voiceover no more than once per ... seconds
    number_of_messages = 3        # number of playing sound files (mp.3 files)
    time_label = 0
    created_instances = {}


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

            #face_names = []
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
                    print(name)
                    print(created_instances)
#============================================================= new version ============================>
                if name != "Unknown":
                    name_instance = compare_face(name, created_instances)
                    print(name_instance)
                    if name_instance is None:
                        name_instance = Person(name)
                        print(name_instance)

                    if name_instance.count_occurrence == 0 or name_instance.current_time() > time_label + frequency_of_greeting:
                        name_instance.was_voiced += 1
                        name_instance.count_occurrence += 1
                        if name_instance.was_voiced < number_of_messages + 1:
                            face_names.append(name)
                            time_label = name_instance.current_time()
                            created_instances[name] = name_instance

                            print('ОЗВУЧКА' * 20)
                            print(name)
                            print(name_instance)
                            print(name_instance.count_occurrence)
                            print(str(name_instance.current_time()))
                            print(name_instance.was_voiced)
                            print(created_instances)


                print('НЕТ озвучки' * 10)
                print(name)
                print(name_instance)
                print(name_instance.count_occurrence)
                print(str(name_instance.current_time()))
                print(time_label)
                print(name_instance.was_voiced)


                # ==> old version
                # if name not in already_said_hello:
                #     face_names.append(name)
                #     already_said_hello.add(name)



        process_this_frame = False
        countFrame += 1
        if countFrame > 15:
            process_this_frame = True
            countFrame = 0


        if len(face_names) > 0 and sound_launch:
            threadNew = multiprocessing.Process(target=thread_sounding(face_names), args=())
            #threadNew = threading.Thread(target=thread_sounding(face_names), args=())
            threadNew.start()
            sound_launch = False

        if threadNew is not None and not threadNew.is_alive():
            sound_launch = True

        # only debug
        # print('face_names: ' + str(face_names))
        # print('already_said_hello: ' + str(already_said_hello))
        # print(str(sound_launch))
        # print(str(threadNew))


        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/2 size
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

        face_names = []

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



