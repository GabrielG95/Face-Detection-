import os
import dlib
import cv2 as cv
import requests
from imutils import face_utils
from pytube import YouTube
from torchvision import transforms

#TODO: try and add face labels to the faces detected.

# load the face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/Users/gabri/Downloads/shape_predictor_68_face_landmarks.dat')

# get user home directory path
user_home = os.path.expanduser('~')
# create the path to the 'Downloads' folder within home dir
video_output_path = os.path.join(user_home, 'Downloads')

# download youtube video and save locally
video_url = 'https://youtu.be/-CeLBsqU6qw'

yt = YouTube(video_url)

# get highest res 
video_stream = yt.streams.get_highest_resolution()

# path to download video
video_stream.download(video_output_path)
video_path = os.path.join(video_output_path, video_stream.default_filename)

# put url for youtube video here
cam = cv.VideoCapture(video_path)

while True:

    # capture video frames
    ret, frame = cam.read()

    # break if video frame not captured
    if not ret:
        print('Error: ret = False')
        break

    # convert to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # represents the bounding box of a detected face [(left(), top()) and (width(), height())]
    faces = detector(gray)
    print(f'Faces detected: {len(faces)}')

    # iterate over each detected face in the faces list.
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()

        print(f'Box cordinates:\nLeft corner: {x}, Top corner: {y}, Width: {w}, Height: {h}\n')
        # draw rectangle -> frame=input, (x, y)=top left cordinate 
        # (x+w, y+h)=bottom right cord. x+w=x-cord(horizontal pos) of bottom right, y+h=y-cord(vertical pos) of bottom-right corner.
        # (0,255,0)=color of box
        # , 4)=rectangle thickness
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

        # facial landmark detection 
        # compute the 68 facial landmark points for the face
        facial_landmarks = predictor(gray, face)
        # convert facial_landmars into numpy arr for easier maniulation and computations on facial landmark points
        facial_landmarks = face_utils.shape_to_np(facial_landmarks)

        # add dots to predicted facial landmarks 
        for (x, y) in facial_landmarks:
            cv.circle(frame, (x, y), 4, (0, 0, 255), -1)

        # output face when face is detected
        if len(faces) < 2:
            cv.putText(frame, 'Face', (10,30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif len(faces) > 1:
            cv.putText(frame, 'Faces', (10,30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv.imshow('Youtube Video', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv.destroyAllWindows()


