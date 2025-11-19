# USAGE
# python recognize_regular.py --conf config/config.json

# import the necessary packages
import cv2
from pyimagesearch.utils import Conf
from imutils.video import VideoStream
from imutils import paths
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from datetime import datetime
from datetime import date
import numpy as np
import argparse
import imutils
import pickle
import time
import os
import re
import tensorflow as tf

def showInMovedWindow(winname, img, x, y):
    cv2.namedWindow(winname)        # Create a named window
    cv2.moveWindow(winname, x, y)   # Move it to (x,y)
    cv2.imshow(winname,img)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True,
    help="path to the input configuration file")
args = vars(ap.parse_args())

# load the configuration file
conf = Conf(args["conf"])


# grab the paths to gesture icon images and then initialize the icons
# dictionary where the key is the gesture name (derived from the image
# path) and the key is the actual icon image
print("[INFO] loading icons...")
imagePaths = paths.list_images(conf["assets_path"])
icons = {}

# loop over the image paths
for imagePath in imagePaths:
    # extract the gesture name (label) the icon represents from the
    # filename, load the icon, and then update the icons dictionary
    filename = os.path.basename(imagePath)
    if not re.match(r'^[a-zA-Z0-9_-]+\.png$', filename):
        print(f"[INFO] Skipping file: {filename}")
        continue
    label = filename.split(".")[0]
    icon = cv2.imread(imagePath)
  
    # Check if the image was loaded successfully
    if icon is None:
        print(f"[ERROR] Unable to load image: {imagePath}")
        continue

    # Resize the image
    try:
        icon = cv2.resize(icon, (75, 100))
    except cv2.error as e:
        print(f"[ERROR] Failed to resize image: {imagePath}, error: {e}")
        continue
    icons[label] = icon

# grab the top-left and and bottom-right (x, y)-coordinates for the
# gesture capture area
TOP_LEFT = tuple(conf["top_left"])
BOT_RIGHT = tuple(conf["bot_right"])

# load the trained gesture recognizer model and the label binarizer
print("[INFO] loading model...")
model = load_model(str(conf["model_path"]))
lb = pickle.loads(open(str(conf["lb_path"]), "rb").read())

# start the video stream thread
print("[INFO] starting video stream thread...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# initialize the current gesture, a bookkeeping variable used to keep
# track of the number of consecutive frames a given gesture has been
# classified as
currentGesture = [None, 0]

# initialize the list of input gestures recognized from the user
# along with the timestamp of when all four gestures were entered
gestures = []
enteredTS = None

# initialize two booleans used to indicate (1) whether or not the
# alarm has been raised and (2) if the correct pass code was entered
alarm = False
correct = False

# loop over frames from the video stream
while True:
    # grab the frame from the threaded video file stream and grab the
    # current timestamp
    frame = vs.read()
    timestamp = datetime.now()

    # resize the frame and then flip it horizontally
    frame = imutils.resize(frame, width=500)
    frame = cv2.flip(frame, 1)

    # clone the original frame and then draw the gesture capture area
    clone = frame.copy()
    cv2.rectangle(clone, TOP_LEFT, BOT_RIGHT, (0, 0, 255), 2)

    # only perform hand gesture classification if the current gestures
    # list is not already full
    if len(gestures) < 1:
        # extract the hand gesture capture ROI from the frame, convert
        # the ROI to grayscale, and then threshold it to reveal a
        # binary mask of the hand
        #roi = frame[TOP_LEFT[1]:BOT_RIGHT[1], TOP_LEFT[0]:BOT_RIGHT[0]]
        
        # now that we have the hand region we need to resize it to be
        # the same dimensions as what our model was trained on, scale
        # it to the range [0, 1], and prepare it for classification
        #roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        #roi = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        #visROI = roi.copy() # add filters here similar to gather examples
        roi = frame[TOP_LEFT[1]:BOT_RIGHT[1], TOP_LEFT[0]:BOT_RIGHT[0]]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi = cv2.adaptiveThreshold(roi, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)

        # Show the same black & white ROI used for model prediction
        visROI = roi

        roi = cv2.resize(roi, (64, 64))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
    
        # classify the input image
        proba = model.predict(roi)[0]
        label = lb.classes_[proba.argmax()]
        print(label)
        
        # check to see if the label from our model matches the label
        # from the previous classification
        if label == currentGesture[0] and label != "ignore":
            # increment the current gesture count
            currentGesture[1] += 1

            # check to see if the current gesture has been recognized
            # for the past N consecutive frames
            if currentGesture[1] == conf["consec_frames"]:
                # update the gestures list with the predicted label
                # and then reset the gesture counter
                gestures.append(label)
                currentGesture = [None, 0]

        # otherwise, reset the current gesture count
        else:
            currentGesture = [label, 0]

    # initialize the canvas used to draw recognized gestures
    canvas = np.zeros((250, 350, 3), dtype="uint8")

    # loop over the number of hand gesture input keys
    for i in range(0, 1):
        # compute the starting x-coordinate of the entered gesture
        x = (i * 100) + 135

        # check to see if an input gesture exists for this index, in
        # which case we should display an icon
        if len(gestures) > i:
            #print (i)
            canvas[65:165, x:x + 75] = icons[gestures[i]]

        # otherwise, there has not been an input gesture for this icon
        else:
            # draw a white box on the canvas, indicating that a
            # gesture has not been entered yet
            cv2.rectangle(canvas, (x, 65), (x + 75, 165),
               (255, 255, 255), -1)

    # initialize the status as "waiting" (implying that we're waiting
    # for the user to input four gestures) along with the color of the
    # status text
    status = "Pending Prediction ..."
    color = (255, 255, 255)

    # check to see if there are four gestures in the list, implying
    # that we need to check the pass code
    if len(gestures) == 1:
        # if the timestamp of when the four gestures has been entered
        # has not been initialized, initialize it
        if enteredTS is None:
            enteredTS = timestamp

        # initialize our status, color, and sound path for the
        # "correct" pass code
        status = "Prediction Complete"
        color = (0, 255, 0)
        
        # after a correct/incorrect pass code we will show the status
        # for N seconds
        if (timestamp - enteredTS).seconds > conf["num_seconds"]:
            # reset the gestures list, timestamp
            gestures = []
            enteredTS = None

    # draw the timestamp and status on the canvas
    ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
    status = "Status: {}".format(status)
    cv2.putText(canvas, ts, (10, canvas.shape[0] - 10),
        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 1)
    if status == "Status: Prediction Complete":
        if len(gestures) == 1:
            txt = "Prediction Result: {}".format(str(gestures[0]).capitalize())
            cv2.putText(canvas, txt, (10, canvas.shape[0] - 40),
                cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 255, 255), 1)
    cv2.putText(canvas, status, (10, 25), cv2.FONT_HERSHEY_COMPLEX,
        0.6, color, 1)

    # show ROI we're monitoring, the output frame, and passcode info
    showInMovedWindow("ROI", visROI, 601, 0)
    showInMovedWindow("Frame", clone, 100, 0)
    showInMovedWindow("Prediction", canvas, 300, 310)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
