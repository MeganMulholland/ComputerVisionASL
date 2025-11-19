# USAGE
# python gather_examples.py --conf config/config.json

# import the necessary packages
from pyimagesearch.utils import Conf
from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import os
import subprocess
import re


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True,
    help="path to the input configuration file")
args = vars(ap.parse_args())

# load the configuration file
conf = Conf(args["conf"])

# grab the top-left and and bottom-right (x, y)-coordinates for the
# gesture capture area
TOP_LEFT = tuple(conf["top_left"])
BOT_RIGHT = tuple(conf["bot_right"])

#  grab the key => class label mappings from the configuration
MAPPINGS = conf["mappings"]

# loop over the mappings
for (key, label) in list(MAPPINGS.items()):
    # update the mappings dictionary to use the ordinal value of the
    # key (the key value will be different on varying operating
    # systems)
    MAPPINGS[ord(key)] = label
    del MAPPINGS[key]

# grab the set of valid keys from the mappings dictionary
validKeys = set(MAPPINGS.keys())

# initialize the counter dictionary used to count the number of times
# each key has been pressed
keyCounter = {}

# start the video stream thread
print("[INFO] starting video stream thread...")

vs = VideoStream(usePiCamera=False).start()


# loop over frames from the video stream
while True:
    # grab the frame from the threaded video file stream
    frame = vs.read()

    # resize the frame and then flip it horizontally
    frame = imutils.resize(frame, width=500)
    frame = cv2.flip(frame, 1)

    # extract the ROI from the frame, convert it to grayscale,
    # and threshold the ROI to obtain a binary mask where the
    # foreground (white) is the hand area and the background (black)
    # should be ignored
    roi = frame[TOP_LEFT[1]:BOT_RIGHT[1], TOP_LEFT[0]:BOT_RIGHT[0]]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    #roi = cv2.threshold(roi, 75, 255, cv2.THRESH_BINARY)[1]
    roi = cv2.adaptiveThreshold(roi,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 2)

    # clone the original frame and then draw the gesture capture area
    clone = frame.copy()
    cv2.rectangle(clone, TOP_LEFT, BOT_RIGHT, (0, 0, 255), 2)

    # show the output frame and ROI, and then record if a user presses
    # a key
    cv2.imshow("Frame", clone)
    cv2.imshow("ROI", roi)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    # otherwise, check to see if a key was pressed that we are
    # interested in capturing
    elif key in validKeys:
        # construct the path to the label subdirectory
        p = os.path.sep.join([conf["dataset_path"], MAPPINGS[key]])

        # if the label subdirectory does not already exist, create it
        #print(p)
        if not os.path.exists(p):
            print(p)
            os.makedirs(p, exist_ok = True)

        # construct the path to the output image
        p = os.path.sep.join([p, "{}.png".format(
            keyCounter.get(key, 0))])
        keyCounter[key] = keyCounter.get(key, 0) + 1

        # save the ROI to disk
        print("[INFO] saving ROI: {}".format(p))
        cv2.imwrite(p, roi)