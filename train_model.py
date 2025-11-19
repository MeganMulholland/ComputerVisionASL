# USAGE
# python train_model.py --conf config/config.json --filter 2

# import the necessary packages
from ctypes.wintypes import PLARGE_INTEGER
from gc import callbacks
import cv2
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pyimagesearch.utils import Conf
from imutils import paths
import numpy as np
import argparse
import pickle
import os
import platform
import tensorflow as tf
from packaging import version
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input, Flatten
from tensorflow.keras.models import Model

# check tensorflow version
current_version = version.parse(tf.__version__)
print("tf version: " + tf.__version__)
desired_version = version.parse("2.10.0")
print("Version higher than 2.10.0: " + str(current_version >= desired_version))

if current_version >= desired_version:
    from tensorflow.keras.optimizers.legacy import Adam
else:
    from tensorflow.keras.optimizers import Adam

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True,
                help="path to the input configuration file")
ap.add_argument("-f", "--filter", required=True,
                help="[the chosen filter]: filter 1 - regular image; \
                    filter 2 - image with filters ")
ap.add_argument("-m", "--model", required=True,
                help="model chosen")
args = vars(ap.parse_args())

# load the configuration file
conf = Conf(args["conf"])

# load the filter type
chosen_filter = args["filter"]
print("Filter " + chosen_filter + " has been chosen")

# load model type
model_id = args["model"]
print("Model " + model_id + " has been chosen")

# model chosen method
if int(model_id) == 1:
    from pyimagesearch.nn.gesturenet import GestureNet
    model_name = GestureNet
    print("CNN Base")
elif int(model_id) == 2:
    from pyimagesearch.nn.gesturenetres import GestureNetRes
    model_name = GestureNetRes
    print("CNN+ResBlock")
elif int(model_id) == 3:
    print("MobileNetV3")
else:
    raise argparse.ArgumentTypeError("%s is an invalid int value -- model input error" % model_id)

# platform recognition
os_name = platform.system()

# ------------------- IMAGE LOADING -------------------
print("[INFO] loading images...")

# go directly to the existing dataset folder
dataset_folder = conf["dataset_path"]  # "datasets/raw_hand_gesture_dataset"
print(f"[DEBUG] Looking for images in: {dataset_folder}")

# grab the list of images
imagePaths = list(paths.list_images(dataset_folder))
imagePaths = [file for file in imagePaths if not os.path.basename(file).startswith('.')]
print(f"[DEBUG] Found {len(imagePaths)} images")

# initialize data and labels lists
data = []
labels = []

# loop over the image paths
for imagePath in imagePaths:
    # extract the class label from the filename
    label = imagePath.split(os.path.sep)[-2]

    # load the image, convert it to grayscale (if needed), and resize
    image = cv2.imread(imagePath)
    if int(chosen_filter) == 1 and int(model_id) == 3:
        pass  # keep RGB for MobileNetV3 + filter 1
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (64, 64))

    # update the data and labels lists
    data.append(image)
    labels.append(label)

# convert the data into a NumPy array and scale pixel intensities
data = np.array(data, dtype="float") / 255.0

# ------------------- RESHAPE FIX -------------------
# reshape the data to include a channel dimension
if len(data.shape) == 3:
    # grayscale images
    data = np.expand_dims(data, axis=-1)
elif len(data.shape) == 4 and data.shape[3] not in [1, 3]:
    # unknown shape
    raise ValueError(f"Unexpected data shape: {data.shape}")
# else RGB images already have correct channel dimension

# one-hot encode the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# partition the data into training and testing splits
(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size=0.25, stratify=labels, random_state=42)

# construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
                         width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
                         horizontal_flip=True, fill_mode="nearest")

# ------------------- MODEL BUILD -------------------
if int(model_id) == 1 or (int(model_id) == 2 and int(chosen_filter) != 1):
    # initialize gesture recognition CNN and compile it
    model = model_name.build(64, 64, 1, len(lb.classes_))

if int(model_id) == 3:
    if int(chosen_filter) != 1:
        base_model = MobileNetV3Small(input_shape=(64, 64, 1), include_top=False, weights=None)
    else:
        base_model = MobileNetV3Small(input_shape=(64, 64, 3), include_top=False, weights=None)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    predictions = Dense(len(lb.classes_), activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=predictions)

# compile the model
opt = Adam(learning_rate=conf["init_lr"], decay=conf["init_lr"] / conf["num_epochs"])
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
print(model.summary())

# ------------------- TRAIN NETWORK -------------------
H = model.fit(
    aug.flow(trainX, trainY, batch_size=conf["bs"]),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // conf["bs"],
    epochs=conf["num_epochs"]
)

# ------------------- EVALUATE NETWORK -------------------
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=conf["bs"])
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1), target_names=lb.classes_))

# ------------------- SAVE MODEL AND LABEL ENCODER -------------------
print("[INFO] saving model...")
model.save(str(conf["model_path"]))

print("[INFO] serializing label encoder...")
with open(str(conf["lb_path"]), "wb") as f:
    f.write(pickle.dumps(lb))
