from pyimagesearch.utils import Conf
import tensorflow as tf
import argparse
import pickle
import os
# From Dr. QC

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True,
	help="path to the input configuration file")
args = vars(ap.parse_args())
# load the configuration file
conf = Conf(args["conf"])

model = tf.keras.models.load_model(str(conf["model_path"]))
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
tflite_model_path = str(conf["model_path"]).replace(".h5", ".tflite")
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)