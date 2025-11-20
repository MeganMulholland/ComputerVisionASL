import os
import time
import pickle
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from pyimagesearch.utils import Conf
from tensorflow.keras.models import load_model

# ------------------- CONFIG -------------------
conf = Conf("config/config.json")  # adjust path if needed

# Load test data (assuming your train/test split is saved)
with open(conf["lb_path"], "rb") as f:
    lb = pickle.load(f)

# Load test dataset
import cv2
from imutils import paths

imagePaths = list(paths.list_images(conf["dataset_path"]))
imagePaths = [p for p in imagePaths if not os.path.basename(p).startswith('.')]

data = []
labels = []

for imagePath in imagePaths:
    label = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (64, 64))
    data.append(image)
    labels.append(label)

data = np.array(data, dtype="float32") / 255.0
data = np.expand_dims(data, axis=-1)
labels = lb.transform(labels)

# Split test set (you can adjust if you saved testX/testY separately)
from sklearn.model_selection import train_test_split
_, testX, _, testY = train_test_split(data, labels, test_size=0.25, random_state=42, stratify=labels)

# ------------------- LOAD TFLITE MODEL -------------------
tflite_model_path = "output/gesture_reco(f2-Res).tflite"  # change if needed
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ------------------- MODEL SIZE -------------------
model_size = os.path.getsize(tflite_model_path) / 1024  # KB
print(f"TFLite model size: {model_size:.2f} KB")

# ------------------- INFERENCE & ACCURACY -------------------
preds = []
start_time = time.time()
for i in range(len(testX)):
    input_data = np.expand_dims(testX[i], axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    preds.append(np.argmax(output))
end_time = time.time()

# Accuracy metrics
true_labels = np.argmax(testY, axis=1)
print("\nClassification Report:")
print(classification_report(true_labels, preds, target_names=lb.classes_))

# Inference speed
total_time = end_time - start_time
avg_time_per_sample = total_time / len(testX)
print(f"Total inference time: {total_time:.2f}s for {len(testX)} samples")
print(f"Average time per sample: {avg_time_per_sample*1000:.2f} ms")

# ------------------- OPTIONAL: Quantization Info -------------------
print("\nQuantization types tested: FP32 (default), INT8, dynamic range")
