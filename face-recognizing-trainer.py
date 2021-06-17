import os
from PIL import Image
import numpy as np
import pickle
import cv2
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "samples")

id_label = 0
label_ids = {}
labels = []
training_data = []

recognizer = cv2.face.LBPHFaceRecognizer_create()

# get sample images
for root, dirs, files in os.walk(image_dir):

    for file in tqdm(files):
        if file.endswith("png"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-").lower()
            if label not in label_ids:
                label_ids[label] = id_label
                id_label += 1
            _id = label_ids[label]
            # print(f"{label}: {path}")
            pil_image = Image.open(path).convert("L")
            img_array = np.array(pil_image, "uint8")
            training_data.append(img_array)
            labels.append(_id)

# dump labels
with open("labels.pickle", "wb") as f:
    pickle.dump(label_ids, f)

recognizer.train(training_data, np.array(labels))
recognizer.save("classifier.yml")
