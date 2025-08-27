import numpy as np
import os
import glob
import random
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend as K
import pandas as pd
from skimage import morphology, measure
import networkx as nx
import re
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from patchify import patchify, unpatchify
from skimage import morphology, img_as_bool, img_as_ubyte, color, io
from skimage.morphology import skeletonize
from skan import Skeleton, summarize, draw
from skan.csr import skeleton_to_csgraph

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda

# 
# NOTE TO USER 
# Before using the pipeline:
#  - As the memory takes too much you need to provide three driectory paths:
#       - One input path of the raw images
#       - Two output paths, one for the predicted masks and one for the csv file with the pixel lenghts and coordinates
#  - Then the pipeline takes the raw images it saves the predicted masks in the specified directory then it analyzes their structure-
#    and outputs a csv folder with all of the necessary information
#

def crop_petri_dish(image):

    if len(image.shape) > 2:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    # apply Gaussian Blur to smooth the image
    blurred = cv2.GaussianBlur(gray_image, (9, 9), 0)
    # apply binary threshold
    _, thresh = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY) 
    
    # finding contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # filtering out very large contours which might be the image borders
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) < image.shape[0] * image.shape[1] * 0.95]

    # findng the largest contour which will be the petri dish
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # modify the bounding rectangle to crop more tightly, the margin can be changed
    margin = 90
    x += margin
    y += margin
    w -= 2 * margin
    h -= 2 * margin
    
    # ensuring the modified coordinates are within image bounds
    x = max(x, 0)
    y = max(y, 0)
    w = min(w, image.shape[1] - x)
    h = min(h, image.shape[0] - y)

    # cropping the image based on calculated coordinates
    cropped_image = image[y:y+h, x:x+w]

    return cropped_image, (x, y, w, h)

def apply_crop(image, bbox):
    x, y, w, h = bbox
    return image[y:y+h, x:x+w]

def padder(image, patch_size):
    h = image.shape[0]
    w = image.shape[1]
    height_padding = ((h // patch_size) + 1) * patch_size - h
    width_padding = ((w // patch_size) + 1) * patch_size - w

    top_padding = int(height_padding/2)
    bottom_padding = height_padding - top_padding

    left_padding = int(width_padding/2)
    right_padding = width_padding - left_padding

    image = cv2.copyMakeBorder(image, top_padding, bottom_padding, left_padding, right_padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return image

def f1(y_true, y_pred):
    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = TP / (Positives+K.epsilon())
        return recall
    
    def precision_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = TP / (Pred_Positives+K.epsilon())
        return precision
    
    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)
    
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def extract_number(filename):
    parts = re.findall(r'\d+', filename)
    return int(parts[-1]) if parts else 0

# loading the trained model
patch_size = 128
example_file_name = 'Main_Model_256px.h5'
model = load_model(example_file_name, custom_objects={"f1": f1})

image_dir = ""
output_dir = ""
os.makedirs(output_dir, exist_ok=True)

image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")), key=extract_number)

# code already explained in another notebook
for idx, img_path in enumerate(image_paths):
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    original_size = image.shape[:2]
    image = cv2.resize(image, (0, 0), fx=1.0, fy=1.0)

    cropped_image, bbox, *_ = crop_petri_dish(image)
    image_padded = padder(cropped_image, patch_size)
    image_patches = patchify(image_padded, (patch_size, patch_size, 3), step=patch_size)
    image_patches_reshaped = image_patches.reshape(-1, patch_size, patch_size, 3)
    predicted_mask = model.predict(image_patches_reshaped / 255)
    
    i, j = image_patches.shape[0], image_patches.shape[1]
    predicted_mask = predicted_mask.reshape(i, j, patch_size, patch_size)
    target_shape = (i * patch_size, j * patch_size)
    predicted_mask = unpatchify(predicted_mask, target_shape)

    output_path = os.path.join(output_dir, f"predicted_mask_{idx}.png")
    cv2.imwrite(output_path, predicted_mask * 255)

def sort_numerically(file_list):
    """Sort file list numerically based on numbers in filenames."""
    return sorted(file_list, key=lambda x: int(re.search(r'\d+', x).group()))

# skeletonizes image
def skeletonize_image(image):
    """Convert image to binary and apply skeletonization."""
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    skeleton = morphology.skeletonize(binary_image > 0)
    return skeleton

# finding the longest path from node to node drawing a the path on the root
def analyze_skeleton(skeleton):
    """Analyze skeleton to find the longest path using Dijkstra's algorithm."""
    # labek the connected components
    labeled_skeleton, _ = measure.label(skeleton, connectivity=2, return_num=True)
    properties = measure.regionprops(labeled_skeleton)

    if properties:
        # identofy the biggest component 
        largest_component = max(properties, key=lambda x: x.area).label
        main_skeleton = labeled_skeleton == largest_component
        graph = nx.Graph()
        for r, c in np.argwhere(main_skeleton):
            # check neighbouring pixels in an 8 connected neighbourhood
            for offsets in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                rr, cc = r + offsets[0], c + offsets[1]
                # if the neighboring pixel is within bounds and part of the skeleton add an edge
                if 0 <= rr < skeleton.shape[0] and 0 <= cc < skeleton.shape[1] and main_skeleton[rr, cc]:
                    graph.add_edge((r, c), (rr, cc))

        if len(graph.nodes) > 1:  # ensuring that there are enough nodes for a path ti be formed
            path = nx.dijkstra_path(graph, source=list(graph.nodes())[0], target=list(graph.nodes())[-1]) #finding the largest path using dijkstras algorythm
            length = nx.dijkstra_path_length(graph, source=list(graph.nodes())[0], target=list(graph.nodes())[-1])

            # ignore paths shorter than 30 pixels
            if length < 30:
                return [], 0, None

            root_tip = path[-1]
            return path, length, root_tip
    return [], 0, None  # needs to be 3 values are returned in all cases

def process_folder_to_csv_numerical(output_dir, output_csv):
    """Process a folder of images, save path lengths to CSV, and overlay paths on images."""
    results = []
    output_overlay_folder = os.path.dirname(output_csv)
    os.makedirs(output_overlay_folder, exist_ok=True)

    image_files = sort_numerically([f for f in os.listdir(output_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    for image_idx, image_file in enumerate(image_files, start=1):
        image_path = os.path.join(output_dir, image_file)
        original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

        # ignore the top 10% of the image
        ignore_top = int(gray_image.shape[0] * 0.10)
        filtered_gray_image = gray_image[ignore_top:, :]

        segment_width = filtered_gray_image.shape[1] // 5

        for segment_idx in range(5):
            x_start = segment_idx * segment_width
            x_end = (segment_idx + 1) * segment_width if segment_idx < 4 else filtered_gray_image.shape[1]

            cropped_segment = filtered_gray_image[:, x_start:x_end]
            skeletonized_segment = skeletonize_image(cropped_segment)
            path, length, root_tip = analyze_skeleton(skeletonized_segment)

            # generating Plant ID in the correct order
            plant_id = f"test_image_{image_idx}_plant_{segment_idx + 1}"

            if not path or length < 30:
                length = 0  # assign 0 if the path is below the threshold

            # appending to results in the correct numerical order for submission
            results.append({"Plant ID": plant_id, "Length (px)": length, "Root Tip (x, y)": root_tip})

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

output_csv = ""

process_folder_to_csv_numerical(output_dir, output_csv)