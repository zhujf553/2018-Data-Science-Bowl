import os
import sys
import random
import math
import re
import time
import numpy as np
import pandas as pd
import cv2
import matplotlib
import matplotlib.pyplot as plt
from skimage.transform import resize
from config import Config
import utils
import train
import validation
import model as modellib
import visualize
from model import log
from tqdm import tqdm

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
    
if sys.argv[1] == "train":
    X_train, Y_train = utils.load_train_data()
    x_train, y_train, x_val, y_val = utils.split_train_val(X_train, Y_train)
    train.train_stage1(MODEL_DIR, COCO_MODEL_PATH, x_train, y_train, x_val, y_val)
    train.train_stage2(MODEL_DIR, x_train, y_train, x_val, y_val)
    train.train_stage3(MODEL_DIR, x_train, y_train, x_val, y_val)
    train.train_stage4(MODEL_DIR, x_train, y_train, x_val, y_val)
    train.train_stage5(MODEL_DIR, x_train, y_train, x_val, y_val)
    
elif sys.argv[1] == "validate":
    log_path = "logs/nuclei20180325T2205/mask_rcnn_nuclei_{:04d}.h5" # change to the path of saved models
    X_train, Y_train = utils.load_train_data()
    x_train, y_train, x_val, y_val = utils.split_train_val(X_train, Y_train)
    
    # Validation dataset
    dataset_val = utils.NuDataset()
    dataset_val.load_nuclei(x_val, y_val)
    dataset_val.prepare()

    val_images = []
    val_masks = []
            
    class InferenceConfig(train.NuConfig2):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        DETECTION_MAX_INSTANCES = 200
        IMAGE_MIN_DIM = 512
        IMAGE_MAX_DIM = 1024
        IMAGE_PADDING = True

    inference_config = InferenceConfig()
        
    for image_id in tqdm(dataset_val.image_ids):
        original_image, _, _, _, true_masks = modellib.load_image_gt(dataset_val, inference_config, image_id, use_mini_mask=False)
        val_images.append(original_image)
        val_masks.append(true_masks)

    va0 = validation.my_validation(val_images, val_masks, inference_config)
    best_mAP = 0
    best_epoch = 0
	# change to the range of epoch you want to validate
    for i in range(162, 171):
        print("epoch " + str(i))
        mAP = va0.validate(MODEL_DIR, path=log_path.format(i))
        if mAP > best_mAP:
            best_mAP = mAP
            best_epoch = i
            
    print("Best epoch: " + str(best_epoch))
    
