import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

from config import Config
import utils
import model as modellib
import visualize
from model import log

from skimage import transform
from skimage.transform import resize
from skimage import color

class NuConfig1(Config):
    NAME = "nuclei"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 

    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 512

    RPN_ANCHOR_SCALES = (4, 8, 16, 32, 64)  # anchor side in pixels
    TRAIN_ROIS_PER_IMAGE = 500
    STEPS_PER_EPOCH = 630
    VALIDATION_STEPS = 34 
    MEAN_PIXEL = [0,0,0]
    LEARNING_RATE = 0.001
    USE_MINI_MASK = True
    MAX_GT_INSTANCES = 500

class NuConfig2(Config):
    NAME = "nuclei"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 

    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 1024

    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels
    TRAIN_ROIS_PER_IMAGE = 500
    STEPS_PER_EPOCH = 630
    VALIDATION_STEPS = 34 
    MEAN_PIXEL = [0,0,0]
    LEARNING_RATE = 0.001
    USE_MINI_MASK = True
    MAX_GT_INSTANCES = 500

class NuConfig3(Config):
    NAME = "nuclei"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 

    IMAGE_MIN_DIM = 768
    IMAGE_MAX_DIM = 1024

    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels
    TRAIN_ROIS_PER_IMAGE = 500
    STEPS_PER_EPOCH = 630
    VALIDATION_STEPS = 34 
    MEAN_PIXEL = [0,0,0]
    LEARNING_RATE = 0.001
    USE_MINI_MASK = True
    MAX_GT_INSTANCES = 500
    
class NuDataset1(utils.NuDataset):
    # augmentation 1
    def data_augment(self, image,masks,angle=180):
        original_shape = image.shape

        color_delta = [np.random.randint(50),np.random.randint(50),np.random.randint(50)]
        image = image + color_delta
        image = np.clip(image, 0, 255).astype(np.uint8)
            
        if random.randint(0, 1):
            image = np.fliplr(image)
            masks = np.fliplr(masks)
        
        image = resize(image, (original_shape[0], original_shape[1]), preserve_range=True, mode='constant')
        masks = resize(masks, (original_shape[0], original_shape[1]), preserve_range=True, mode='constant',order=0)

        return image, masks
        
    def load_and_augment(self, config, image_id, augment):
        image = self.load_image(image_id)
        mask, class_ids = self.load_mask(image_id)
        shape = image.shape
        image, window, scale, padding = utils.resize_image(
            image,
            min_dim=config.IMAGE_MIN_DIM,
            max_dim=config.IMAGE_MAX_DIM,
            padding=config.IMAGE_PADDING)
        mask = utils.resize_mask(mask, scale, padding)

        if augment:
            image, mask = self.data_augment(image, mask)
            
        return image, mask, class_ids, window, shape

class NuDataset2(utils.NuDataset):
    # augmentation 2
    def data_augment(self, image,masks,angle=180):
        original_shape = image.shape

        sh = random.random()/2-0.25
        rotate_angle = random.random() * angle
        
        image = transform.rotate(image, rotate_angle, resize=True, preserve_range=True)
        masks = transform.rotate(masks, rotate_angle, resize=True, preserve_range=True, order=0)
        
        affine_shear = transform.AffineTransform(shear=sh)
        image = transform.warp(image, inverse_map=affine_shear, preserve_range=True, mode='constant')
        masks = transform.warp(masks, inverse_map=affine_shear, preserve_range=True, mode='constant',order=0)
        
        if random.randint(0, 1):
            image = np.fliplr(image)
            masks = np.fliplr(masks)
        
        image = resize(image, (original_shape[0], original_shape[1]), preserve_range=True, mode='constant')
        masks = resize(masks, (original_shape[0], original_shape[1]), preserve_range=True, mode='constant',order=0)
        
        exclude = np.where(np.sum(masks,axis=(0,1)) <= 10)[0]
        masks = np.delete(masks, exclude, axis=2)
        
        #shift in rgb space
        color_delta = [np.random.randint(-50,50),np.random.randint(-50,50),np.random.randint(-50,50)]
        image = image + color_delta
        image = np.clip(image, 0, 255).astype(np.uint8)

        #shift in hsv space
        image = color.rgb2hsv(image)
        image = (image * 255.0).astype(np.uint8)
        color_delta = [np.random.randint(-50,50),np.random.randint(-50,50),np.random.randint(-50,50)]
        image = image + color_delta
        image = np.clip(image, 0, 255).astype(np.uint8)
        image = color.hsv2rgb(image)
        image = (image * 255.0).astype(np.uint8)
        
        return image, masks
        
    def load_and_augment(self, config, image_id, augment):
        image = self.load_image(image_id)
        mask, class_ids = self.load_mask(image_id)
        shape = image.shape
        image, window, scale, padding = utils.resize_image(
            image,
            min_dim=config.IMAGE_MIN_DIM,
            max_dim=config.IMAGE_MAX_DIM,
            padding=config.IMAGE_PADDING)
        mask = utils.resize_mask(mask, scale, padding)

        if augment:
            image, mask = self.data_augment(image, mask)
            
        return image, mask, class_ids, window, shape
        
class NuDataset3(utils.NuDataset):
    # augmentation 3
    def data_augment(self, image,masks,angle=180):
        original_shape = image.shape
        original_image = image
        original_masks = masks

        if random.randint(0, 1):
            x1 = int(random.random() * 0.5 * original_shape[0])
            x2 = x1 + int(original_shape[0] / 2)
            y1 = int(random.random() * 0.5 * original_shape[1])
            y2 = y1 + int(original_shape[1] / 2)
            image = image[x1:x2,y1:y2,:]
            masks = masks[x1:x2,y1:y2,:]

        
        sh = random.random()/2-0.25
        rotate_angle = random.random() * angle
        
        image = transform.rotate(image, rotate_angle, resize=False, preserve_range=True)
        masks = transform.rotate(masks, rotate_angle, resize=False, preserve_range=True, order=0)
        
        affine_shear = transform.AffineTransform(shear=sh)
        image = transform.warp(image, inverse_map=affine_shear, preserve_range=True, mode='constant')
        masks = transform.warp(masks, inverse_map=affine_shear, preserve_range=True, mode='constant',order=0)
        
        if random.randint(0, 1):
            image = np.fliplr(image)
            masks = np.fliplr(masks)
        
        image = resize(image, (original_shape[0], original_shape[1]), preserve_range=True, mode='constant')
        masks = resize(masks, (original_shape[0], original_shape[1]), preserve_range=True, mode='constant',order=0)
        
        exclude = np.where(np.sum(masks,axis=(0,1)) <= 10)[0]
        if len(exclude) < masks.shape[2]:
            masks = np.delete(masks, exclude, axis=2)
        else:
            image = original_image
            masks = original_masks
            exclude = []
        
        if random.randint(0, 1):
            #shift in rgb space
            color_delta = [np.random.randint(-50,50),np.random.randint(-50,50),np.random.randint(-50,50)]
            image = image + color_delta
            image = np.clip(image, 0, 255).astype(np.uint8)
        else:
            #shift in hsv space
            image = image.astype(np.uint8)
            image = color.rgb2hsv(image)
            image = (image * 255.0).astype(np.uint8)
            color_delta = [np.random.randint(-50,50),np.random.randint(-50,50),np.random.randint(-50,50)]
            image = image + color_delta
            image = np.clip(image, 0, 255).astype(np.uint8)
            image = color.hsv2rgb(image)
            image = (image * 255.0).astype(np.uint8)
            
        if np.min(np.max(image,axis=(0,1))-np.min(image,axis=(0,1))) < 10:
            image = original_image
            masks = original_masks
            exclude = []
            
        if random.randint(0, 1):
            image[:,:,0] = 255 - image[:,:,0]
        if random.randint(0, 1):
            image[:,:,1] = 255 - image[:,:,1]
        if random.randint(0, 1):
            image[:,:,2] = 255 - image[:,:,2]
        
        masks = masks.astype(np.int32)
        
        return image, masks, exclude
        
    def load_and_augment(self, config, image_id, augment):
        image = self.load_image(image_id)
        mask, class_ids = self.load_mask(image_id)
        
        if augment:
            image, mask, exclude = self.data_augment(image, mask)
            class_ids = np.delete(class_ids, exclude, axis=0)
        
        # Load image and mask
        shape = image.shape
        image, window, scale, padding = utils.resize_image(
            image,
            min_dim=config.IMAGE_MIN_DIM,
            max_dim=config.IMAGE_MAX_DIM,
            padding=config.IMAGE_PADDING)
        mask = utils.resize_mask(mask, scale, padding)
            
        return image, mask, class_ids, window, shape

class NuDataset4(utils.NuDataset):
    # augmentation 4
    def data_augment(self, image,masks,angle=180):
        original_shape = image.shape
        original_image = image
        original_masks = masks

        if random.randint(0, 1):
            x1 = int(random.random() * 0.5 * original_shape[0])
            x2 = x1 + int(original_shape[0] / 2)
            y1 = int(random.random() * 0.5 * original_shape[1])
            y2 = y1 + int(original_shape[1] / 2)
            image = image[x1:x2,y1:y2,:]
            masks = masks[x1:x2,y1:y2,:]

        
        sh = random.random()/2-0.25
        rotate_angle = random.random() * angle
        
        image = transform.rotate(image, rotate_angle, resize=False, preserve_range=True)
        masks = transform.rotate(masks, rotate_angle, resize=False, preserve_range=True, order=0)
        
        affine_shear = transform.AffineTransform(shear=sh)
        image = transform.warp(image, inverse_map=affine_shear, preserve_range=True, mode='constant')
        masks = transform.warp(masks, inverse_map=affine_shear, preserve_range=True, mode='constant',order=0)
        
        if random.randint(0, 1):
            image = np.fliplr(image)
            masks = np.fliplr(masks)
        
        image = resize(image, (original_shape[0], original_shape[1]), preserve_range=True, mode='constant')
        masks = resize(masks, (original_shape[0], original_shape[1]), preserve_range=True, mode='constant',order=0)
        
        exclude = np.where(np.sum(masks,axis=(0,1)) <= 10)[0]
        if len(exclude) < masks.shape[2]:
            masks = np.delete(masks, exclude, axis=2)
        else:
            image = original_image
            masks = original_masks
            exclude = []
            
        image = image.astype(np.uint8)
        
                  
        if random.randint(0, 1):
            image[:,:,0] = 255 - image[:,:,0]
        if random.randint(0, 1):
            image[:,:,1] = 255 - image[:,:,1]
        if random.randint(0, 1):
            image[:,:,2] = 255 - image[:,:,2]
        
        masks = masks.astype(np.int32)
        
        return image, masks, exclude
        
    def load_and_augment(self, config, image_id, augment):
        image = self.load_image(image_id)
        mask, class_ids = self.load_mask(image_id)
        
        if augment:
            image, mask, exclude = self.data_augment(image, mask)
            class_ids = np.delete(class_ids, exclude, axis=0)
        
        # Load image and mask
        shape = image.shape
        image, window, scale, padding = utils.resize_image(
            image,
            min_dim=config.IMAGE_MIN_DIM,
            max_dim=config.IMAGE_MAX_DIM,
            padding=config.IMAGE_PADDING)
        mask = utils.resize_mask(mask, scale, padding)
            
        return image, mask, class_ids, window, shape
        
def train_stage1(MODEL_DIR, COCO_MODEL_PATH, x_train, y_train, x_val, y_val):
    
    print("Training stage 1")
    
    config = NuConfig1()
    config.display()
    
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=MODEL_DIR)
                              
    init_with = "coco"  # imagenet, coco, or last

    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                    "mrcnn_bbox", "mrcnn_mask"])
    
    # Training dataset
    dataset_train = NuDataset1()
    dataset_train.load_nuclei(x_train, y_train)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = NuDataset1()
    dataset_val.load_nuclei(x_val, y_val)
    dataset_val.prepare()
    
    model.train(dataset_train, dataset_val, 
                learning_rate=config.LEARNING_RATE, 
                epochs=50, 
                layers='heads')
                
    model.train(dataset_train, dataset_val, 
                learning_rate=config.LEARNING_RATE,
                epochs=70, 
                layers="all")
            
    model.train(dataset_train, dataset_val, 
                learning_rate=config.LEARNING_RATE / 10,
                epochs=120, 
                layers="all")
            
    return model
    
def train_stage2(MODEL_DIR, x_train, y_train, x_val, y_val):
    print("Training stage 2")
    
    config = NuConfig2()
    config.display()

    # Training dataset
    dataset_train = NuDataset1()
    dataset_train.load_nuclei(x_train, y_train)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = NuDataset1()
    dataset_val.load_nuclei(x_val, y_val)
    dataset_val.prepare()
    
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=MODEL_DIR)
    
    model_path = "logs/nuclei20180412T1229/mask_rcnn_nuclei_0120.h5"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    model.train(dataset_train, dataset_val, 
                learning_rate=config.LEARNING_RATE,
                epochs=160, 
                layers="all")
                
    model.train(dataset_train, dataset_val, 
                learning_rate=config.LEARNING_RATE / 10,
                epochs=180, 
                layers="all")

    return model

def train_stage3(MODEL_DIR, x_train, y_train, x_val, y_val):
    print("Training stage 3")
    
    config = NuConfig2()
    config.display()

    # Training dataset
    dataset_train = NuDataset2()
    dataset_train.load_nuclei(x_train, y_train)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = NuDataset2()
    dataset_val.load_nuclei(x_val, y_val)
    dataset_val.prepare()
	
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=MODEL_DIR)
    
    model_path = "logs/nuclei20180325T2205/mask_rcnn_nuclei_0180.h5"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    model.train(dataset_train, dataset_val, 
                learning_rate=config.LEARNING_RATE,
                epochs=200, 
                layers="all")
                
    return model
                
def train_stage4(MODEL_DIR, x_train, y_train, x_val, y_val):

    print("Training stage 4")
    
    config = NuConfig2()
    config.display()

    # Training dataset
    dataset_train = NuDataset3()
    dataset_train.load_nuclei(x_train, y_train)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = NuDataset3()
    dataset_val.load_nuclei(x_val, y_val)
    dataset_val.prepare()
	
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=MODEL_DIR)
    
    model_path = "logs/nuclei20180325T2205/mask_rcnn_nuclei_0200.h5"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    model.train(dataset_train, dataset_val, 
                learning_rate=config.LEARNING_RATE,
                epochs=220, 
                layers="heads")
                
    return model
	
def train_stage5(MODEL_DIR, x_train, y_train, x_val, y_val):

    print("Training stage 5")
    
    config = NuConfig3()
    config.display()

    # Training dataset
    dataset_train = NuDataset4()
    dataset_train.load_nuclei(x_train, y_train)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = NuDataset4()
    dataset_val.load_nuclei(x_val, y_val)
    dataset_val.prepare()
    
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=MODEL_DIR)
	
    model_path = "logs/nuclei20180325T2205/mask_rcnn_nuclei_0220.h5"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    model.train(dataset_train, dataset_val, 
                learning_rate=config.LEARNING_RATE / 10,
                epochs=250, 
                layers="all")
                
    return model