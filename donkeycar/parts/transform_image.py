import numpy as np
import cv2
import time
import random
import collections
from PIL import Image
import os
import urllib.request


class TransformImage(object):

    def __init__(self, debug=False):
        self.debug = debug

    def adjust_gamma(self, image, gamma=1.0):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")

        # apply gamma correction using the lookup table
        return cv2.LUT(image, table)

    def region_of_interest(self, img):
        # Define a blank matrix that matches the image height/width.
        mask = np.zeros_like(img)
        # Retrieve the number of color channels of the image.
        channel_count = img.shape[2]
        # Create a match color with the same color channel counts.
        match_mask_color = (255,) * channel_count

        # Fill inside the polygon
        #cv2.fillPoly(mask, vertices, match_mask_color)
        cv2.rectangle(mask,(0,0),(320,240),match_mask_color,-10)
        # Returning the image only where mask pixels match
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def region_to_crop(self, img):

        #print('here in crop')
        y = 80
        x = 0
        h = 240
        w = 320
        crop_img = img[y:h, x:w]
        return crop_img

    def region_of_interest_proc(self, img):
        # Define a blank matrix that matches the image height/width.
        mask = np.zeros_like(img)
        # Retrieve the number of color channels of the image.
    # channel_count = img.shape[2]
        # Create a match color with the same color channel counts.
        match_mask_color = 255

        # Fill inside the polygon
        #cv2.fillPoly(mask, vertices, match_mask_color)
        cv2.rectangle(mask,(0,20),(320,240),match_mask_color,-10)
        # Returning the image only where mask pixels match
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image


    def lineDetection(self, srcImage):
        srcImage = self.adjust_gamma(srcImage, 4.0)
        srcImage = cv2.cvtColor(srcImage, cv2.COLOR_BGR2RGB)
        srcImage = self.region_of_interest(srcImage)
        srcImage = cv2.Canny(srcImage, 100, 200)
        srcImage = self.region_to_crop(srcImage)
        return srcImage
    
    def preprocess(self, image):
        height, _, _ = image.shape
        image = image[int(height/5):,:,:]  # remove 20% of the image, as it is not relevant for lane following
        #image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)  # Nvidia model said it is best to use YUV color space
        #image = cv2.GaussianBlur(image, (3,3), 0)
        #image = cv2.resize(image, (200,66)) # input image size (200,66) Nvidia model
        #image = image / 255 # normalizing
        return image

    def run(self, img_arr, debug=False):
        if img_arr is None:
            return img_arr
        img = self.preprocess(img_arr)
        #img = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
        #val = self.lineDetection(img)
        return img