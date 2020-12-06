''''| Author: Jean Vitor de Paulo
	| Date: 29/09/2018
	| 
'''


import cv2 
import numpy as np
from PIL import Image

NoneType = type(None)


def _detect(image, bc=None):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # converting from gbr to hsv color space
    img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # skin color range for hsv color space 
    HSV_mask = cv2.inRange(img_HSV, (0, 40, 0), (25,255,255)) 
    HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    # converting from gbr to YCbCr color space
    img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    # skin color range for hsv color space 
    YCrCb_mask = cv2.inRange(img_YCrCb, (0, 138, 67), (255,173,133)) 
    YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    # merge skin detection (YCbCr and hsv)
    image_mask = cv2.bitwise_and(YCrCb_mask, HSV_mask)
    image_mask = cv2.medianBlur(image_mask, 3)
    image_mask = cv2.morphologyEx(image_mask, cv2.MORPH_OPEN, np.ones((4,4), np.uint8))
    
    # constructing global mask
    image_mask = cv2.erode(image_mask, None, iterations = 3)  # remove noise
    image_mask = cv2.dilate(image_mask, None, iterations = 3)  # smoothing eroded mask
    
    return image_mask


def detect_skin(image, bc=None, return_imgarray=False):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    image_mask = _detect(image)
    if type(bc) != NoneType:
        bc_mask = _detect(bc)
        image_mask = image_mask | bc_mask
    
    output = cv2.bitwise_and(img, img, mask = image_mask)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    image_mask = cv2.cvtColor(image_mask, cv2.COLOR_GRAY2RGB)
    
    if return_imgarray:
        return img, image_mask, output
    
    img = Image.fromarray(img)
    image_mask = Image.fromarray(image_mask)
    output = Image.fromarray(output)
    return img, image_mask, output
