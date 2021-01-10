''''| Author: Jean Vitor de Paulo
	| Date: 29/09/2018
	| 
'''


import cv2 
import numpy as np
from PIL import Image

import pickle

NoneType = type(None)

with open('src/models/rf_hskin.pkl', 'rb') as f:
    model = pickle.load(f)
N = (5, 5)  # neighbourhood of pixel


def _detect(image):
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


def __rgb_to_hsv(r, g, b):
    maxc = max(r, g, b)
    minc = min(r, g, b)
    v = maxc
    if minc == maxc:
        return 0.0, 0.0, v
    s = (maxc-minc) / maxc
    rc = (maxc-r) / (maxc-minc)
    gc = (maxc-g) / (maxc-minc)
    bc = (maxc-b) / (maxc-minc)
    if r == maxc:
        h = bc-gc
    elif g == maxc:
        h = 2.0+rc-bc
    else:
        h = 4.0+gc-rc
    h = (h/6.0) % 1.0
    return h, s, v

def __rgb_to_ycbcr(r, g, b):
    y = np.trunc((0.257 * r) + (0.504 * g) + (0.098 * b) + 16)
    cb = np.trunc(((-0.148) * r) - (0.291 * g) + (0.439 * b) + 128)
    cr = np.trunc((0.439 * r) - (0.368 * g) - (0.071 * b) + 128)
    return y, cb, cr

def _detect2(image):
    img = np.array(image)
    image_mask = np.zeros(img.shape, dtype=np.uint8)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            
            r, g, b = img[i][j]
            y, cr, cb = __rgb_to_ycbcr(r, g, b)
            h, s, v = __rgb_to_hsv(r, g, b)
            cond0 = 0. <= h <= 50.0 and 0.23 <= s <= 0.68 and\
                    r > 95 and g > 40 and b > 20 and r > g and r > b and\
                    r - g > 15
            cond1 = r > 95 and g > 40 and b > 20 and r > g and r > b and\
                    r - g > 15  and cr > 135 and\
                    cb > 85 and y > 80 and\
                    cr <= 1.5862*cb+20 and\
                    cr >= 0.3448*cb+76.2069 and\
                    cr >= -4.5652*cb+234.5652 and\
                    cr <= -1.15*cb+301.75 and\
                    cr <= -2.2857*cb+432.85
            if cond0 or cond1:
                image_mask[i][j] = [255, 255, 255]
    
    image_mask = cv2.cvtColor(image_mask, cv2.COLOR_RGB2GRAY)
    
    # smoothing mask by applying erosion and dilatation
    image_mask = cv2.erode(image_mask, None, iterations = 3)  # remove noise
    image_mask = cv2.dilate(image_mask, None, iterations = 3)  # smoothing eroded mask
    
    return image_mask


def _detect3(image):
    image = np.array(image)
    
    img_arr = np.zeros((image.shape[0] + N[0]-1, image.shape[1] + N[1]-1, 3))
    off0, off1 = N[0]//2, N[1]//2
    img_arr[off0:-off0, off1:-off1] = image/255.
    gt_arr = np.zeros(image.shape[:2])
    
    features_list = list()
    for i in range(gt_arr.shape[0]):
        for j in range(gt_arr.shape[1]):
            features_list.append(img_arr[i:i+N[0], j:j+N[1]].reshape(-1))
    
    features = np.array(features_list)
    preds = model.predict(features)
    
    image_mask = preds.reshape(gt_arr.shape).astype(np.uint8)
    image_mask[image_mask==1] = 255
    
    # smoothing mask by applying erosion and dilatation
    image_mask = cv2.erode(image_mask, None, iterations = 3)  # remove noise
    image_mask = cv2.dilate(image_mask, None, iterations = 3)  # smoothing eroded mask
    
    return image_mask


def detect_skin(image, bc=None, return_imgarray=False):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    image_mask = _detect(image)
    if type(bc) != NoneType:
        bc_mask = _detect3(bc)
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
