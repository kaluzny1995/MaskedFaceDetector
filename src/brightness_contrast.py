import cv2 
import numpy as np
from PIL import Image


def apply_brightness_contrast(image, brightness = 255, contrast = 127, return_imgarray=False):


    def map(x, in_min, in_max, out_min, out_max):
        return int((x-in_min) * (out_max-out_min) / (in_max-in_min) + out_min)
    
    
    input_img = np.array(image)
    
    b = map(brightness, 0, 510, -255, 255)
    c = map(contrast, 0, 254, -127, 127)

    if b != 0:
        if b > 0:
            shadow = b
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + b
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if c != 0:
        f = float(131 * (c + 127)) / (127 * (131 - c))
        alpha_c = f
        gamma_c = 127*(1-f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)
    
    if return_imgarray:
        return buf

    output = Image.fromarray(buf)
    return output
