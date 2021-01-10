import cv2 
import numpy as np
from PIL import Image


def change_brightness_contrast(image, brightness = 255, contrast = 127, return_imgarray=False):


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

def clahe(image, return_imgarray=False):
    input_img = np.array(image)

    lab = cv2.cvtColor(input_img, cv2.COLOR_RGB2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)

    output_img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    if return_imgarray:
        return output_img

    output = Image.fromarray(output_img)
    return output

def equalize_histogram(image, return_imgarray=False):
    input_img = np.array(image)
    
    img_y_cr_cb = cv2.cvtColor(input_img, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(img_y_cr_cb)

    # Applying equalize Hist operation on Y channel.
    y_eq = cv2.equalizeHist(y)

    img_y_cr_cb_eq = cv2.merge((y_eq, cr, cb))
    output_img = cv2.cvtColor(img_y_cr_cb_eq, cv2.COLOR_YCR_CB2RGB)
    
    if return_imgarray:
        return output_img

    output = Image.fromarray(output_img)
    return output

def gaussian_blur(image, return_imgarray=False):
    input_img = np.array(image)
    hh, ww = input_img.shape[:2]
    mx = max(hh, ww)

    # illumination normalize
    ycrcb = cv2.cvtColor(input_img, cv2.COLOR_BGR2YCrCb)

    # separate channels
    y, cr, cb = cv2.split(ycrcb)

    # get background which paper says (gaussian blur using standard deviation 5 pixel for 300x300 size image)
    # account for size of input vs 300
    sigma = 5 * mx / 300
    gaussian = cv2.GaussianBlur(y, (0, 0), sigma, sigma)

    # subtract background from Y channel
    y = (y - gaussian + 100)

    # merge channels back
    ycrcb = cv2.merge([y, cr, cb])

    #convert to BGR
    output_img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    
    if return_imgarray:
        return output_img

    output = Image.fromarray(output_img)
    return output

# Automatic brightness and contrast optimization with optional histogram clipping
def autochange_brightness_contrast(image, clip_hist_percent=25, return_imgarray=False):
    
    
    def convert_scale(img, alpha, beta):
        """Add bias and gain to an image with saturation arithmetics. Unlike
        cv2.convertScaleAbs, it does not take an absolute value, which would lead to
        nonsensical results (e.g., a pixel at 44 with alpha = 3 and beta = -210
        becomes 78 with OpenCV, when in fact it should become 0).
        """

        new_img = img * alpha + beta
        new_img[new_img < 0] = 0
        new_img[new_img > 255] = 255
        return new_img.astype(np.uint8)
    
    
    input_img = np.array(image)
    gray = cv2.cvtColor(input_img, cv2.COLOR_RGB2GRAY)

    # Calculate grayscale histogram
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    output_img = convert_scale(input_img, alpha=alpha, beta=beta)
    
    if return_imgarray:
        return output_img, alpha, beta

    output = Image.fromarray(output_img)
    return output, alpha, beta
