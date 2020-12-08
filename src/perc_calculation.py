import cv2 
import numpy as np
from PIL import Image

PERC_THR = 0.8

#  Felzenszwalb et al.
def non_max_suppression_slow(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list, add the index
        # value to the list of picked indexes, then initialize
        # the suppression list (i.e. indexes that will be deleted)
        # using the last index
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]

        # loop over all indexes in the indexes list
        for pos in range(0, last):
            # grab the current index
            j = idxs[pos]
            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])
            # compute the width and height of the bounding box
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)
            # compute the ratio of overlap between the computed
            # bounding box and the bounding box in the area list
            overlap = float(w * h) / area[j]
            # if there is sufficient overlap, suppress the
            # current bounding box
            if overlap > overlapThresh:
                suppress.append(pos)

        # delete all indexes from the index list that are in the
        # suppression list
        idxs = np.delete(idxs, suppress)

    # return only the bounding boxes that were picked
    return boxes[pick]


def masking_mask(final_mask, restr_mask):
    final_m = cv2.cvtColor(np.array(final_mask), cv2.COLOR_RGB2GRAY)
    restr_m = cv2.cvtColor(np.array(restr_mask), cv2.COLOR_RGB2GRAY)
    
    result = final_m | np.bitwise_not(restr_m)
    
    return cv2.cvtColor(np.bitwise_not(result), cv2.COLOR_GRAY2RGB)


def intersects(x1, y1, x2, y2, fx1, fy1, fx2, fy2):
    s_xran = list(range(x1, x2+1)) if x1<x2 else list(range(x2, x1+1))
    s_yran = list(range(y1, y2+1)) if y1<y2 else list(range(y2, y1+1))
    
    f_xran = list(range(fx1, fx2+1)) if fx1<fx2 else list(range(fx2, fx1+1))
    f_yran = list(range(fy1, fy2+1)) if fy1<fy2 else list(range(fy2, fy1+1))
    
    x_int = np.intersect1d(s_xran, f_xran)
    y_int = np.intersect1d(s_yran, f_yran)
    
    return len(x_int) > 0 and len(y_int) > 0


def in_ellipse(xmin, ymin, xmax, ymax, x, y):
    x0 = (xmax+xmin)/2
    y0 = (ymax+ymin)/2
    a = np.abs(x0-xmin)
    b = np.abs(y0-ymin)
    
    lbv = np.floor(-np.sqrt(b**2 - b**2*(x - x0)**2/a**2) + y0)  # lower bound value
    ubv = np.ceil(np.sqrt(b**2 - b**2*(x - x0)**2/a**2) + y0)  # upper bound value
    
    return xmin<=x<=xmax and lbv<=y<=ubv


def count_pixels(arr, xmin, ymin, xmax, ymax):
    cnt_px = dict({0:0, 255:0})
    part = arr[ymin:ymax, xmin:xmax]
    
    for i in range(len(part)):
        for j in range(len(part[i])):
            if in_ellipse(ymin, xmin, ymax, xmax, x=i+ymin, y=j+xmin):
                cnt_px[part[i][j]] += 1
    
    return cnt_px


def calculate_masking_percentage(image, skin_mask, restr_mask, haar_regions, perc_thr=PERC_THR, return_imgarray=False):
    output = np.array(image)
    skin_m = cv2.cvtColor(np.array(skin_mask), cv2.COLOR_RGB2GRAY)
    restr_m = cv2.cvtColor(np.array(restr_mask), cv2.COLOR_RGB2GRAY)
    
    # filter skim mask by haar cascade detected areas
    skin_m_red = np.array(skin_m)  # reduced skin mask
    for _, regions in haar_regions.items():
        for (xmin, ymin), (xmax, ymax) in regions:
            skin_part = skin_m[ymin:ymax, xmin:xmax]
            
            cnt_px = count_pixels(skin_m, xmin, ymin, xmax, ymax)
            if cnt_px[255] >= perc_thr*(cnt_px[255] + cnt_px[0]):
                # make ellipse area white
                for i in range(len(skin_part)):
                    for j in range(len(skin_part[i])):
                        if in_ellipse(ymin, xmin, ymax, xmax, x=i+ymin, y=j+xmin):
                            skin_m_red[i+ymin][j+xmin] = 255
    # erode and dilate reduced skin mask
    skin_m_red = cv2.erode(skin_m_red, None, iterations = 3)  # remove noise
    skin_m_red = cv2.dilate(skin_m_red, None, iterations = 3)  # smoothing eroded mask
    
    # combine skin and restriction mask (bitwise and)
    final_m = skin_m_red & restr_m
    # calculate percentage
    u_px_restr, cnt_px_restr = np.unique(restr_m, return_counts=True)
    u_px_final, cnt_px_final = np.unique(final_m, return_counts=True)
    perc = 1. - cnt_px_final[np.where(u_px_final == 255)[0]] / cnt_px_restr[np.where(u_px_restr == 255)[0]]
    
    output = cv2.bitwise_and(output, output, mask = final_m)
    
    skin_m_red = cv2.cvtColor(skin_m_red, cv2.COLOR_GRAY2RGB)
    final_m = cv2.cvtColor(final_m, cv2.COLOR_GRAY2RGB)
    
    if return_imgarray:
        return perc[0], skin_m_red, final_m, output
    
    skin_m_red = Image.fromarray(skin_m_red)
    final_m = Image.fromarray(final_m)
    output = Image.fromarray(output)
    return perc[0], skin_m_red, final_m, output
    
def draw_roi(image, jawline_pts, return_imgarray=False):
    output = np.array(image)
    
    top_left_corner = (jawline_pts[0][0], 0)
    top_right_corner = (jawline_pts[-1][0], 0)
    
    face_roi_pts = list([top_left_corner, *jawline_pts, top_right_corner])
    
    for i in range(len(face_roi_pts)):
        start = face_roi_pts[i]
        end = face_roi_pts[(i+1)%len(face_roi_pts)]
        cv2.line(output, start, end, (0,255,0), 2)
    
    if return_imgarray:
        return output
    
    output = Image.fromarray(output)
    return output
