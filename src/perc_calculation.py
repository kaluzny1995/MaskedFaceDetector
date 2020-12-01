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


def calculate_masking_percentage(image, skin_mask, restr_mask, haar_regions):
    output = np.array(image)
    skin_m = cv2.cvtColor(np.array(skin_mask), cv2.COLOR_RGB2GRAY)
    restr_m = cv2.cvtColor(np.array(restr_mask), cv2.COLOR_RGB2GRAY)
    
    # filter skim mask by haar cascade detected areas
    for _, regions in haar_regions.items():
        for (xmin, ymin), (xmax, ymax) in regions:
            skin_part = skin_m[ymin:ymax, xmin:xmax]
            u_px, cnt_px = np.unique(skin_part, return_counts=True)
            if 255 in u_px:
                id_255 = np.where(u_px == 255)[0]
                s_h, s_w = skin_part.shape
                if cnt_px[id_255] >= PERC_THR*s_h*s_w:
                    # make certain area white
                    skin_m[ymin:ymax, xmin:xmax] = np.full(skin_part.shape, 255)
            
    # combine skin and restriction mask (bitwise and)
    final_m = skin_m & restr_m
    # calculate percentage
    u_px_restr, cnt_px_restr = np.unique(restr_m, return_counts=True)
    u_px_final, cnt_px_final = np.unique(final_m, return_counts=True)
    perc = 1. - cnt_px_final[np.where(u_px_final == 255)[0]] / cnt_px_restr[np.where(u_px_restr == 255)[0]]
    
    output = cv2.bitwise_and(output, output, mask = final_m)
    
    final_m = cv2.cvtColor(final_m, cv2.COLOR_GRAY2RGB)
    
    final_m = Image.fromarray(final_m)
    output = Image.fromarray(output)
    return perc[0], final_m, output
    
def draw_roi(image, jawline_pts):
    output = np.array(image)
    
    jawline_pts = sorted(jawline_pts, key=lambda k: k[0])
    top_left_corner = (jawline_pts[0][0], 0)
    top_right_corner = (jawline_pts[-1][0], 0)
    
    face_roi_pts = list([top_left_corner, *jawline_pts, top_right_corner])
    
    for i in range(len(face_roi_pts)):
        start = face_roi_pts[i]
        end = face_roi_pts[(i+1)%len(face_roi_pts)]
        cv2.line(output, start, end, (0,255,0), 2)
    
    output = Image.fromarray(output)
    
    return output
