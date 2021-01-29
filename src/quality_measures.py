import numpy as np
from PIL import Image
import cv2

from sklearn.metrics import accuracy_score, precision_score, recall_score


def highlight_detections(image, gt_faces, pred_faces, pred_w_faces, highlight_int=False, return_array=False):
    img_arr = np.array(image)
    
    # highlight ground truth faces
    for gtf in gt_faces:
        color = (0, 0, 255)
        cv2.rectangle(img_arr, (int(gtf[0]), int(gtf[1])), (int(gtf[2]), int(gtf[3])), color, 2)
    
    # highlight predicted faces
    for pf in pred_faces:
        color = (255, 255, 0)
        cv2.rectangle(img_arr, (int(pf[0]), int(pf[1])), (int(pf[2]), int(pf[3])), color, 2)
    
    if highlight_int:
        # highlight predicted faces
        for gt, pr in zip(gt_faces, pred_faces):
            max_x1, max_y1 = np.max([gt[0], pr[0]]), np.max([gt[1], pr[1]])
            min_x2, min_y2 = np.min([gt[2], pr[2]]), np.min([gt[3], pr[3]])
            color = (0, 255, 0)
            
            cv2.rectangle(img_arr, (int(max_x1), int(max_y1)), (int(min_x2), int(min_y2)), color, 2)
    
    # highlight incorrectly predicted faces
    for pwf in pred_w_faces:
        color = (255, 0, 0)
        cv2.rectangle(img_arr, (int(pwf[0]), int(pwf[1])), (int(pwf[2]), int(pwf[3])), color, 2)
    
    if return_array:
        return img_arr
    
    img_out = Image.fromarray(img_arr)
    return img_out


def intersection_of_union(gt_faces, pred_faces):
    iou = list([])
    
    for gt, pr in zip(gt_faces, pred_faces):
        max_x1, max_y1 = np.max([gt[0], pr[0]]), np.max([gt[1], pr[1]])
        min_x2, min_y2 = np.min([gt[2], pr[2]]), np.min([gt[3], pr[3]])
        
        area_int = (min_x2 - max_x1)*(min_y2 - max_y1)
        area_gt = (gt[2] - gt[0])*(gt[3] - gt[1])
        area_pr = (pr[2] - pr[0])*(pr[3] - pr[1])
        
        iou.append(area_int/(area_gt + area_pr - area_int)*100)
    
    return np.array(iou)


def highlight_predictions(image, f_coords, f_gt, f_pr, put_numbers=False, return_array=False):
    img_arr = np.array(image)
    
    # TP (green) | TN (blue) | FP (red) | FN (yellow)
    # highlight faces with appropriate color
    for fc, gt, pr, i in zip(f_coords, f_gt, f_pr, range(len(f_pr))):
        if gt == 1 and pr == 1:
            color = (0, 255, 0)
        elif gt == 0 and pr == 0:
            color = (0, 0, 255)
        elif gt == 1 and pr == 0:
            color = (255, 0, 0)
        else:
            color = (255, 255, 0)
        
        cv2.rectangle(img_arr, (int(fc[0]), int(fc[1])), (int(fc[2]), int(fc[3])), color, 3)
        if put_numbers:
            cv2.putText(img_arr, f'#{i}', (int(fc[0]), int(fc[1])), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 3)
    
    if return_array:
        return img_arr
    
    img_out = Image.fromarray(img_arr)
    return img_out


def classification_metrics(true, pred):
    acc = accuracy_score(y_true=true, y_pred=pred)
    _, prec = precision_score(y_true=true, y_pred=pred, labels=[0, 1], average=None, zero_division=0)
    _, rec = recall_score(y_true=true, y_pred=pred, labels=[0, 1], average=None, zero_division=0)
    
    return acc, prec, rec


def convert_mask(mask_image, return_array=False):
    m_img_arr = np.array(mask_image)
    
    for i in range(m_img_arr.shape[0]):
        for j in range(m_img_arr.shape[1]):
            p = m_img_arr[i, j]
            if p[-1] == 0:  # white pixel
                m_img_arr[i, j] = np.array([255, 255, 255, 255], dtype=np.uint8)
            else:  # black pixel
                m_img_arr[i, j] = np.array([0, 0, 0, 255], dtype=np.uint8)
    m_img_arr = cv2.cvtColor(m_img_arr, cv2.COLOR_RGBA2RGB)
    
    if return_array:
        return m_img_arr
    
    image_out = Image.fromarray(m_img_arr)
    return image_out

def invert_mask(mask_image, return_array=False):
    m_img_arr = np.array(mask_image)
    
    m_img_arr = cv2.cvtColor(m_img_arr, cv2.COLOR_RGB2GRAY)
    m_img_arr = ~m_img_arr
    m_img_arr = cv2.cvtColor(m_img_arr, cv2.COLOR_GRAY2RGB)
    
    if return_array:
        return m_img_arr
    
    image_out = Image.fromarray(m_img_arr)
    return image_out


def subtract_masks(mask_image, mask_image_, return_array=False):
    m_img_arr = np.array(mask_image)
    m_img_arr_ = np.array(mask_image_)
    
    m_img_arr = cv2.cvtColor(m_img_arr, cv2.COLOR_RGB2GRAY)
    m_img_arr_ = cv2.cvtColor(m_img_arr_, cv2.COLOR_RGB2GRAY)
    m_img_arr = m_img_arr | ~m_img_arr_
    m_img_arr = cv2.cvtColor(m_img_arr, cv2.COLOR_GRAY2RGB)
    
    if return_array:
        return m_img_arr
    
    image_out = Image.fromarray(m_img_arr)
    return image_out


def mask_area_metrics(mask_gt_image, mask_pr_image):
    m_gt_arr = np.array(mask_gt_image)
    m_pr_arr = np.array(mask_pr_image)
    
    m_gt_arr = cv2.cvtColor(m_gt_arr, cv2.COLOR_RGB2GRAY)
    m_pr_arr = cv2.cvtColor(m_pr_arr, cv2.COLOR_RGB2GRAY)
    
    gt, pr = list([]), list([])
    for i in range(m_gt_arr.shape[0]):
        for j in range(m_gt_arr.shape[1]):
            gt.append(m_gt_arr[i, j])
            pr.append(m_pr_arr[i, j])
    
    gt, pr = np.array(gt), np.array(pr)
    
    acc = accuracy_score(y_true=gt, y_pred=pr)
    _, prec = precision_score(y_true=gt, y_pred=pr, labels=[0, 255], average=None, zero_division=0)
    _, rec = recall_score(y_true=gt, y_pred=pr, labels=[0, 255], average=None, zero_division=0)
    
    return acc, prec, rec
