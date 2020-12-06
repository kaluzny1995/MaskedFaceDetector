import cv2
import torch
import numpy as np
from PIL import Image

from sklearn.tree import DecisionTreeClassifier
import pickle

from detecto.core import Model
from detecto.utils import reverse_normalize, normalize_transform, _is_iterable
from detecto import utils, visualize

from src.face_detector import inference
from src.perc_calculation import non_max_suppression_slow, intersects

SCORE_THR = 0.7
OVERLAP_THR = 0.3
model = Model(device='cpu')


def detect_human_silhouettes(image, return_imgarray=False):
    img = np.copy(image)
    try:
        labels, boxes, scores = model.predict(img)
    except:
        return [], image
    
    persons = [(label, box, score) for label, box, score in zip(labels, boxes, scores) if label == 'person' and score >= SCORE_THR]
    if persons:
        persons_object = zip(*persons)
        [labels, boxes, scores] = list(persons_object)
        boxes = np.stack(boxes)

        for i in range(boxes.shape[0]):
            box = boxes[i]
            first_pos = (int(box[0].item()), int(box[1].item()))
            second_pos = (int(box[2].item()), int(box[3].item()))
            color = (0, 0, 255)
            cv2.rectangle(img, first_pos, second_pos, color, 3)
        
        if return_imgarray:
            return boxes, img
        
        img_output = Image.fromarray(img)
        return boxes, img_output
    else:
        if return_imgarray:
            return [], img
        
        return [], image


def detect_human_silhouettes_and_faces(image, highlight_neg=False, return_imgarray=False):
    with open('src/models/mm_sc.pkl', 'rb') as f:
        sc_x, sc_y = tuple(pickle.load(f))
    with open('src/models/dt_pos.pkl', 'rb') as f:
        clf = pickle.load(f)
    
    silhs, img = detect_human_silhouettes(image)
    img_arr = np.array(img)
    faces, _ = inference(np.array(image))
    
    positive_faces, negative_faces = list([]), list([])
    #silhs = np.concatenate([silhs, [[0, 0, image.width, image.height]]])
    for face in faces:
        fx1, fy1, fx2, fy2 = tuple(face[2:])
        
        votes = list([])
        for silh in silhs:
            x1, y1, x2, y2 = tuple(np.array(silh, dtype=np.int32))
            
            # if exists intersection between silhouette and face rectangles
            if intersects(x1, y1, x2, y2, fx1, fy1, fx2, fy2):
                s_x = np.array(sc_x.transform([[x1, x2]])).flatten()
                f_x = np.array(sc_x.transform([[fx1, fx2]])).flatten()
                s_y = np.array(sc_y.transform([[y1, y2]])).flatten()
                f_y = np.array(sc_y.transform([[fy1, fy2]])).flatten()
                s = np.array([[s_x[0], s_y[0], s_x[1], s_y[1], f_x[0], f_y[0], f_x[1], f_y[1]]])
                p = clf.predict(s)
            
                votes.append(p.flatten()[0])
        
        if np.any(votes):
            positive_faces.append([fx1, fy1, fx2, fy2])
        else:
            negative_faces.append([fx1, fy1, fx2, fy2])

    positive_faces = np.array(positive_faces)
    negative_faces = np.array(negative_faces)
    #positive_faces = non_max_suppression_slow(positive_faces, overlapThresh=OVERLAP_THR)
    #negative_faces = non_max_suppression_slow(negative_faces, overlapThresh=OVERLAP_THR)

    for p in positive_faces:
        color = (0, 255, 0)
        cv2.rectangle(img_arr, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), color, 3)
    if highlight_neg:
        for n in negative_faces:
            color = (255, 0, 0)
            cv2.rectangle(img_arr, (int(n[0]), int(n[1])), (int(n[2]), int(n[3])), color, 3)

    if return_imgarray:
        return positive_faces, negative_faces, img_arr
    
    img_output = Image.fromarray(img_arr)
    return positive_faces, negative_faces, img_output
