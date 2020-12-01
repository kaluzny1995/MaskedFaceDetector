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
from src.perc_calculation import non_max_suppression_slow

SCORE_THR = 0.7
OVERLAP_THR = 0.3
model = Model(device='cpu')

with open('src/models/mm_sc.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('src/models/dt_pos.pkl', 'rb') as f:
    clf = pickle.load(f)


def detect_human_silhouettes(image):
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

        img_output = Image.fromarray(img)
        
        return boxes, img_output
    else:     
        return [], image


def detect_human_silhouettes_and_faces(image, highlight_neg=False):
    silhs, img = detect_human_silhouettes(image)
    
    img_arr = np.array(img)
    positive_faces, negative_faces = list([]), list([])
    for silh in silhs:
        x1, y1, x2, y2 = tuple(np.array(silh, dtype=np.int32))
        
        silh_img_arr = img_arr[y1:y2, x1:x2]
        faces, _ = inference(silh_img_arr)
        
        for face in faces:
            fx1, fy1, fx2, fy2 = tuple(face[2:])
            s = scaler.transform([[x1, y1, x2, y2, fx1, fy1, fx2, fy2]])
            p = clf.predict(s)
            
            if p[0]:
                positive_faces.append([fx1+x1, fy1+y1, fx2+x1, fy2+y1])
            else:
                negative_faces.append([fx1+x1, fy1+y1, fx2+x1, fy2+y1])

    positive_faces = np.array(positive_faces)
    negative_faces = np.array(negative_faces)
    positive_faces = non_max_suppression_slow(positive_faces, overlapThresh=OVERLAP_THR)
    negative_faces = non_max_suppression_slow(negative_faces, overlapThresh=OVERLAP_THR)

    for p in positive_faces:
        color = (0, 255, 0)
        cv2.rectangle(img_arr, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), color, 3)
    if highlight_neg:
        for n in negative_faces:
            color = (255, 0, 0)
            cv2.rectangle(img_arr, (int(n[0]), int(n[1])), (int(n[2]), int(n[3])), color, 3)

    img_output = Image.fromarray(img_arr)
    
    return positive_faces, negative_faces, img_output


'''
def detect_human_silhouettes_and_faces(image, highlight_neg=False):
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
            
            # if face is in silhouette region
            if x1>=fx1 and fx2<=x2 and y1>=fy1 and fy2<=y2:
                s = scaler.transform([[x1, y1, x2, y2, fx1, fy1, fx2, fy2]])
                p = clf.predict(s)
            
                votes.append(1 if p[0] else 0)

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

    img_output = Image.fromarray(img_arr)
    
    return positive_faces, negative_faces, img_output
'''
