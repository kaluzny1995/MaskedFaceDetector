import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from src.face_detector import inference
from src.human_silhouette_detector import detect_human_silhouettes_and_faces
from src.haar_detector import detect_face_parts
from src.brightness_contrast import clahe
from src.facial_landmarks_detector import detect_facial_landmarks, get_face_between_jaw_and_forehead
from src.human_skin_detector import detect_skin
from src.perc_calculation import calculate_masking_percentage, draw_roi, masking_mask


BRIGHNESS = 180
CONTRAST = 180
PERC_THR = 0.5
TRANSPARENCY = 0.25

def example(image, highlight_masking=False, return_imgarray=False):
    # detect faces
    faces, _, _ = detect_human_silhouettes_and_faces(image, return_imgarray=True)
    
    # for each detected face get their masking percentages with rois
    rois, m_masks, percs = list([]), list([]), list([])
    for face in faces:
        # crop face from input image
        face_arr = np.array(image.crop(tuple(face)))
        # detect face parts (Haar Cascades)
        face_parts, _ = detect_face_parts(face_arr, return_imgarray=True)
        # detect facial landmarks (HOG)
        face_landmarks, _ = detect_facial_landmarks(face_arr, return_imgarray=True)
        # restrict face area to area above the face jawline, get roi
        face_jawline_pts, _, face_restr_mask, face_ae_mask, _ = get_face_between_jaw_and_forehead(face_arr, face_landmarks['jaw'], face_landmarks['forehead'], face_landmarks['left_eye'], face_landmarks['right_eye'], return_imgarray=True)
        rois.append(face_jawline_pts)
        # decrease face brightness and increase contrast
        face_arr_clahe = clahe(face_arr, return_imgarray=True)
        # detect skin on face
        _, face_skin_mask, _ = detect_skin(face_arr, face_arr_clahe, return_imgarray=True)
        # calculate masking percentage
        face_perc, _, final_mask, _ = calculate_masking_percentage(face_arr, face_skin_mask, face_restr_mask, face_ae_mask, face_parts, perc_thr=PERC_THR)
        percs.append(face_perc)
        # get masking mask
        m_mask = masking_mask(final_mask, face_restr_mask)
        m_masks.append(m_mask)
    
    
    # draw the final result
    img = np.array(image)
    
    for face, roi, m_mask, perc in zip(faces, rois, m_masks, percs):
        # highlight face masking mask (blue) if set
        if highlight_masking:
            color = (0, 0, 255)
            face_arr = img[face[1]:face[3], face[0]:face[2]]
            color_layer = np.array(np.ones(face_arr.shape)*color).astype(np.uint8)
            transp = cv2.addWeighted(face_arr, TRANSPARENCY, color_layer, 1-TRANSPARENCY, 0.)
            frag1 = cv2.bitwise_and(transp, m_mask)
            frag2 = cv2.bitwise_and(face_arr, cv2.bitwise_not(m_mask))
            final = cv2.bitwise_or(frag1, frag2)
            
            img[face[1]:face[3], face[0]:face[2]] = final
        
        # face rectangle (green)
        color = (0, 255, 0)
        cv2.rectangle(img, tuple(face[:2]), tuple(face[2:]), color, 6)
        # RoI polygon (yellow)
        color = (255, 255, 0)
        for i in range(len(roi)):
            start = roi[i]
            start = (start[0] + face[0], start[1] + face[1])
            end = roi[(i+1)%len(roi)]
            end = (end[0] + face[0], end[1] + face[1])
            cv2.line(img, start, end, color, 4)
        # percentage 0.0001 precision (red)
        color = (255, 0, 0)
        cv2.putText(img, f'{np.round(perc*100, 4)}%', (face[0] + 2, face[1] - 2),
                        cv2.FONT_HERSHEY_PLAIN, 3, color, 5)
    
    if return_imgarray:
        return faces, rois, percs, img
    
    output_img = Image.fromarray(img)
    return faces, rois, percs, output_img
