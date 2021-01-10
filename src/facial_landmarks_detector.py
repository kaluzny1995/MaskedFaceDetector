import os
os.chdir(os.getcwd() + '/src')

import cv2
import numpy as np
from PIL import Image
import dlib
import imutils
from imutils import face_utils

COLORS = [(19, 199, 109), (79, 76, 240), (230, 159, 23), (168, 100, 168), (158, 163, 32),
          (163, 38, 32), (180, 42, 220), (19, 199, 109), (19, 199, 109)]

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('models/shape_predictor_81_face_landmarks.dat')

os.chdir(os.path.dirname(os.getcwd()))

def visualize_facial_landmarks(image, shape, landmarks_idxs, colors=None, alpha=0.75):
    # create two copies of the input image -- one for the
    # overlay and one for the final output image
    overlay = image.copy()
    output = image.copy()

    # if the colors list is None, initialize it with a unique
    # color for each facial landmark region
    if colors is None:
        colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
            (168, 100, 168), (158, 163, 32),
            (163, 38, 32), (180, 42, 220)]

    # loop over the facial landmark regions individually
    for (i, name) in enumerate(landmarks_idxs.keys()):
        # grab the (x, y)-coordinates associated with the
        # face landmark
        (j, k) = landmarks_idxs[name]
        pts = shape[j:k]

        # check if are supposed to draw the jawline
        if name in ['jaw', 'forehead']:
            # since the jawline and forehead are non-enclosed facial regions,
            # just draw lines between the (x, y)-coordinates
            for l in range(1, len(pts)):
                ptA = tuple(pts[l - 1])
                ptB = tuple(pts[l])
                cv2.line(overlay, ptA, ptB, colors[i], 2)

        # otherwise, compute the convex hull of the facial
        # landmark coordinates points and display it
        else:
            hull = cv2.convexHull(pts)
            cv2.drawContours(overlay, [hull], -1, colors[i], -1)

    # apply the transparent overlay
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    # return the output image
    return output

def detect_facial_landmarks(image, return_imgarray=False):
    # copy original image and convert another copy to grayscale
    output = np.array(image)
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    
    # determine the facial landmarks for the face image
    rect = dlib.rectangle(0, 0, *gray.shape[::-1])
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
    # change order of forehead points - [9, 7, 8, 0, 1, 2, 3, 12, 4, 5, 11, 6, 10]
    shape[68:81] = np.array([shape[i+68] for i in [9, 7, 8, 0, 1, 2, 3, 12, 4, 5, 11, 6, 10][::-1]])
    landmarks_idxs = dict(face_utils.FACIAL_LANDMARKS_IDXS)
    landmarks_idxs['forehead'] = (68, 81)
    
    # visualize all facial landmarks with a transparent overlay
    output = visualize_facial_landmarks(output, shape, landmarks_idxs, colors=COLORS)
    
    # loop over the face parts individually
    output_info = dict({})
    for (name, (i, j)) in landmarks_idxs.items():
        # loop over the subset of facial landmarks, drawing the specific face part
        pts = list([])
        for (x, y) in shape[i:j]:
            pts.append((x, y))
            cv2.circle(output, (x, y), 1, (0, 0, 255), -1)
        output_info[name] = pts
    
    if return_imgarray:
        return output_info, output
    
    output = Image.fromarray(output)
    return output_info, output

def get_face_between_jaw_and_forehead(image, jawline_pts, forehead_pts, left_eye_pts, right_eye_pts, draw_roi=False, return_imgarray=False):
    img = np.array(image)
    output = np.array(image)
    output_roi = np.array(image)
    
    # for face restriction area
    # expand width of jawline to the width edges of detection
    min_id = np.argmin(np.array(jawline_pts)[:, ::2])
    min_x = jawline_pts[min_id][0]
    max_id = np.argmax(np.array(jawline_pts)[:, ::2])
    max_x = jawline_pts[max_id][0]
    
    jawline_pts = list([(int(((jp[0] - min_x)*img.shape[1]) / (max_x - min_x)), jp[1]) for jp in jawline_pts])
    
    # expand width of forehead to horizontal positions of jawline end points (to match landmarks)
    jaw_width = jawline_pts[-1][0] - jawline_pts[0][0]
    # WARNING!: Min. pt. must be the last and Max. - the first due to the reversed order of forehead pts.
    min_x = forehead_pts[-1][0]
    max_x = forehead_pts[0][0]
    
    forehead_pts = list([(int(((fp[0] - min_x)*jaw_width) / (max_x - min_x)), fp[1]) for fp in forehead_pts])
    
    face_polygon_pts = list([*jawline_pts, *forehead_pts])
    
    # restrict face polygon to face detection rectangle
    face_polygon_pts = list([(fpp[0], fpp[1] if fpp[1] >= 0 else 0) for fpp in face_polygon_pts])  # top
    face_polygon_pts = list([(fpp[0], fpp[1] if fpp[1] <= img.shape[0] else img.shape[0]) for fpp in face_polygon_pts])  # bottom
    face_polygon_pts = list([(fpp[0] if fpp[0] >= 0 else 0, fpp[1]) for fpp in face_polygon_pts])  # left
    face_polygon_pts = list([(fpp[0] if fpp[0] <= img.shape[1] else img.shape[1], fpp[1]) for fpp in face_polygon_pts])  # right
    
    face_polygon = cv2.convexHull(np.array(face_polygon_pts, dtype=np.float32))
    
    # for area above eyes line
    # get left and right eye lowest (i.e. furthest from top) coordinate
    max_left_id = np.argmax(np.array(left_eye_pts)[:, ::2])
    left_pt = left_eye_pts[max_left_id]
    max_right_id = np.argmax(np.array(right_eye_pts)[:, ::2])
    right_pt = right_eye_pts[max_right_id]
    
    a = 0 if right_pt[1] == left_pt[1] else (right_pt[0] - left_pt[0])/(right_pt[1] - left_pt[1])
    b = left_pt[0] - a*left_pt[1]
    
    # prepare mask images
    restr_mask_img = np.zeros(img.shape[:2], dtype=np.uint8)
    ae_mask_img = np.zeros(img.shape[:2], dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            dist_coeff = cv2.pointPolygonTest(face_polygon, (j, i), True)
            if dist_coeff >= 0:
                restr_mask_img[i][j] = 255
            if (a!= 0 and i <= (j-b)/a) or (a==0 and i <= right_pt[0]):
                ae_mask_img[i][j] = 255
                
    # draw face ROI
    if draw_roi:
        for i in range(len(face_polygon_pts)):
            start = face_polygon_pts[i]
            end = face_polygon_pts[(i+1)%len(face_polygon_pts)]
            cv2.line(output_roi, start, end, (0,255,0), 2)
                
    output = cv2.bitwise_and(output, output, mask = restr_mask_img)
    ae_mask_img = ae_mask_img & restr_mask_img
    
    restr_mask_img = cv2.cvtColor(restr_mask_img, cv2.COLOR_GRAY2RGB)
    ae_mask_img = cv2.cvtColor(ae_mask_img, cv2.COLOR_GRAY2RGB)
    
    if return_imgarray:
        return face_polygon_pts, output_roi, restr_mask_img, ae_mask_img, output
    
    output_roi = Image.fromarray(output_roi)
    restr_mask_img = Image.fromarray(restr_mask_img)
    ae_mask_img = Image.fromarray(ae_mask_img)
    output = Image.fromarray(output)
    return face_polygon_pts, output_roi, restr_mask_img, ae_mask_img, output
