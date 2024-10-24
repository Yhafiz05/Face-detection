from __future__ import print_function

from collections import deque

import cv2 as cv
import numpy as np
def alpha_mask(frame, img_with_alpha, orig=None, threshold=0):
    """Insert an image with alpha channel into another image. """
    # frame: the image to insert into
    # img_with_alpha: the image to be inserted
    # orig: the position to insert the image
    # threshold: the threshold to apply the mask
    # return: the modified image

    # get the size of the frame
    w, h, _ = frame.shape

    # if orig is None, set it to the top left of the frame
    if orig is None:
        orig = [0, 0]
    # if the position is out of the frame, return the frame
    if not (0 <= orig[0] < w and 0 <= orig[1] < h):
        return frame

    # get the start position to insert the image
    x_start, y_start = orig
    # if the position is out of the frame, set it to the top left of the frame
    x_start, y_start = max(x_start, 0), max(y_start, 0)

    # idem for the end position
    x_end, y_end = x_start + img_with_alpha.shape[0], y_start + img_with_alpha.shape[1]
    x_end, y_end = min(x_end, w), min(y_end, h)

    # create the mask
    mask = np.zeros((w, h, 4), dtype=np.uint8)
    # insert the image into the mask
    mask[x_start:x_end, y_start:y_end, :] = img_with_alpha[:x_end - x_start, :y_end - y_start, :]

    # apply the mask to the frame
    np.copyto(frame, mask[:, :, :3], where=(mask[:, :, 3] > threshold)[:, :, None])

    return frame

mustache_image = cv.imread('mustache.png', cv.IMREAD_UNCHANGED)

pts = deque(maxlen=12)
def detectAndDisplay(frame):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    #-- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    if len(faces) > 0:
        for (x,y,w,h) in faces:
            #Tracé du rectangle sur tous les visages détectés
            frame = cv.rectangle(frame,(x,y), (x+w,y+h), (255,0,255),2)
            faceROI = frame_gray[y:y+h,x:x+w]

            #-- In each face, detect eyes
            eyes = eyes_cascade.detectMultiScale(faceROI)
            for (x2,y2,w2,h2) in eyes:
                #Tracé du rectangle sur les yeux détectés
                frame = cv.rectangle(frame, (x + x2, y + y2), (x + x2 + w2, y + y2 + h2), (255, 0, 0), 2)

            mouths,rejectLevel,levelWeight= mouth_cascade.detectMultiScale3(faceROI,outputRejectLevels=True)
            for i in range(len(levelWeight)):
                if(levelWeight[i]>2.5):
                    (x3,y3,w3,h3) = mouths[i]
                    #Tracé du rectangle autour de la bouche
                    frame = cv.rectangle(frame, (x + x3, y + y3), (x + x3 + w3, y + y3 + h3), (0, 255, 0), 2)
                    mouthsROI = frame_gray[y+y3:y+y3+h3,x+x3:x3+x+w3]

                    #Detection de sourrire
                    smile, rejectLevel_s, levelWeight_s = smile_cascade.detectMultiScale3(mouthsROI,outputRejectLevels=True)
                    for j in range(len(levelWeight_s)):
                        if(levelWeight_s[j]>2):
                            for (x6, y6, w6, h6) in smile:
                                frame = cv.putText(frame, "Smile detected", (y, x), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            #Detection du nez
            noses,rejectLevel_n,levelWeight_n = nose_cascade.detectMultiScale3(faceROI,outputRejectLevels=True)
            for i in range(len(levelWeight_n)):
                if(levelWeight_n[i]>1.8):

                    (x4,y4,w4,h4) = noses[i]
                    #Tracé du rectangle autour du nez
                    frame = cv.rectangle(frame, (x + x4, y + y4), (x + x4 + w4, y + y4 + h4), (0, 0, 255), 2)

                    mustache_height, mustache_width, _ = mustache_image.shape
                     # Coordonnées de la région du nez
                    x_nose, y_nose, w_nose, h_nose = x + x4, (y + y4)+h4//2, w4, h4
                    # Redimensionner la moustache pour l'adapter à la région du nez
                    mustache_resized = cv.resize(mustache_image, (w_nose, mustache_height * w_nose // mustache_width))
                    x5 = int((x+x4)+w4)-w_nose
                    y5 = int((y+y4)+h4)-h_nose//2
                    frame = alpha_mask(frame,mustache_resized,(y5,x5))

    fist_pts = find_feature(frame, fist_cascade, (0, 255, 255))
    if (len(fist_pts) != 0) : pts.append((fist_pts[0] + fist_pts[2]//2, fist_pts[1] + fist_pts[3]//2))

    find_feature(frame, left_palm_cascade, (0, 255, 255))
    fist_histo()

def find_feature(frame, cascade_file, color):
    feature, rejectLevel, levelWeight = cascade_file.detectMultiScale3(frame, outputRejectLevels=True)
    if(len(feature) <= 0): return []
    max = 0
    for i in range(len(levelWeight)):
        if (levelWeight[i] > 0):
            if levelWeight[i] >= max: max = i
            x, y, w, h = feature[i]
            cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    return feature[max]

def fist_histo():
    for i in range(len(pts)-1):
        thickness = int(np.sqrt(pts.maxlen / (len(pts) - i + 1)) * 2.5)
        cv.line(frame, pts[i], pts[i+1], (0, 255, 255), thickness)



    cv.imshow('Capture - Face detection', frame)
#Load des fichiers

face_cascade = cv.CascadeClassifier('./haarcascade_frontalface_default.xml')
eyes_cascade = cv.CascadeClassifier('./haarcascade_eye_tree_eyeglasses.xml')
mouth_cascade = cv.CascadeClassifier('./haarcascade_mcs_mouth.xml')
nose_cascade = cv.CascadeClassifier('./haarcascade_mcs_nose.xml')
smile_cascade = cv.CascadeClassifier('./haarcascade_smile.xml')
left_palm_cascade = cv.CascadeClassifier('./lpalm.xml')
fist_cascade = cv.CascadeClassifier('./fist.xml')

#-- 2. Read the video stream
cap = cv.VideoCapture(0)
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)
while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break
    detectAndDisplay(frame)
    if cv.waitKey(10) == 27:
        break

