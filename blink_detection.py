
import cv2
import  scipy.spatial
from scipy.spatial import distance
import numpy as np
import dlib
import imutils
from imutils import face_utils





Eye_AR_Thresh =  0.3
Eye_AR_Consec_frames = 3 
counter = 0
total = 0
font = cv2.FONT_HERSHEY_COMPLEX

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A+B) / (2*C)
    return ear

detector =  dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()
    if frame is None:
        break
    frame = imutils.resize(frame, width =500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    
    for face in rects:
        (x1, y1) = (face.left(), face.top())
        (x2, y2) = (face.right(), face.bottom())
        
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)
        
        
        leftEye =  shape [36:42]
        rightEye = shape[42:48]
        
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        EAR = (leftEAR + rightEAR) /2
        
        if EAR < Eye_AR_Thresh:
            counter +=1
        else:
            if counter > Eye_AR_Consec_frames:
                total +=1
            counter = 0
        
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 2)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 2)
        
        cv2.putText(frame, "Blinks: {}".format(total), (10, 20), font, 0.55, (0, 0, 255), 1)
        #cv2.putText(frame, "EAR: {:.2f}".format(EAR), (10, 50), font, 0.55, (0, 0, 255), 1)
        
        
        
    
    cv2.imshow("frame", frame)
    if cv2.waitKey(30) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
    
 


