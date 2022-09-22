import cv2
import mediapipe as mp
import numpy as np

def rescale(frame, scale = 1):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

def adjust(img, scale = 1.0):
    img = cv2.resize(img, (750, 750), interpolation=cv2.INTER_CUBIC)
    return rescale(img, scale)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success: continue
        image.flags.writeable = False
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        img_h, img_w, _ = image.shape
        #coord = (results.pose_landmarks.landmark[16].x*img_w, results.pose_landmarks.landmark[16].y*img_h)
        # print(len(results.pose_landmarks.landmark))

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        blank = np.zeros(image.shape[:2], dtype='uint8')
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        mp_drawing.draw_landmarks(blank, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        
        cv2.imshow('Original', (cv2.flip(image, 1)))
        cv2.imshow('KeyPoints', (cv2.flip(blank, 1)))
        if cv2.waitKey(5) & 0xFF == 27: break
cap.release()