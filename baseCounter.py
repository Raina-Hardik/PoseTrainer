import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
counter = 0
stage = None
create = None

def findPosition(image, draw=True):
  lmList = []
  if results.pose_landmarks:
      mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
      for id, lm in enumerate(results.pose_landmarks.landmark):
          h, w, _ = image.shape
          cx, cy = int(lm.x * w), int(lm.y * h)
          lmList.append([id, cx, cy])
  
  return lmList

cap = cv2.VideoCapture(0)

with mp_pose.Pose(

    min_detection_confidence=0.7,

    min_tracking_confidence=0.7) as pose:

  while cap.isOpened():

    success, image = cap.read()
    image = cv2.resize(image, (640,480), interpolation=cv2.INTER_CUBIC)
    
    if not success: continue
    
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    lmList = findPosition(image, draw=True)

    if len(lmList) != 0:
      cv2.circle(image, (lmList[12][1], lmList[12][2]), 20, (0, 0, 255), cv2.FILLED)
      cv2.circle(image, (lmList[11][1], lmList[11][2]), 20, (0, 0, 255), cv2.FILLED)
      cv2.circle(image, (lmList[12][1], lmList[12][2]), 20, (0, 0, 255), cv2.FILLED)
      cv2.circle(image, (lmList[11][1], lmList[11][2]), 20, (0, 0, 255), cv2.FILLED)
      
      if (lmList[12][2] and lmList[11][2] >= lmList[14][2] and lmList[13][2]):
        cv2.circle(image, (lmList[12][1], lmList[12][2]), 20, (0, 255, 0), cv2.FILLED)
        cv2.circle(image, (lmList[11][1], lmList[11][2]), 20, (0, 255, 0), cv2.FILLED)
        stage = "down"

      if (lmList[12][2] and lmList[11][2] <= lmList[14][2] and lmList[13][2]) and stage == "down":
        stage = "up"
        counter += 1
        print(counter)

    text = "COUNT:{}".format(counter)

    cv2.putText(image, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('MediaPipe Pose', image)

    if create is None:
      fourcc = cv2.VideoWriter_fourcc(*'XVID')
      create = cv2.VideoWriter('PushUps', fourcc, 30, (image.shape[1], image.shape[0]), True)
    
    create.write(image)
    if cv2.waitKey(5) & 0xFF == 27: break

cv2.destroyAllWindows()