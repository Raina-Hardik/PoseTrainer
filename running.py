import sys
import cv2
import numpy as np
from keras.models import load_model

model = load_model('pose_classification_model.h5')

# Uses camera by default if no video path provided
if len(sys.argv) == 1: 
    video = cv2.VideoCapture(0)
else:
    video = cv2.VideoCapture(sys.argv[1])

while True: 
    ret, frame = video.read()

    if not ret: break

    # Preprocessing
    frame = cv2.resize(frame, (224, 224))
    frame = frame.astype('float32') / 255.0
    frame = np.expand_dims(frame, axis=0)

    prediction = model.predict(frame)

    class_idx = np.argmax(prediction)

    # Going to change print to speaking in next version
    print(class_idx)                   

    if cv2.waitKey(1) & 0xFF == ord('q'): break

video.release()
cv2.destroyAllWindows()
