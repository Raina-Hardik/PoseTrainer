import cv2
import numpy as np
from keras.models import load_model

model = load_model('pose_classification_model.h5')

num_frames = 5

video = cv2.VideoCapture('example.mp4')

prediction_buffer = []

while True:
    ret, frame = video.read()
    if not ret: break

    frame = cv2.resize(frame, (224, 224))
    frame = frame.astype('float32') / 255.0
    frame = np.expand_dims(frame, axis=0)

    prediction = model.predict(frame)

    prediction_buffer.append(prediction)

    if len(prediction_buffer) > num_frames: prediction_buffer.pop(0)
    average_prediction = np.mean(prediction_buffer, axis=0)

    class_idx = np.argmax(average_prediction)
    print(class_idx)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video file and close the window
video.release()
cv2.destroyAllWindows()