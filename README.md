import cv2

# Load cascade shipped with OpenCV
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
    cv2.imshow("Haar face detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break
cap.release()
cv2.destroyAllWindows()

import cv2
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret: break
    h,w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)), 1.0, (300,300), (104.0,177.0,123.0))
    net.setInput(blob)
    detections = net.forward()
    for i in range(detections.shape[2]):
        conf = detections[0,0,i,2]
        if conf > 0.5:
            box = detections[0,0,i,3:7] * [w,h,w,h]
            (x1,y1,x2,y2) = box.astype("int")
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"{conf:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    cv2.imshow("DNN SSD", frame)
    if cv2.waitKey(1) & 0xFF == 27: break
cap.release()
cv2.destroyAllWindows()

import cv2
import mediapipe as mp

mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as detector:
    while True:
        ret, frame = cap.read()
        if not ret: break
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(img_rgb)
        if results.detections:
            for det in results.detections:
                mp_draw.draw_detection(frame, det)
        cv2.imshow("MediaPipe Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27: break
cap.release()
cv2.destroyAllWindows()
