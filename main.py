import cv2
import os
import numpy as np
import datetime
import csv
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
dataset_path = "dataset"
faces = []
labels = []
names = {}
label_id = 0

for file in os.listdir(dataset_path):

    path = os.path.join(dataset_path, file)

    img = cv2.imread(path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detected_faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in detected_faces:

        face = gray[y:y+h, x:x+w]

        face = cv2.resize(face, (200, 200))

        faces.append(face)

        labels.append(label_id)

        name = os.path.splitext(file)[0]

        names[label_id] = name

        label_id += 1

recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.train(faces, np.array(labels))

cap = cv2.VideoCapture(0)

marked = set()

while True:

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detected_faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in detected_faces:

        face = gray[y:y+h, x:x+w]

        face = cv2.resize(face, (200, 200))

        label, confidence = recognizer.predict(face)

        if confidence < 80:
            name = names[label]
        else:
            name = "Unknown"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, name, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,255,0), 2)
        if name != "Unknown" and name not in marked:

            time_now = datetime.datetime.now().strftime("%H:%M:%S")

            with open("attendance.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([name, time_now])

            print("Attendance Saved:", name)

            marked.add(name)

    cv2.imshow("Face Recognition Attendance", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()