import cv2
import os
import time 
from datetime import datetime
from deepface import DeepFace

# LOADING HAAR CASCADE CLASSIFIERS

face          =     cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
right_eye     =     cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')
left_eye      =     cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')
smile         =     cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Starting Video Capture from camera

cap  =  cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    raise SystemExit("cannot open webcam")

video     =     None
prev      =     time.time()
font      =     cv2.FONT_HERSHEY_SIMPLEX

while True:
    ok, frame = cap.read()
    if not ok: break

    # Convert to grayscale

    gray           =     cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces          =     face.detectMultiScale          (gray, 1.1, 5)
    right_eyes     =     right_eye.detectMultiScale     (gray, 1.1, 5)
    left_eyes      =     left_eye.detectMultiScale      (gray, 1.1, 5)
    smiles         =     smile.detectMultiScale         (gray, 1.1, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h),(0, 255, 0), 1)
        cv2.putText(frame, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        roi_gray       =     gray[y:y + h, x:x + w]
        roi_color      =     frame[y:y + h, x:x + w]
        right_eyes     =     right_eye.detectMultiScale(roi_gray, 1.1, 5)
        for (rex, rey, rew, reh) in right_eyes:
            cv2.rectangle(roi_color, (rex, rey), (rex + rew, rey + reh), (0, 255, 0), 1)
            cv2.putText(roi_color, "Right Eye", (rex, rey - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        left_eyes      =     left_eye.detectMultiScale(roi_gray, 1.1, 5)
        for (lex, ley, lew, leh) in left_eyes:
            cv2.rectangle(roi_color, (lex, ley), (lex + lew, ley + leh), (0, 255, 0), 1)
            cv2.putText(roi_color, "Left Eye", (lex, ley - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        smiles         =     smile.detectMultiScale(roi_gray, 1.5, 5)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 1)
            cv2.putText(roi_color, "Smile", (sx, sy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        counts = {"faces": len(faces), "right_eyes": len(right_eyes), "left_eyes": len(left_eyes), "smiles": len(smiles)}
        print(f"Counts: {counts}")
        sidebar_text = f"Faces: {len(faces)} | Right Eyes: {len(right_eyes)} | Left Eyes: {len(left_eyes)} | Smiles: {len(smiles)}"
        cv2.putText(frame, sidebar_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(frame, "Date & Time: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"), (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        try:

            # Perform emotion analysis using DeepFace on the detected face region

            face_img     =     cv2.resize(roi_color, (224, 224))
            emotion      =     DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
            cv2.putText(frame, f"Emotion: {emotion[0]['dominant_emotion']}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        except Exception as e:
            print(f"Error occurred while analyzing emotion: {e}")

    now      =     time.time()
    fps      =     100  / (now - prev) if now != prev else 0
    prev     =     now
    cv2.putText(frame, f"Faces: {len(faces)} FPS: {int(fps)}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if video is not None:
        video.write(frame)

    cv2.imshow("Face Detection", frame)
    k = cv2.waitKey(100) & 0xFF
    if k != 255:
        print("Key Pressed:", k)
    if k == ord('q'):
        break

    elif k == ord('s'):
        filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(filename, frame)
        print(f"Screenshot saved: {filename}")

    elif k == ord('r'):
        if video is None:
            filename     =     f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
            fourcc       =     cv2.VideoWriter_fourcc(*'XVID')
            video        =     cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))
            print(f"Recording started: {filename}")
        else:
            video.release()
            video = None
            print("Recording stopped.")

cap.release()
if video is not None:
    video.release()
cv2.destroyAllWindows()
