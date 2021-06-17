import cv2
import pickle

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
profile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_profileface.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
try:
    recognizer.read("classifier.yml")
    fresh = False
except cv2.error as e:
    fresh = True
with open("labels.picle", 'rb') as f:
    loaded_labels = pickle.load(f)
    labels = {v:k for k,v in loaded_labels.items()}

cam = cv2.VideoCapture(0)
#i = 0
while True:

    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)  # play with numbers for better results

    for (x,y,w,h) in faces:
        #print(f'Face{i}: x={x} y={y} w={w} h={h}')
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        person = name = "Unknown"
        if not fresh:
            _id, conf = recognizer.predict(roi_gray)

            if conf <= 50:
                name = f"{str(labels[_id]).capitalize()} conf:{format(conf, '.2f')}"
                person = name.split(" ")[0]
            #i += 1
        #face = f"{person}{i}.png"
        #cv2.imwrite(face, roi_color)

        if(name is "Unknown"):
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        stroke = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        width = x + w
        height = y + h
        fontScale = width / 900
        print(fontScale)
        cv2.rectangle(frame, (x, y), (width, height), color, int(stroke*2))
        cv2.putText(frame, name, (x, y-5), font, fontScale, color, stroke, cv2.LINE_AA)

    cv2.imshow('NOT Skynet', frame)


cv2.waitKey(0)
cv2.destroyAllWindows()