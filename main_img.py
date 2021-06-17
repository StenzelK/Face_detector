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

with open("labels.pickle", 'rb') as f:
    loaded_labels = pickle.load(f)
    labels = {v:k for k,v in loaded_labels.items()}



img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3) #play with numbers for better results

i = 0


for (x,y,w,h) in faces:
    #print(f'Face{i}: x={x} y={y} w={w} h={h}')
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

    person = name = "Unknown"
    if not fresh:
        _id, conf = recognizer.predict(roi_gray)

        if conf <= 50:
            name = f"{str(labels[_id]).capitalize()} conf:{format(conf,'.2f')}"
            person = name.split(" ")[0]


        i += 1
    face = f"{person}{i}.png"
    cv2.imwrite(face, roi_color)

    if(name is "Unknown"):
        color = (0, 0, 255)
    else:
        color = (0, 255, 0)
    stroke = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    width = x + w
    height = y + h
    fontScale = width / 900
    cv2.rectangle(img, (x, y), (width, height), color, int(stroke*2))
    cv2.putText(img, name, (x, y-5), font, fontScale, color, stroke, cv2.LINE_AA)

img = cv2.resize(img, (1280, 720))
cv2.imshow('NOT Skynet', img)
cv2.waitKey(0)
cv2.destroyAllWindows()