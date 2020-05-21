import cv2 
import numpy as np
import pyttsx3
import random
from datetime import datetime
from playsound import playsound
from os import walk
import pickle


f = []
for (dirpath, dirnames, filenames) in walk("/home/anthony/Documentos/Madalena/speak/"):
    f.extend(filenames)
    break

print(filenames)

engine = pyttsx3.init()
engine.setProperty('voice','brazil')
frases = {'adolfo': "Adolfo detectado",
         'chico': "Chico detectado",
         'tadeu':"tadeu detectado",
         'anthony': "anthony detectado",
         'valeria': "valeria detectada",
         'cafe': "cafe detectado",
         'heloise': "heloise detectada",
         'romulo': "romulo detectado",
         'thiago': "thiago detectado",
         'gi':"gi detectada"}

#formas geometricas e texto estã no tutorial 3

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

with open("labels.pickle", "rb") as f:
    labels = pickle.load(f)
f.close()


cap = cv2.VideoCapture(0)

counter = 0
max = 100
img_num = 0
while True:

    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.5,5)
    if len(faces) == 1 and counter > 10:
    
        for (x,y,w,h) in faces:
            roi = frame[y:y+h,x:x+w]
            roi = cv2.resize(roi,(100,100))
            
            pred_roi = gray[y:y+h,x:x+w]
            roi_roi = cv2.resize(roi,(100,100))
            id_, conf = recognizer.predict(pred_roi)
            if conf <= 50:
                cv2.rectangle(frame, (x,y),(x+w,y+h),(255,0,0),2)
                print(labels[id_], conf)
                engine.say(frases[str(labels[id_])])
                engine.runAndWait()


        #playsound("/home/anthony/Documentos/Madalena/speak/%s"%(filenames[random.randint(0,len(filenames)-1)]))
        
        #p = vlc.MediaPlayer("/home/anthony/Documentos/sentdex/opencv/audios/%s"%(filenames[random.randint(0,len(filenames)-1)]))
        #p.play()
        #mixer.init()
        #mixer.music.load("/home/anthony/Documentos/sentdex/opencv/audios/%s"%(filenames[random.randint(0,len(filenames)-1)]))
        #mixer.music.play()
        #wave.open("/home/anthony/Documentos/sentdex/opencv/faustao-errou.mp3",'r')   
        #print("To vendo um doente")
        #engine.say(frases[random.randint(0,len(frases)-1)])
        #engine.say("Chegou o Corno.")
        #engine.runAndWait()
        #cv2.imwrite("/home/anthony/Documentos/sentdex/opencv/imagens/test.png",roi)
        counter = 0

    cv2.imshow('frame',frame)

    counter += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



