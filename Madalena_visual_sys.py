import cv2 
import numpy as np
import pyttsx3
import random
from datetime import datetime
from playsound import playsound
from os import walk


f = []
for (dirpath, dirnames, filenames) in walk("/home/anthony/Documentos/Madalena/speak/"):
    f.extend(filenames)
    break

print(filenames)

engine = pyttsx3.init()
engine.setProperty('voice','brazil')
frases = [" FRASE "] #insira aqui as frases que serão sintetizadas


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

counter = 0
max = 100

while True:

    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.5,5)

    if len(faces) == 1 and counter > 10:
        for (x,y,w,h) in faces:
            roi = frame[y:y+h,x:x+w]
            cv2.imwrite("/home/anthony/Documentos/Madalena/visual_memory/%s.png"%dt_string,roi)
            cv2.rectangle(frame, (x,y),(x+w,y+h),(255,0,0),2)
        
        
        playsound("/home/anthony/Documentos/Madalena/speak/%s"%(filenames[random.randint(0,len(filenames)-1)]))
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
