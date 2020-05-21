import cv2 
import sys
import numpy as np
import random
from datetime import datetime
from os import walk, path, makedirs, listdir
import pickle

class visual_sys(object):

    def __init__(self):

        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.name = sys.argv[1]
        
        # cria o diretorio com o nome se ele não existe
        if not path.exists(f"/home/anthony/Documentos/Madalena/lab_persons/{self.name}"):
            makedirs(f"/home/anthony/Documentos/Madalena/lab_persons/{self.name}",0o777,exist_ok=True)

    def start_capture(self):
        cap = cv2.VideoCapture(1)
        counter = 0
        
        while True:
            now = datetime.now()
            dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")

            ret, frame = cap.read()
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray,1.5,5)
            
            #se rosto detectado e contador maior que 10
            if len(faces) == 1 and counter > 10:
            
                for (x,y,w,h) in faces:
                    # regiao do rosto
                    roi = frame[y:y+h,x:x+w]

                    # salvar imagens até chegar em 50
                    if len(listdir(f"/home/anthony/Documentos/Madalena/lab_persons/{self.name}")) < 50:
                        cv2.imwrite(f"/home/anthony/Documentos/Madalena/lab_persons/{self.name}/{dt_string}.png",roi)
                        cv2.rectangle(frame, (x,y),(x+w,y+h),(255,0,0),2)

            cv2.imshow('frame',frame)
            counter += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

visual = visual_sys()
visual.start_capture()