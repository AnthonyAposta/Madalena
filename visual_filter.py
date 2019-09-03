import cv2
import numpy as np
from os import walk,makedirs,path

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

f = []
for (dirpath, dirnames, filenames) in walk("/home/anthony/Documentos/Madalena/lab_persons"):

        if len(dirnames)>0:
                names = dirnames
        
print(names)
for name in names:
        for (dirpath_2, dirnames_2, filenames_2) in walk("/home/anthony/Documentos/Madalena/lab_persons/"+name):
                num = 0
                for images in filenames_2:

                        
                        #print(dirpath_2+"/"+images)
                        img = cv2.imread(dirpath_2+'/'+images,cv2.IMREAD_GRAYSCALE)
                        img = cv2.resize(img,(100,100))
                        M_90 = cv2.getRotationMatrix2D((50,50),90,1)
                        M_270 = cv2.getRotationMatrix2D((50,50),270,1)
                        
                        img_1 = cv2.warpAffine(img,M_90,(100,100))
                        img_2 = cv2.warpAffine(img,M_270,(100,100))
                        transf = [img,img_1,img_2]
                        #print(img)
                        if len(img) > 0:
                                if not path.exists('/home/anthony/Documentos/Madalena/lab_proce/'+name):
                                        makedirs('/home/anthony/Documentos/Madalena/lab_proce/'+name,0o777,exist_ok=True)
                                for i in transf:
                                        num +=1
                                        cv2.imwrite('/home/anthony/Documentos/Madalena/lab_proce/'+name+'/'+str(num)+".png",i)
                        print("done")








#img = cv2.imread("/home/anthony/Documentos/Madalena/visual_memory/test.jpeg",1)
#cv2.imwrite("testgray.jpeg",img)
#print(img)
#cv2.imshow('img',img)
#cv2.waitKey(0)
#cv2.destroyAllwindows()

"""
for imagem in f:
    img = cv2.imread('/home/anthony/Documentos/Madalena/visual_memory/'+str(imagem),cv2.IMREAD_GRAYSCALE)
    img = face_cascade.detectMultiScale(img,1.5,5)
    if len(img) > 0:
        cv2.imwrite('/home/anthony/Documentos/Madalena/proce_memory/'+str(imagem),img)
        print("done")
"""
         

