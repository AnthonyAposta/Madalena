import numpy as np 
import cv2
import os
import pickle


path =  "/home/anthony/Documentos/Madalena/lab_proce/"
recognizer = cv2.face.LBPHFaceRecognizer_create(radius = 1,grid_x = 7, grid_y = 7 ,neighbors=9)

def crate_train_dataset(path):
    X_train = []
    y_label = []
    labels_id = {}
    current_id = 0

    for nome in os.listdir(path):

        nome_path = os.path.join(path,nome)
        labels_id[current_id] = nome
        print(f"Loading file: {nome}, id={current_id}")

        for image in os.listdir(nome_path):

            image_path = os.path.join(nome_path,image)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            X_train.append(image)
            y_label.append(current_id)
            
        current_id += 1 

    return X_train,y_label,labels_id

x,y, id_ = crate_train_dataset(path)

recognizer.train(x,np.array(y))
recognizer.save("trainner.yml")

with open("labels.pickle", "wb") as f:
    pickle.dump(id_, f)
f.close()

with open("x_train.pickle", "wb") as f:
    pickle.dump(x, f)
f.close()

with open("y_labels.pickle", "wb") as f:
    pickle.dump(y, f)
f.close()


