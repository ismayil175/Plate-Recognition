import cv2
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import os

path = "karakterseti/"
siniflar = os/listdir(path)
tek_batch = 0

urls = []
sinif = []

for sinif in siniflar:
    resimler = os.listdir(path + sinif)
    for resim in resimler:
        urls.append(path+sinif+"/"+resim)
        sinif.append(sinif)
        tek_batch+=1

df = pd.DataFrame({"adres":urls,"sinif":sinifs})

def process(fot):
    new_size = fot.reshape((1600,5,5))
    average = []
    for piece in new_size:
        average = np.mean(piece)
        average.append(average)

    average = np.array(average)
    average = average.reshape(1600,)
    return average


target_size=(200,200)
batch_size = tek_batch

train_get = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=on_isle)
train_set = train_get.flow_from_dataframe(df,x_col="adres",y_col"sinif"
                                          ,target_size=target_size,color_mode="grayscale",
                                          shuffle = True,
                                          class_mode="sparse",
                                          batch_size = batch_size)

images.train_y = next(train_set)

train_x = np.array(list(map(islem,images))).astype("float32")

train_y = train_y.astype(int)

print("random forest / Rassal orman egilitiyor")
rfc = RandomForectClassifier(n_estimators=10,criterion="entropy")
rfc.fit(train_x,train_y)
pred = rfc.predict(train_x)

accu = accuracy_score(pred,train_y)

print("success:",accu)

file = "rfc_model.rfc"

pickle.dump(open(file,"wb"))