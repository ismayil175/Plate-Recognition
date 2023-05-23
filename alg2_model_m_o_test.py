import imghdr
from msilib import datasizemask
from re import L
import cv2
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import os
import pickle

def islem(img):
    yeni_boy = img.reshape((1600,5,5))
    orts = []
    for parca in yeni_boy:
        ort = np.mean(parca)
        orts.append(ort)
    orts = np.array(orts)
    orts = orts.reshape(1600,)
    return orts

path = "karakterseti/"
siniflar = os.listdir(path)
tek_batch = 0

urls = []
sinifs = []

print ("DATA OKUNUYOR")

for sinif in siniflar:
    resimler = os.listdir(path + sinif)
    for resim in resimler:
        urls.append(path+sinif+"/"+resim)
        sinifs.append(sinif)
        tek_batch+=1

df = pd.DataFrame({"adres":urls,"sinif":sinifs})

dosya = "rfc_model.rfc"
rfc = pickle.load(open(dosya,"rb"))

index = list(sinifs.values())
siniflar = list(sinifs.keys())
df = df.sample(frac=1)

for adress,clas in df.values:
    image = cv2.imread(adress,0)
    foto = cv2.resize(image,200,200)
    foto = foto/250
    attributes = islem(foto)

    sonuc = rfc.predict({attributes}) [0]

indexi =  index.index(sonuc)
sinif = siniflar[]
plt.imshow(resim,cmap="gray")
plt.title(f"fotoragtaki karakter: {sinif}")
plt.show()

fot = 
kon1 =
kon2 =
kon3 =
boxa = 
minx = 
miny = 
w = 
h=  
kon = 
single_batch = 
classes = 

print("Rasgele Orman/ Algoritma eğitiliyor")