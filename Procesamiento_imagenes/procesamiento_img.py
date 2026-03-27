# import necesarios
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import shutil
import zipfile
from google.colab.patches import cv2_imshow  #solo para colab
from google.colab import drive   #solo para colab

#"""Procesamiento de las imagenes"""

##conectar a drive // solo en colab
drive.mount('/content/drive')

#leer el csv
csv_path = '/content/drive/MyDrive/Estancia_Profesional/class_dict.csv'
datos  = pd.read_csv(csv_path, index_col="name") #poner los datos en un dataframe

personas = datos.loc['person'].values.tolist() #obtener el codigo de las personas

##descomprimir las imagenes // solo en colab
zip = ['train','val'] #nombre de ambas carpetas con las imagenes
for i in zip:
  ruta_zip = f'/content/drive/MyDrive/Estancia_Profesional/{i}.zip' #ruta en donde se almacena
  with zipfile.ZipFile(ruta_zip, 'r') as zip_ref:
    zip_ref.extractall('/content/dataset')

print(f"Descomprimido")

##Funcion para convertir las imagenes
def transformar_imagen(origen, destino):
  img = cv2.imread(origen) ## lee imagen
  img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convierte a rgb
  mask = np.all(img_rgb == personas, axis=-1) # encuentra a las personas en una máscara
  black = np.zeros_like(img_rgb) # crea una imagen en negro
  black[mask] = [255,255,255] #pinta las personas de blanco
  img_final = cv2.cvtColor(black,cv2.COLOR_RGB2BGR) # convierte a bgr
  cv2.imwrite(destino, img_final) #guarda la imagen

# crea una lista de string con la ubicacion de cada imagen en la carpeta "segmented"
for file in zip:
  path = f'/content/Original/{file}/segmented'
  if file == 'train':
    train_list = sorted([f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
  elif file == 'val':
    val_list = sorted([f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

#a cada imagen en las listas creadas, se aplica la segmentación de personas
for file in zip:
  # rutas en el entorno de colab
  origen = f'/content/Original/{file}/segmented'
  destino = f'/content/Entrenamiento/{file}/segmented'

  if not os.path.exists(destino): ##crea la carpeta si no existe
    os.makedirs(destino)

  if file == 'train':
    for i in train_list:
      origin = os.path.join(origen, i)
      destiny = os.path.join(destino, i)
      transformar_imagen(origin, destiny)
  elif file == 'val':
    for i in val_list:
      origin = os.path.join(origen, i)
      destiny = os.path.join(destino, i)
      transformar_imagen(origin, destiny)
