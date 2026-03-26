# Matting_Robusto

### Artículo
Se leyó el artículo RVM: Robust High-Resolution Video Matting with Temporal Guidance para entender el matting, donde el modelo  reemplaza el fondo en tiempo real con un fondo verde y utiliza la información temporal para ver las características de cada frame anterior y actual. Además, no necesita trimaps ni imágenes de fondo previamente obtenidas, ya que utiliza una arquitectura recurrente para analizar la información temporal de frames anteriores. Utiliza el modelo MobileNet V3-Large y Resnet-50. Su modelo tiene 3 bloques: el bottleneck, el upsampling y el output, al igual que usa un módulo de filtrado guiado (DGF). Su entrenamiento se basó en 4 pasos, donde, en cada uno iban ajustando los hiperparametros, las épocas y la cantidad de frames que recibían de los videos.
Entre sus resultados se encuentra que, su modelo segmenta mejor a los humanos que otros modelos, igualmente funciona mejor con fondos dinámicos que otros y pueden aumentar el tamaño de su modelo para aplicaciones desde un servidor.
Sus limitaciones son que el modelo es ideal con objetos claros, si hay personas en el fondo empeora el resultado, al igual que si hay fondos complejos.
### Procesamiento de las Imágenes
Primero se definió el objeto a segmentar de las imágenes ubicadas en: DronSafe-Landing: A Semi-Supervised Dataset for Urban Aerial Semantic Segmentation, el cual son las personas. Su código RGB es (255,22,96), con él se segmentó las imagenes con blanco (255,255,255) donde hubiera personas y con negro (0,0,0) el resto de la imagen, haciendo esto para el total de imagenes descargados.
La función utilizada fue la siguiente:

def transformar_imagen(origen, destino):
  img = cv2.imread(origen) ## lee imagen
  img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convierte a rgb
  mask = np.all(img_rgb == personas, axis=-1) # encuentra a las personas en una máscara
  black = np.zeros_like(img_rgb) # crea una imagen en negro
  black[mask] = [255,255,255] #pinta las personas de blanco
  img_final = cv2.cvtColor(black,cv2.COLOR_RGB2BGR) # convierte a bgr
  cv2.imwrite(destino, img_final) #guarda la imagen

Donde, después de obtener la ubicación de las personas en una máscara y pintar dicha ubicación en una imagen negra del mismo tamaño que la original, se almacena el resultado.

### GitHub
Al momento de probar el modelo que se encuentra en el github del creador (RobustVideoMatting) se encontraron algunos problemas a la hora de ejecutar el convert_video y el VideoWriter, por lo que se siguieron los siguientes pasos para probar el modelo.
Primero, en el entorno de Google Colab se creó una copia del repositorio de github para llamar directamente desde ahí los métodos.
Luego, se instalaron los requerimientos de inferencia, se obtuvieron los pesos del modelo y se cargó el modelo MobileNet V3 con dichos pesos a la GPU del entorno.
Después se importaron las librerías para el VideoReader y el VideoWriter, y se aplicó un parche al constructor de la clase VideoWriter ubicada en RobustVideoMatting/inference_utils.py, en cual, es el siguiente:

import av
from fractions import Fraction


def patched_init(self, path, frame_rate, bit_rate=1000000):
    self.container = av.open(path, mode='w')
    # Cambia el f-string por Fraction
    self.stream = self.container.add_stream('h264', rate=Fraction(frame_rate))
    self.stream.pix_fmt = 'yuv420p'
    self.stream.bit_rate = bit_rate


#Aplica el parche a la clase VideoWriter
VideoWriter.__init__ = patched_init
print("parcheada")

Finalmente, se ejecutó el código indicado en el repositorio, logrando que el modelo procese correctamente los videos.
### Entrenamiento
—
