# Aprendizaje profundo con memoria temporal para el matting robusto de figuras humanas en entornos aéreos dinámicos

## Introducción

En el artículo [RVM: Robust High-Resolution Video Matting with Temporal Guidance](https://peterl1n.github.io/RobustVideoMatting/#/) [1] se desarrolló un modelo de inteligencia artificial y visión artificial permite obtener el mating, el cual es el proceso de predecir el alpha matte, de una imagen con personas, donde el modelo remplaza el fondo de la imagen en tiempo real con un color verde.
Este modelo utiliza información temporal para ver las caracteristicasde cada frame anterior al actual, lo que hace que no necesite trimaps ni imagenes de fondo previamente obtenidas. Esto lo consigue al tener una arquitectura recurrente para analizar la información temporal de frames anteriores. Está basado en mobileNet V3-Large y ResNet-50, donde emplealos siguientes 3 bloques de manera general:
  - Bloque de BottleNeck: Opera en una escala de caracteristicas de 1/16 despues del modulo de segmentación.
  - Bloque de UpSampling: Se repite en una escala de 1/8, 1/4 y 1/2 aplicando una convolución recurrente.
  - Bloque de Output: En este bloque se refinan los resultados obtenidos.
Usa un modulo de Filtrado Guiado (DGF) donde pasan imagenes de baja calidad a traves de este filtro para producir alta resolución en el video o imagen resultante.
### Limitaciones
El modelo prefiere videos con objetos claros. Cuando hay personas en el fondo, los sujetos de interes se vuelven ambiguos. Esto favorece a fondos simples para producir mas exactitud en el matting.
###
Por lo tanto, lo que se busca implementar en este trabajo es un fine-tuning del modelo RVM con e fin de especializarlo en imagenes que contienen vistas aereas obtenidas de un dron.

## Dataset
El objeto que se busca segmentar para este trabajo son las personas, por lo tanto, en las imágenes ubicadas en: [DronSafe-Landing: A Semi-Supervised Dataset for Urban Aerial Semantic Segmentation](https://zenodo.org/records/17614252?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjVhYzRjOWE4LWFhZjctNDI4Mi1iZDJjLWE2N2UzNGI4MTk4MSIsImRhdGEiOnt9LCJyYW5kb20iOiI1Y2M3MmVlM2MwNjQ5NDNkMzYzZmMwMTgyMjlhZDQ4MSJ9.QdRdtjTefigdGW4wZTdTwsvO6ilAz2kqorrEmkeY2Kwo0BeXl6h4lkbUJLYIxqvt7BZRulAhaep5NQUhUTm_sg) [2], se consiguió el código RGB el cual es (255,22,96), con él se segmentó las imagenes con blanco (255,255,255) donde hubiera personas y con negro (0,0,0) el resto de la imagen, haciendo esto para el total de imagenes descargados. El dataset obtenido para este trabajo se encuentra en [3]:

La función utilizada fue la siguiente:
~~~python
def transformar_imagen(origen, destino):
  img = cv2.imread(origen) ## lee imagen
  img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convierte a rgb
  mask = np.all(img_rgb == personas, axis=-1) # encuentra a las personas en una máscara
  black = np.zeros_like(img_rgb) # crea una imagen en negro
  black[mask] = [255,255,255] #pinta las personas de blanco
  img_final = cv2.cvtColor(black,cv2.COLOR_RGB2BGR) # convierte a bgr
  cv2.imwrite(destino, img_final) #guarda la imagen
~~~
Donde, después de obtener la ubicación de las personas en una máscara y pintar dicha ubicación en una imagen negra del mismo tamaño que la original, se almacena el resultado.

## Modelo de GitHub
Al momento de probar el modelo que se encuentra en el github del autor ([RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting/tree/master?tab=readme-ov-file)) se encontraron algunos problemas a la hora de ejecutar el `convert_video` y el `VideoWriter`, por lo que se siguieron los siguientes pasos para probar el modelo.
Primero, en el entorno de Google Colab se creó una copia del repositorio de github para llamar directamente desde ahí los métodos.
Luego, se instalaron los requerimientos de inferencia, se obtuvieron los pesos del modelo y se cargó el modelo MobileNet V3 con dichos pesos a la GPU del entorno.
Después se importaron las librerías para el `VideoReader` y el `VideoWriter`, y se aplicó un parche al constructor de la clase `VideoWriter` ubicada en `RobustVideoMatting/inference_utils.py`, en cual, es el siguiente:
~~~python
import av
from fractions import Fraction


def patched_init(self, path, frame_rate, bit_rate=1000000):
    self.container = av.open(path, mode='w')
    # Cambió el f-string por Fraction
    self.stream = self.container.add_stream('h264', rate=Fraction(frame_rate))
    self.stream.pix_fmt = 'yuv420p'
    self.stream.bit_rate = bit_rate


#Aplica el parche a la clase VideoWriter
VideoWriter.__init__ = patched_init
print("parche aplicado")
~~~
Finalmente, se ejecutó el código indicado en el repositorio, logrando que el modelo procese correctamente los videos.
## Entrenamiento
Debido a que el modelo fue creado especificamente para videos y a limitaciones en el codigo de `train.py`, se creo una clase para procesar la imagen original junto a su mascara binaria.

Para calcular el rendimiento del modelo, se utilizó la función `MattingLoss` que fue hecha por los autores del trabajo, adémas se calcularon diferentes metricas como el BCEWithLogitsLoss, la unión sobre la intercección (IoU) y el coeficiente DICE.

La validación se realiza antes de empezar cada ciclo de entrenamiento y muestra una comparación entre la imagen real, la mascara de esa imagen y la imagen obtenida por el modelo.

El entrenamiendo realiza un fine-tuning sobre la capa de `Decoder` y `project_mat` del modelo donde lo hiperparametros que recibe son el learning rate y el batch size, al igual que la cantidad de epocas. 

Al finalizar cada entrenamiento se muestra la perdida obtenida por el `Matting Loss` y el BCEWithLogitsLoss, y las metricas de IoU junto con el coeficiente DICE.
## Resultados
## Conclusión
-- -- 
## Referencias
[1] Lin, S., Yang, L., Saleemi, I., & Sengupta, S. (2021). Robust High-Resolution Video Matting with Temporal Guidance. arXiv preprint arXiv:2108.11515.
<details>
<summary><b>Click para ver BibTeX</b></summary>
@misc{rvm,
  title={Robust High-Resolution Video Matting with Temporal Guidance}, 
  author={Shanchuan Lin and Linjie Yang and Imran Saleemi and Soumyadip Sengupta},
  year={2021},
  eprint={2108.11515},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
</details>
[2] M. S. Soriano-Garcia, D. Mercado, . israel . becerraand J. De La Torre-Vanegas, “DronSafe-Landing: A Semi-Supervised Dataset for Urban Aerial Semantic Segmentation”. Zenodo, Nov. 15, 2025. doi: 10.5281/zenodo.17614252.
<details>
<summary><b>Click para ver BibTeX</b></summary>
@misc{soriano_garcia_2025_17614252,
  author       = {Soriano-Garcia, Miguel S. and
                  Mercado, Diego and
                  becerra, israel and
                  De La Torre-Vanegas, Julio},
  title        = {DronSafe-Landing: A Semi-Supervised Dataset for
                   Urban Aerial Semantic Segmentation
                  },
  month        = nov,
  year         = 2025,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17614252},
  url          = {https://doi.org/10.5281/zenodo.17614252},
}
</details>
[3] J. M. Cortes Lozano, “Dataset de aprendizaje profundo con memoria temporal para el matting robusto de figuras humanas en entornos aéreos dinámicos”. Zenodo, Apr. 18, 2026. doi: 10.5281/zenodo.19637728.
<details>
<summary><b>Click para ver BibTeX</b></summary>
@dataset{cortes_lozano_2026_19637728,
  author       = {Cortes Lozano, Jose Martin},
  title        = {Dataset de aprendizaje profundo con memoria
                   temporal para el matting robusto de figuras
                   humanas en entornos aéreos dinámicos
                  },
  month        = apr,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {Version 1.0},
  doi          = {10.5281/zenodo.19637728},
  url          = {https://doi.org/10.5281/zenodo.19637728},
}
</details>
