# Aprendizaje profundo con memoria temporal para el matting robusto de figuras humanas en entornos aéreos dinámicos

## Introducción

En el presente repositorio se encuentra los codigos utilizados para el entrenamiento de los arquitecturas de visión artificial, espeficicamente Robust Video Matting (RVM) y H-Net, que se presentan a continuación.

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

## Modelo de GitHub RVM
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
## RVM
### Descripción
Robust Video Matting (RVM) es un método de matting de video en tiempo real y alta resolución propuesto por [1]. A diferencia de los métodos tradicionales que procesan cada fotograma de forma independiente, RVM utiliza una arquitectura recurrente para explotar la información temporal del video, logrando resultados más coherentes y robustos sin necesidad de entradas auxiliares como trimaps o imágenes de fondo precapturadas.
### Características principales
- Tiempo real en alta resolución: procesa video 4K a 76 FPS y HD a 104 FPS en una GPU NVIDIA GTX 1080Ti
- Arquitectura recurrente: aprovecha la información temporal para mejorar la coherencia entre fotogramas y reducir el parpadeo (flicker)
- Sin entradas auxiliares: no requiere trimap ni imagen de fondo precapturada
- Modelo ligero: solo 3.749 millones de parámetros (14.5 MB), frente a los 6.487 M de MODNet o los 60.996 M de DeepLabV3
- Entrenamiento dual: combina objetivos de matting y segmentación semántica simultáneamente para mejorar la robustez
### Arquitectura
El modelo se compone de tres módulos principales:
- Encoder de extracción de características: basado en MobileNetV3-Large con módulo LR-ASPP, extrae mapas de características a escalas 1/2, 1/4, 1/8 y 1/16​ para cada fotograma individual
- Decoder recurrente: emplea unidades ConvGRU a múltiples escalas para agregar información temporal. Opera sobre la mitad de los canales mediante división y concatenación, reduciendo el costo computacional sin sacrificar rendimiento
- Módulo Deep Guided Filter (DGF): permite el muestreo ascendente (upsampling) a alta resolución de forma eficiente y entrenable de extremo a extremo.

El codigo utlizado para realizar el entrenamiento con RVM se encuentra [aquí](./RVM)
## H-Net
### Descripción
H-Net es una arquitectura de red neuronal convolucional (CNN) diseñada para recuperar información estructural de imágenes degradadas por medios dispersivos, como niebla, humo o tejido biológico. Propuesta por [4], esta arquitectura ofrece un balance entre precisión y eficiencia computacional, logrando resultados comparables a modelos más complejos como U-Net con una fracción de los recursos necesarios.
### Características principales
- Bajo costo computacional: solo 18,999 parámetros entrenables frente a los más de 21 millones de U-Net
- Eficiencia en memoria: consume 3.3 GB de memoria GPU frente a los 11.5 GB de U-Net
- Velocidad de inferencia: 4 segundos para 100 imágenes frente a los 11 segundos de U-Net
- SSIM promedio de 0.8 en reconstrucción de estructuras en medios dispersivos
- Versatilidad: aplicable a eliminación de neblina, reconstrucción de estructuras vasculares y segmentación semántica
### Arquitectura
H-Net se basa en la arquitectura MS-D e incorpora dos tipos de bloques principales:
- Bloques de convolución dilatada (DCB): capturan características locales y globales de la imagen mediante distintas tasas de dilatación, permitiendo un campo receptivo amplio sin incrementar el número de parámetros
- Bloques de convolución estándar (CB): refinan las características extraídas por los DCB para mejorar la precisión de la reconstrucción

El codigo utilizado para entrenar el modelo H-Net se encuentra [aquí](./H-net)
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

[4] R. Chiu-Coutino, M. S. Soriano-Garcia, C. I. Medel-Ruiz,  S. M. Afanador-Delgado, E. Villafaña-Rauda, and R. Chiu, 
"Breaking through scattering: The H-Net CNN model for image retrieval," Computer Methods and Programs in Biomedicine, 
vol. 265, p. 108723, 2025, doi: 10.1016/j.cmpb.2025.108723.
<details>
  <summary><b>Click para ver BibTex</b></summary>
    @article{chiucoutino2025,
  author    = {Chiu-Coutino, Roger and Soriano-Garcia, Miguel S. and 
               Medel-Ruiz, Carlos Israel and Afanador-Delgado, S.M. and 
               Villafaña-Rauda, Edgar and Chiu, Roger},
  title     = {Breaking through scattering: The {H-Net} {CNN} model for 
               image retrieval},
  journal   = {Computer Methods and Programs in Biomedicine},
  volume    = {265},
  pages     = {108723},
  year      = {2025},
  doi       = {10.1016/j.cmpb.2025.108723}
}
</details>
