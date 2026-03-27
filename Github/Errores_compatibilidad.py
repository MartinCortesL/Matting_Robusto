# Se corrigieron errores de compatibilidad del matting robusto con GoogleColab actual
###"""Modelo del github"""

#Clonar el github al entorno de colab
!git clone https://github.com/PeterL1n/RobustVideoMatting.git
!cd RobustVideoMatting

!pip install -r RobustVideoMatting/requirements_inference.txt  #instalar el requirements_inference.txt

#obtiene los pesos del modelo
!wget https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3.pth


# cargar el modelo.
from RobustVideoMatting.model import MattingNetwork

model = MattingNetwork('mobilenetv3')  # or "resnet50"
model.load_state_dict(torch.load('rvm_mobilenetv3.pth')) #poner los pesos en el modelo

!pip install av tqdm pims  #instalar el av

#Librerias para el modelo
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from RobustVideoMatting.inference_utils import VideoReader, VideoWriter
from RobustVideoMatting.inference_utils import ImageSequenceReader, ImageSequenceWriter

!gdown https://drive.google.com/uc?id=1I0v72-hNlK1hm9q1OwyaATUYApXpotS6 -O input.mp4  ##video de prueba del creador

#función para parchear y corregir el error de compatibilidad
import av
from fractions import Fraction

def patched_init(self, path, frame_rate, bit_rate=1000000):
    self.container = av.open(path, mode='w')
    # Cambia el f-string por Fraction
    self.stream = self.container.add_stream('h264', rate=Fraction(frame_rate))
    self.stream.pix_fmt = 'yuv420p'
    self.stream.bit_rate = bit_rate

# Aplica el parche a la clase VideoWriter
VideoWriter.__init__ = patched_init
print("parcheada")

#cambiar el modelo a cuda
model = model.eval().cuda()

#Prueba para el video
reader = VideoReader('input.mp4', transform=ToTensor())
writer = VideoWriter('output.mp4', frame_rate=25)  ##24-25 para tener la misma cantidad de segundos

bgr = torch.tensor([.47, 1, .6]).view(3, 1, 1).cuda()  # Green background.                         
rec = [None] * 4                                       # Initial recurrent states.
downsample_ratio = 0.25                                # Adjust based on your video.

with torch.no_grad():
    for src in DataLoader(reader):                     # RGB tensor normalized to 0 ~ 1.
        fgr, pha, *rec = model(src.cuda(), *rec, downsample_ratio)  # Cycle the recurrent states.  
        com = fgr * pha + bgr * (1 - pha)              # Composite to green background.
        writer.write(com)                              # Write frame.
    writer.close()
print("Procesamiento finalizado")

from IPython.display import HTML
from base64 import b64encode

#Mostrar los videos con html y python

mp4_1 = open('input.mp4','rb').read()
data_url = "data:video/mp4;base64,"+b64encode(mp4_1).decode()
mp4_2 = open('output.mp4','rb').read()
data_url2 = "data:video/mp4;base64,"+b64encode(mp4_2).decode()

HTML(
    """
    <div style="display:flex;">
        <video width = 400 controls style="margin-right:10px;"><source src = "%s" type = "video/mp4"></video>
        <video width = 400 controls><source src = "%s" type = "video/mp4"></video>
    </div>
    """ % (data_url, data_url2)
)

### Prueba para las imagenes
#cargar la carpeta con las imagenes desde colab (imagenes previamente descargadas)
reader = ImageSequenceReader('dataset/val/imgs', transform=ToTensor())
writer = ImageSequenceWriter('output_folder/', extension='jpg')

bgr = torch.tensor([.47, 1, .6]).view(3, 1, 1).cuda()  ##cuda
with torch.no_grad():
    for src in DataLoader(reader):
        rec = [None] * 4 #reinicia el estado recurrente si no son una secuencia
        fgr, pha, *rec = model(src.cuda(), *rec, downsample_ratio)        #cuda
        com = fgr * pha + bgr * (1 - pha)
        writer.write(com)
print("Procesamiento finalizado.")
