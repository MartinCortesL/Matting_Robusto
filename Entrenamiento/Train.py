# Librerias necesarias
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import os
from RobustVideoMatting.model import MattingNetwork
from RobustVideoMatting.train_loss import matting_loss_light as matting_loss
import torch.optim as optim
import torch.nn as nn
import time
from tqdm import tqdm
import csv

#Clase para el entrenamiento
class SimpleMattingDataset(Dataset):
    def __init__(self, img_dir, mask_dir, size=(512, 512)):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.filenames = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]
        self.img_transform = T.Compose([
            T.Resize(size),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]
            )
        ])

        self.mask_transform = T.Compose([
            T.Resize(size),
            T.ToTensor()  # SIN normalize
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]
        # Cargar imagen y máscara (asegurando que la máscara sea escala de grises 'L')
        img = Image.open(os.path.join(self.img_dir, name)).convert('RGB')
        mask = Image.open(os.path.join(self.mask_dir, name)).convert('L')

        # Retornar como tensores [3, H, W] y [1, H, W]
        return self.img_transform(img), self.mask_transform(mask)

#Función para el IoU
def IoU(pred_pha, true_pha, threshold=0.5):
    # 1. Convertir a máscaras binarias (0 o 1)
    pred_bin = (pred_pha > threshold).float()
    true_bin = (true_pha > threshold).float()

    # 2. Calcular Intersección y Unión
    # .view(-1) aplana los tensores para procesar todos los píxeles del batch
    dims = tuple(range(1, pred_bin.ndim))

    intersection = (pred_bin * true_bin).sum(dim=dims)
    union = pred_bin + true_bin - (pred_bin * true_bin)
    union = union.sum(dim=dims)

    # 3. Evitar división por cero
    iou = (intersection + 1e-7) / (union + 1e-7)

    # Retornamos el promedio del batch
    return iou.mean().item()

#Función para el Coeficiente Dice
def dice(pred, target):
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    dims = tuple(range(1, pred.ndim))
    intersection = (pred * target).sum(dim=dims)

    dice_score = (2. * intersection + 1e-7) / (pred.sum(dim=dims) + target.sum(dim=dims) + 1e-7)

    return dice_score.mean()

#Función para la validación
def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    val_Bloss = 0
    val_iou = 0
    val_dice = 0
    i = 0

    with torch.no_grad():
        for imgs, masks in tqdm(val_loader, desc="Validación", leave = False):
            i += 1
            imgs, masks = imgs.to(device), masks.to(device)

            # Ajustar a 5D si es necesario
            imgs = imgs.unsqueeze(1)
            masks = masks.unsqueeze(1)

            true_fgr = imgs * masks  #obtiene el objeto solo
            true_fgr = true_fgr + 1e-7

            # Forward pass
            # RVM devuelve (fgr, pha, *others)
            rec = [None]*4
            pred_fgr, pred_pha, *rec = model(imgs)

            # Calcular Loss
            losses = matting_loss(pred_fgr, pred_pha, true_fgr, masks)
            losses['total'] = losses['pha_l1'] + losses['fgr_l1']

            loss = losses['total']
            val_Bloss += criterion(pred_pha, masks).item()
            
            val_loss += loss.item()

            # Calcular Métrica
            pred_pha  = torch.sigmoid(pred_pha)
            val_iou  += IoU(pred_pha, masks)
            val_dice += dice(pred_pha, masks).item()

    # Promedios finales
    avg_loss = val_loss / len(val_loader)
    avg_Bloss = val_Bloss / len(val_loader)
    avg_iou  = val_iou / len(val_loader)
    avg_dice = val_dice / len(val_loader)

    return avg_loss, avg_Bloss, avg_iou, avg_dice

#Función para probar y comparar las imagenes
def test_and_compare(model_path, img_path, mask_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Cargar el modelo
    model = MattingNetwork('mobilenetv3').to(device)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.eval()

    # 2. Preparar la imagen y la máscara real
    img_org = Image.open(img_path).convert('RGB')
    mask_org = Image.open(mask_path).convert('L') # Cargar máscara real en escala de grises

    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor()
    ])

    img_tensor = transform(img_org).unsqueeze(0).unsqueeze(0).to(device)

    # 3. Inferencia
    with torch.no_grad():
        # RVM devuelve (fgr, pha, estados recurrentes)
        pred_fgr, pred_pha, pred_seg, *others = model(img_tensor)
        pred_pha = torch.sigmoid(pred_pha)

    # 4. Post-procesamiento
    # Predicción a Numpy
    pha_pred = pred_pha.squeeze().cpu().numpy()
    pha_pred = (pha_pred * 255).astype(np.uint8)

    # Máscara real a Numpy (para comparar)
    mask_true = mask_org.resize((256, 256))
    mask_true_np = np.array(mask_true)

    # 5. Visualización Comparativa
    plt.figure(figsize=(15, 5))

    # Subplot 1: Imagen Original
    plt.subplot(1, 3, 1)
    plt.imshow(img_org.resize((256, 256)))
    plt.title("Imagen Original")
    plt.axis("off")

    # Subplot 2: Máscara Real (Ground Truth)
    plt.subplot(1, 3, 2)
    plt.imshow(mask_true_np, cmap='gray')
    plt.title("Máscara Real (Esperada)")
    plt.axis("off")

    # Subplot 3: Predicción del Modelo
    plt.subplot(1, 3, 3)
    plt.imshow(pha_pred, cmap='gray')
    plt.title("Predicción del Modelo")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

#Función para la grafica de comparación entre las métricas
def plot_training_results(Intento, train_losses, val_losses, train_ious, val_ious, train_dices, val_dices, train_Mloss, val_Mloss):
    epochs = range(len(train_losses))

    plt.figure(figsize=(14, 5))

    # Gráfico 1: Pérdidas (Losses)
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, 'b-o', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-o', label='Val Loss')
    plt.title('Pérdida (Loss)')
    plt.xlabel('Épocas')
    plt.ylabel('Valor de Pérdida')
    plt.legend()
    plt.grid(True)

    # Gráfico 2: Métricas (IoU y Dice)
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_ious, 'g--', label='Train IoU')
    plt.plot(epochs, val_ious, 'g-', label='Val IoU')
    plt.plot(epochs, train_dices, 'm--', label='Train Dice')
    plt.plot(epochs, val_dices, 'm-', label='Val Dice')
    plt.title('Métricas (IoU & Dice)')
    plt.xlabel('Épocas')
    plt.ylabel('Puntaje (0-1)')
    plt.legend()
    plt.grid(True)

    #Grafico 3: Matting Loss
    plt.subplot(1, 3, 3)
    plt.plot(epochs, train_Mloss, 'c--', label='Train Matting')
    plt.plot(epochs, val_Mloss, 'c-', label='Val Matting')
    plt.title('Matting Loss')
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'/content/drive/MyDrive/Estancia_Profesional/Graficas_train/resultado_intento_{Intento}.png')
    plt.show()
    plt.close()

#Función para realizar el entrenamiento
def train_model(
    Intento = 1,
    Epocas = 1,
    LR = 0.0001,
    BATCH_SIZE = 8,
    CHECKPOINT_PATH = "/content/"
    ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MattingNetwork('mobilenetv3').to(device)  # or "resnet50"
    model.load_state_dict(torch.load('/content/rvm_mobilenetv3.pth', map_location = device), strict=False)


    # 3. Congelar todo
    for param in model.parameters():
            param.requires_grad = False
    # 2. Descongela el decoder
    for param in model.decoder.parameters():
            param.requires_grad = True
    for param in model.project_mat.parameters():
            param.requires_grad = True

    # 4. Configurar Datos
    dataset = SimpleMattingDataset(img_dir='/content/dataset/train/fgr', mask_dir='/content/dataset/train/pha')
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    val_dataset = SimpleMattingDataset(img_dir='/content/dataset/val/fgr', mask_dir='/content/dataset/val/pha')
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # 5. Optimizador (Solo para las capas activas)
    optimizer = optim.Adam([
            {'params': model.decoder.parameters(), 'lr': LR},
            {'params': model.project_mat.parameters(), 'lr': LR},
        ])
    pos_weight = torch.tensor([20.0]).to(device)  # penaliza el fondo
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)  #Error con una capa sigmoide y BCELoss

    history = {
    'train_loss': [], 'val_loss': [],
    'train_iou': [], 'val_iou': [],
    'train_dice': [], 'val_dice': [],
    'train_Mloss': [], 'val_Mloss': []
    }

    best = 0
    for epoch in range(Epocas):
        #inicializa las metricas de validación
        total_Mloss_train = 0
        total_Bloss_train = 0
        total_Mloss_val = 0
        total_Bloss_val = 0
        total_iou_train = 0
        total_iou_val = 0
        total_dice = 0
        #validación antes del entrenamiento
        total_Mloss_val, total_Bloss_val, total_iou_val, total_dice_val = validate(model, val_loader, criterion, device)
        #grafica de comparación
        if epoch == 0:
          test_and_compare('/content/rvm_mobilenetv3.pth', '/content/dataset/val/fgr/101_99.jpg', '/content/dataset/val/pha/101_99.jpg')
        else:
          test_and_compare(f'{CHECKPOINT_PATH}/{Intento}_checkpoint.pth', '/content/dataset/val/fgr/101_99.jpg', '/content/dataset/val/pha/101_99.jpg')
        model.train()
        #inicia a contar el tiempo
        start = time.time()
        for imgs, masks in tqdm(dataloader, desc=f"Epoca {epoch}", leave = False):
            M_loss = 0
            B_loss = 0
            val = 0
            imgs, masks = imgs.to(device), masks.to(device)
            # RVM espera 5D [B, T, C, H, W].
            imgs = imgs.unsqueeze(1)
            masks = masks.unsqueeze(1)

            true_fgr = imgs * masks
            true_fgr = true_fgr + 1e-7

            optimizer.zero_grad()

            # Forward: RVM devuelve (fgr, pha, recurrent_states)
            rec = [None]*4
            pred_fgr, pred_pha, *rec = model(imgs, None, None, None, None)


            # 2. Calcular la pérdida usando la función oficial
            # Nota: matting_loss devuelve un DICCIONARIO con diferentes componentes
            losses = matting_loss(pred_fgr, pred_pha, true_fgr, masks)

            # 3. Extraer la pérdida total para el backward
            losses['total'] = losses['pha_l1'] + losses['fgr_l1']
            M_loss = losses['total']

            val = 0
            dices = 0
            B_loss = criterion(pred_pha, masks)
            pred_pha = torch.sigmoid(pred_pha)
            val = IoU(pred_pha,masks)
            dices = dice(pred_pha,masks).item()
            B_loss.backward()
            optimizer.step()

            total_Mloss_train += M_loss.item()
            total_Bloss_train += B_loss.item()
            total_iou_train   += val
            total_dice        += dices

        epoch_train_Mloss = total_Mloss_train/len(dataloader)
        epoch_train_Bloss = total_Bloss_train/len(dataloader)
        epoch_iou_train  = total_iou_train/len(dataloader)
        epoch_dice_train = total_dice/len(dataloader)

        end = time.time() #termina de contar el tiempo

        print(f"\nEpoch: {epoch}")
        print(f"Val Matting Loss: {total_Mloss_val:.4e}, Val BCELoss {total_Bloss_val:.4e}, Val IoU: {total_iou_val:.4e}, Val Dice: {total_dice_val:.4e}")
        print(f"Train Matting loss: {epoch_train_Mloss:.4e}, Train BCELoss {epoch_train_Bloss:.4e}, Train IoU: {epoch_iou_train:.4e}, Train Dice: {epoch_dice_train:.4e}")
        print(f"Tiempo de entrenamiento: {(end - start)/60:.3f} Min")

        #Guardar datos para la gráfica
        history['train_Mloss'].append(epoch_train_Mloss)
        history['train_loss'].append(epoch_train_Bloss)
        history['train_iou'].append(epoch_iou_train)
        history['train_dice'].append(epoch_dice_train)

        history['val_Mloss'].append(total_Mloss_val)
        history['val_loss'].append(total_Bloss_val)
        history['val_iou'].append(total_iou_val)
        history['val_dice'].append(total_dice_val)

        # Guardar progreso
        if best < total_dice_val:
          best = total_dice_val
          torch.save(model.state_dict(), f'{CHECKPOINT_PATH}/{Intento}_checkpoint.pth')
    #Mostrar gráfica
    plot_training_results( Intento,
      history['train_loss'], history['val_loss'],
      history['train_iou'], history['val_iou'],
      history['train_dice'], history['val_dice'],
      history['train_Mloss'], history['val_Mloss']
    )


'''
Ejemplo de como llamar la función de entrenamiento
train_model(Intento=4,
            Epocas=8,
            LR=0.000005,
            BATCH_SIZE=8,
            CHECKPOINT_PATH='/content/drive/MyDrive/Estancia_Profesional/checkpoint'
          )
'''
