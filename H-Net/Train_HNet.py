#Ejecutar el modelo H-Net
!python /content/a_hnet_article.py
!python /content/hnet_skipy.py

import tensorflow as tf
import glob
import time
import json
from PIL import Image
import random
# Obtener el modelo 
from hnet_skipy import get_model   # o a_hnet_article

model = get_model()
model.summary()


def LR(epoch):
    if epoch < 40:
        return 0.0001
    elif epoch < 80:
        return 0.00001
    else:
        return 0.000001

BS = 16
Epocas = 10
path = "/content/drive/MyDrive/checkpoint/" # Dirección de la carpeta con los checkpoint

from tensorflow.keras.saving import register_keras_serializable
#Función para calcular la función de perdida
@register_keras_serializable()
def dice_bce_loss(y_true, y_pred, smooth=1e-6):
    alpha = 0.01
    gamma = 0.99

    # ── BCE ───────────────────────────────────────────────────
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)

    # ── Dice (DSC) ────────────────────────────────────────────
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    dsc = (2.0 * intersection + smooth) / (
          tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

    # ── Combinación del paper: L = α*H - γ*DSC ───────────────
    return alpha * bce - gamma * dsc

#función para calcular el coeficiente de DICE
@register_keras_serializable()
def dice_metric(y_true, y_pred, smooth=1e-6):
    y_true_f     = tf.keras.backend.flatten(y_true)
    y_pred_f     = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
            tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

#Compilar
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss= dice_bce_loss,
    metrics=[
        dice_metric,
        tf.keras.metrics.IoU(num_classes=2, target_class_ids=[0,1])
        ]
)


# Descongelar todas las capas
for layer in model.layers[:]:
    layer.trainable = True

model.compile(  # recompilar tras cambiar trainable
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),  # LR más bajo
    loss=  dice_bce_loss,
    metrics=[
        dice_metric,
        tf.keras.metrics.IoU(num_classes=2, target_class_ids=[0,1])
    ]
)



# Recolectar rutas y ordenarlas
image_paths = sorted(glob.glob("/content/dataset/train/fgr/*.jpg"))
mask_paths  = sorted(glob.glob("/content/dataset/train/pha/*.jpg"))
image_val = sorted(glob.glob("/content/dataset/val/fgr/*.jpg"))
mask_val  = sorted(glob.glob("/content/dataset/val/pha/*.jpg"))

# verificar que coinciden en cantidad y orden
print(f"Imágenes: {len(image_paths)}, Máscaras: {len(mask_paths)}")
print(image_paths[0], "↔", mask_paths[0])

#crear dataset
def load_data(image_path, mask_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=1)
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.image.resize(img, [256, 256])

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.cast(mask, tf.float32) / 255.0
    mask = tf.image.resize(mask, [256, 256])
    mask = tf.round(mask)

    return img, mask

#entrenamiento
dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
dataset = dataset.map(load_data).batch(BS).prefetch(tf.data.AUTOTUNE)
#validación
val_dataset = tf.data.Dataset.from_tensor_slices((image_val, mask_val))
val_dataset = val_dataset.map(load_data).batch(BS).prefetch(tf.data.AUTOTUNE)

#obtener el tiempo restante
class TiempoRestante(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        elapsed     = time.time() - self._epoch_start
        epochs_left = self.params['epochs'] - (epoch + 1)
        eta         = elapsed * epochs_left

        # formatear a hh:mm:ss
        def fmt(s):
            h, r = divmod(int(s), 3600)
            m, s = divmod(r, 60)
            return f"{h:02d}:{m:02d}:{s:02d}"

        print(f"  ⏱  Época {epoch+1}/{self.params['epochs']} "
              f"— duración: {fmt(elapsed)} "
              f"— tiempo restante estimado: {fmt(eta)}")

  # Guardar checkpoint completo (pesos + optimizer + época) 
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=f"{path}checkpoint_epoch_train.keras",
    save_weights_only=False,   # guarda todo, no solo pesos
    save_best_only=False,      # guardar en cada época
    verbose=1
)

# guardar solo guardar el mejor
checkpoint_best = tf.keras.callbacks.ModelCheckpoint(
    filepath=f"{path}best_model.keras",
    save_weights_only=False,
    save_best_only=True,
    monitor="val_loss",
    verbose=1
)
#guardar la información
def guardar_history(history, path, nombre="history.json"):
    historia_path = os.path.join(path, nombre)

    # cargar history previo si existe
    if os.path.exists(historia_path):
        with open(historia_path, "r") as f:
            historia_previa = json.load(f)
        # concatenar
        for key in history.history:
            if key in historia_previa:
                historia_previa[key].extend(history.history[key])
            else:
                historia_previa[key] = history.history[key]
        historia_combinada = historia_previa
    else:
        historia_combinada = history.history

    with open(historia_path, "w") as f:
        json.dump(historia_combinada, f)
    print(f"History guardado en {historia_path}")
    return historia_combinada

def cargar_history(path, nombre="history.json"):
    historia_path = os.path.join(path, nombre)
    if os.path.exists(historia_path):
        with open(historia_path, "r") as f:
            return json.load(f)
    print("No se encontró history previo")
    return {}


#entrenar la primera etapa
callbacks = [
    tf.keras.callbacks.LearningRateScheduler(LR, verbose=1),
    checkpoint_cb,
    checkpoint_best,
    TiempoRestante()
]

history = model.fit(
    dataset,
    validation_data=val_dataset,
    epochs=Epocas,
    callbacks=callbacks)

guardar_history(history, "/content/drive/MyDrive/historial") #dirección donde se almacena el historial
'''
# cargar el modelo completo (arquitectura + pesos + estado del optimizer)
model = tf.keras.models.load_model(
    f"{path}checkpoint_epoch_train.keras",
    custom_objects={
        "loss": dice_bce_loss,
        "dice_metric":   dice_metric
    }
)

# verificar en qué época quedó
print("LR actual:", model.optimizer.learning_rate.numpy())
historia_acumulada = cargar_history("/content/drive/MyDrive/historial/")'''

#entrenar desde la segunda etapa en adelante
'''
callbacks = [
    tf.keras.callbacks.LearningRateScheduler(LR, verbose=1),
    checkpoint_cb,
    checkpoint_best,
    TiempoRestante()
]
# continuar desde época 101 hasta terminar
history2 = model.fit(
    dataset,
    validation_data=val_dataset,
    epochs=100,               # época FINAL, no cantidad adicional 
    initial_epoch=70,        # <-- continúa desde aquí            
    callbacks=callbacks
)
guardar_history(history2, "/content/drive/MyDrive/historial/") #actualiza el historial
'''

#guardar la grafica
def plot_training(histories):
    if not isinstance(histories, list):
        histories = [histories]

    combined = {}
    for h in histories:
        for key, values in h.history.items():
            if key not in combined:
                combined[key] = []
            combined[key].extend(values)

    # longitud mínima solo entre métricas principales
    main_keys = [k for k in combined if k != 'learning_rate']
    n_epochs  = min(len(combined[k]) for k in main_keys)
    combined  = {k: v[:n_epochs] for k, v in combined.items()}
    epochs    = list(range(1, n_epochs + 1))

    print(f"Épocas totales: {n_epochs}")

    train_metrics = [k for k in combined if not k.startswith('val_')
                                         and k != 'learning_rate']
    has_lr  = 'learning_rate' in combined and len(combined['learning_rate']) == n_epochs
    n_plots = len(train_metrics) + (1 if has_lr else 0)

    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    axes = list(axes.flatten()) if n_plots > 1 else [axes]

    color_train = '#2196F3'
    color_val   = '#F44336'

    for i, metric_name in enumerate(train_metrics):
        ax      = axes[i]
        y_train = combined[metric_name]

        ax.plot(epochs, y_train,
                color=color_train, linewidth=2, label='Entrenamiento')

        val_key = f'val_{metric_name}'
        if val_key in combined:
            ax.plot(epochs, combined[val_key],
                    color=color_val, linewidth=2,
                    linestyle='--', label='Validación')

        if 'loss' in metric_name:
            best_epoch = y_train.index(min(y_train)) + 1
            best_val   = min(y_train)
        else:
            best_epoch = y_train.index(max(y_train)) + 1
            best_val   = max(y_train)

        ax.axvline(x=best_epoch, color='gray', linestyle=':', alpha=0.7)
        ax.annotate(f'mejor: {best_val:.4f}\népoca {best_epoch}',
                    xy=(best_epoch, best_val),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=9, color='gray')

        for stage_epoch, label in [(200, 'LR→1e-5'), (400, 'LR→1e-6')]:
            if stage_epoch < n_epochs:
                ax.axvline(x=stage_epoch, color='orange',
                           linestyle='-.', alpha=0.5)
                ax.text(stage_epoch + 2, min(y_train),
                        label, fontsize=8, color='orange')

        ax.set_title(metric_name.replace('_', ' ').title(), fontsize=13)
        ax.set_xlabel('Época')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # LR solo si tiene la longitud correcta
    if has_lr:
        ax_lr = axes[-1]
        ax_lr.semilogy(epochs, combined['learning_rate'],
                       color='green', linewidth=2)
        ax_lr.set_title('Learning Rate', fontsize=13)
        ax_lr.set_xlabel('Época')
        ax_lr.grid(True, alpha=0.3)

    plt.suptitle('Historial de entrenamiento H-Net', fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig('/content/drive/MyDrive/Graficas_train/training_history_bw2.png', dpi=150, bbox_inches='tight') ## almacenar la gráfica resultante
    plt.show()
    print(f"Gráfica guardada — {n_epochs} épocas totales")

class HistoryWrapper:
    def __init__(self, d):
        self.history = d

historia_acumulada = cargar_history("/content/drive/MyDrive/historial/")
plot_training(HistoryWrapper(historia_acumulada))

#realizar una inferencia
model2 = get_model()
model2.load_weights(f"{path}best_model.keras")  # los pesos entrenados

# Preprocesar una imagen
def preprocess(image_path):
    img = Image.open(image_path).convert("L")   # forzar escala de grises
    img = img.resize((256, 256))
    img = np.array(img, dtype=np.float32) / 255.0
    img = img[..., np.newaxis]                  # (256,256) → (256,256,1)
    img = np.expand_dims(img, axis=0)           # → (1,256,256,1) batch dim
    return img

# Inferencia
def predict_original_size(image_path, threshold=0.5):
    img_pil = Image.open(image_path).convert("L")
    W, H = img_pil.size        # (W, H) original
    print(f"Tamaño original: {W}x{H}")

    # inferencia a 256×256
    img_resized = img_pil.resize((256, 256))
    img_array   = np.array(img_resized, dtype=np.float32) / 255.0
    # normalización ImageNet
    #mean = np.array([0.485, 0.456, 0.406])
    #std  = np.array([0.229, 0.224, 0.225])
    #img_array = (img_array - mean) / std   # (256,256,3)

    img_array   = img_array[np.newaxis, ..., np.newaxis]  # (1,256,256,3) np.newaxis

    pred = model2.predict(img_array)[0, ..., 0]            # (256,256)

    # devolver al tamaño original
    from skimage.transform import resize as sk_resize

    prob_map_orig = sk_resize(
        pred,
        (256, 256),                  # skimage usa (H, W) no (W, H)
        order=1,                 # interpolación bilineal
        mode='reflect',
        anti_aliasing=True
    ).astype(np.float32)         # shape: (H, W)


    mask_orig = (prob_map_orig > threshold).astype(np.uint8)
    print(f"Prob map: {prob_map_orig.shape}")   # debe ser (H, W)
    print(f"Máscara : {mask_orig.shape}")       # debe ser (H, W)
    return prob_map_orig, mask_orig


# Visualizar
def visualize(image_path, mask_path, mask):
    img_original =  np.array(Image.open(image_path).convert("RGB").resize((256, 256)))
    mask_original = np.array(Image.open(mask_path).convert("L").resize((256, 256)))

    # verificar que coinciden
    print(f"Imagen original : {np.array(img_original).shape}")   # (H, W, 3)
    print(f"Mapa de prob    : {mask_original.shape}")             # (H, W)
    print(f"Máscara binaria : {mask.shape}")

    #img_display = denormalizar(img_original.numpy())
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(img_original)
    axes[0].set_title("Imagen original")

    axes[1].imshow(mask_original, cmap="gray")
    axes[1].set_title("Máscara real")

    axes[2].imshow(mask, cmap="gray")
    axes[2].set_title("Máscara binaria")

    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    #plt.savefig("resultado.png", dpi=150)
    plt.show()

idx = random.randint(0, len(image_val)-1)
image_path = image_val[idx]
mask_path  = mask_val[idx]

prob_map, mask = predict_original_size(image_path)
visualize(image_path, mask_path, mask)
