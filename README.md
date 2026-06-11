# CNN — Reconocimiento de Emociones Faciales (FER-2013)

Red neuronal convolucional que clasifica expresiones faciales en 7 emociones:
`angry`, `disgust`, `fear`, `happy`, `sad`, `surprise`, `neutral`

---

## Estructura del proyecto

```
proyecto/
├── Dataset/               ← imágenes crudas (NO se suben al repo, ver paso 1)
│   ├── train/
│   │   ├── angry/
│   │   ├── disgust/
│   │   └── ...
│   └── test/
├── data/
│   └── processed/         ← archivos .npy generados por preprocess.py (NO se suben)
├── models/                ← modelo entrenado .keras (NO se sube)
├── results/               ← imágenes de resultados (NO se suben)
├── src/
│   ├── preprocess.py
│   ├── exploracion_dataset.py
│   ├── demo_simple_convolucion.py
│   └── demo_entrenamiento.py
├── .gitignore
└── README.md
```

---

## Pasos para ejecutar

### 1. Bajar el dataset

El dataset **no está incluido** en este repositorio por su tamaño.

Descárgalo desde Kaggle:
https://www.kaggle.com/datasets/msambare/fer2013?resource=download

Una vez descargado, descomprime y coloca la carpeta `Dataset/` en la raíz del proyecto con la estructura `Dataset/train/<clase>/` y `Dataset/test/<clase>/`.

### 2. Instalar dependencias

```bash
pip install tensorflow numpy opencv-python matplotlib scikit-learn seaborn
```

### 3. Preprocesamiento

Convierte las imágenes `.jpg` en arrays `.npy` normalizados (valores 0–1):

```bash
python src/preprocess.py
```

Genera en `data/processed/`:
- `X_train.npy` — imágenes de entrenamiento `(N, 48, 48, 1)` float32
- `y_train.npy` — etiquetas de entrenamiento
- `X_test.npy` / `y_test.npy` — imágenes y etiquetas de test
- `classes.npy` — nombres de las 7 clases

> Estos archivos son grandes (~300 MB en total) y están en `.gitignore`. Hay que regenerarlos localmente.

### 4. Exploración del dataset (opcional)

```bash
python src/exploracion_dataset.py
```

Genera en `results/exploración/`:
- Una imagen aleatoria del dataset
- Una imagen por clase
- Distribución de clases train vs test (útil para ver el desbalance)

### 5. Demo 1 — Convolución manual paso a paso

```bash
python src/demo_simple_convolucion.py
```

Muestra el pipeline completo de **una sola capa CNN** aplicado a una imagen:
`Imagen → Padding → Convolución → ReLU → MaxPooling → Flatten`

Cada paso se imprime en terminal con sus dimensiones y una explicación de por qué existe.
Genera visualizaciones en `results/demo1/`.

Para cambiar la imagen de ejemplo, edita la variable `IMAGE_INDEX` al inicio del script.

### 6. Demo 2 — Entrenamiento + Forward Pass

```bash
python src/demo_entrenamiento.py
```

Entrena la CNN completa y muestra el camino completo de una predicción:
`Imagen → Logits → Softmax → Predicción`

Para cambiar parámetros, edita las variables al inicio del script:

```python
EPOCHS      = 30     # épocas máximas
BATCH_SIZE  = 64
RETRAIN     = False  # True para reentrenar aunque ya exista el modelo
TEST_INDEX  = 0      # índice de la imagen de ejemplo en el forward pass
```

Genera en `results/entrenamiento/`:
- `training_curves.png` — loss y accuracy por época
- `confusion_matrix.png` — matriz de confusión sobre el test set completo
- `prediction.png` — forward pass visual: imagen → logits → probabilidades

---

## Manejo del desbalance de clases

El dataset FER-2013 está desbalanceado: `happy` tiene ~3× más imágenes que `disgust`.
Sin corrección, el modelo aprende a predecir siempre `happy` porque le da una accuracy alta.

Se aplican dos estrategias combinadas:

**`class_weight`** — penaliza más los errores en clases raras durante el entrenamiento.
El peso de cada clase es inversamente proporcional a su frecuencia:
`peso_i = total / (n_clases × count_i)`

**Data Augmentation** — genera variantes de cada imagen en tiempo real (rotaciones, desplazamientos, zoom, flip horizontal). Efectivamente "agranda" el dataset sin guardar imágenes nuevas en disco.

---

## Arquitectura de la CNN

```
Input (48, 48, 1)
  └─ Conv2D(32, 3×3) + BatchNorm + ReLU + MaxPool(2×2) + Dropout(0.25)  → (24, 24, 32)
  └─ Conv2D(64, 3×3) + BatchNorm + ReLU + MaxPool(2×2) + Dropout(0.25)  → (12, 12, 64)
  └─ Conv2D(128,3×3) + BatchNorm + ReLU + MaxPool(2×2) + Dropout(0.25)  → (6,  6, 128)
  └─ Flatten                                                              → 4608
  └─ Dense(256) + ReLU + Dropout(0.5)
  └─ Dense(7)   ← logits (sin activación)
       ↓
     Softmax  ← se aplica manualmente para mostrar logits vs probabilidades
```

**BatchNormalization** normaliza las activaciones de cada capa para que el entrenamiento sea más estable y rápido.

**Dropout** apaga neuronas al azar durante el entrenamiento para evitar sobreajuste (memorización).

---

## Notas importantes para el repo

- **No subir el dataset** (`Dataset/`) — son ~300 MB de imágenes, se baja desde Kaggle.
- **No subir los `.npy`** (`data/processed/`) — se regeneran con `preprocess.py`.
- **No subir el modelo** (`models/`) — se regenera con `demo_entrenamiento.py`.
- **No subir las imágenes de resultados** (`results/`) — se regeneran con los scripts.
- **No subir imágenes que sean ruido** — el `.gitignore` cubre `*.jpg`, `*.png` fuera de `src/`.

El `.gitignore` ya está configurado para ignorar todo lo anterior.
Solo se versiona el código fuente en `src/`.
