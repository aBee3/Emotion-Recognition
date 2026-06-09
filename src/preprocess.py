"""
Preprocesamiento: carga las imágenes, las normaliza a binario, y las guarda 
como .npy

Archivos:
    X_train.npy  (N_train, 48, 48, 1) float32
    y_train.npy  (N_train,)           int64
    X_test.npy   (N_test,  48, 48, 1) float32
    y_test.npy   (N_test,)            int64
    classes.npy  (7,)                 nombre de las clases
"""

from pathlib import Path
import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "Dataset"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

# Clases (en orden del dataseet)
CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def load_split(split_dir, split_name):
    images = []
    labels = []
    for class_idx, cls in enumerate(CLASSES):
        files = sorted((split_dir / cls).glob("*.jpg"))
        print(f"  [{split_name}] cargando imágenes de la clase {class_idx} '{cls}' ({len(files)} )...")
        for path in files:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            images.append(img.astype(np.float32) / 255.0)
            labels.append(class_idx)
    X = np.array(images).reshape(-1, 48, 48, 1)
    y = np.array(labels, dtype=np.int64)
    return X, y


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Cargando set ENTRENAMIENTO...")
    X_train, y_train = load_split(DATASET_DIR / "train", "train")
    print(f"  -> X_train: {X_train.shape} float32  ({X_train.nbytes / 1e6:.1f} MB)")

    print("Cargando set TEST...")
    X_test, y_test = load_split(DATASET_DIR / "test", "test")
    print(f"  -> X_test:  {X_test.shape} float32  ({X_test.nbytes / 1e6:.1f} MB)")

    print("Guardando los archivos .npy en data/processed/ ...")
    np.save(PROCESSED_DIR / "X_train.npy", X_train)
    np.save(PROCESSED_DIR / "y_train.npy", y_train)
    np.save(PROCESSED_DIR / "X_test.npy", X_test)
    np.save(PROCESSED_DIR / "y_test.npy", y_test)
    np.save(PROCESSED_DIR / "classes.npy", np.array(CLASSES))

    print("\nLISTO !:")
    for p in sorted(PROCESSED_DIR.iterdir()):
        print(f"  {p.relative_to(REPO_ROOT)}  ({p.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
