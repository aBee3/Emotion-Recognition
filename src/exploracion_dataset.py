"""
Exploración Inicial del Dataset: reconocimiento de emociones

Genera 3 imágenes guardadas en results/exploration/:
    1. Una imagen random del dataset
    2. Una imagen de cada clase (7 emociones)
    3. Distribución de clases (emociones)
"""

from __future__ import annotations

# Librerías
import argparse
import random
from pathlib import Path

# Librerías para Ilustración de imágenes
import cv2
import matplotlib.pyplot as plt
import numpy as np

# DATASETS
REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "Dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"
RESULTS_DIR = REPO_ROOT / "results" / "exploración"

# Etiquetas de clase (en orden del dataset)
CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


# Lista las imagene
# Ordena y regresa las imagenes .jpg / .png en sus respectivas clases
def list_images(class_dir: Path) -> list[Path]:
    return sorted(p for p in class_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

# Imágenes por clase (para el plot)
def count_per_class(split_dir: Path) -> dict[str, int]:
    return {cls: len(list_images(split_dir / cls)) for cls in CLASSES}

# Carga una sóla imágen en escala de grises 2d
def load_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Error al leer la imágen: {path}")
    return img

# PLOT 1 MUESTRA
def plot_random_sample(split_dir: Path, rng: random.Random, save_path: Path, show: bool) -> Path:
    cls = rng.choice(CLASSES)
    img_path = rng.choice(list_images(split_dir / cls))
    img = load_gray(img_path)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.set_title(f"Muestra random emoción: {cls}\n{img.shape[0]}x{img.shape[1]} escala de grises\n{img_path.name}")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return save_path


# PLOT POR EMOCIONES
def plot_one_per_class(split_dir: Path, rng: random.Random, save_path: Path, show: bool) -> Path:
    fig, axes = plt.subplots(1, len(CLASSES), figsize=(2 * len(CLASSES), 2.6))
    for ax, cls in zip(axes, CLASSES):
        img_path = rng.choice(list_images(split_dir / cls))
        img = load_gray(img_path)
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(cls)
        ax.axis("off")
    fig.suptitle("Muestra random por clase", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return save_path

# PLOT DE DISTRIBUCIONES
# Grouped bar chart of image counts per class (train vs test).
def plot_class_distribution(
    train_counts: dict[str, int],
    test_counts: dict[str, int],
    save_path: Path,
    show: bool,
) -> Path:  
    classes = list(train_counts.keys())
    train_values = [train_counts[c] for c in classes]
    test_values = [test_counts[c] for c in classes]

    x = np.arange(len(classes))
    width = 0.4

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_train = ax.bar(x - width / 2, train_values, width, label="train", color="#3b82f6")
    bars_test = ax.bar(x + width / 2, test_values, width, label="test", color="#f97316")

    # Highlight the most-used emotion in the train split.
    top_idx = int(np.argmax(train_values))
    bars_train[top_idx].set_color("#1d4ed8")
    bars_train[top_idx].set_edgecolor("black")
    bars_train[top_idx].set_linewidth(1.2)

    for bars in (bars_train, bars_test):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel("Número de imágenes")
    ax.set_title(
        f"Clase más popular (entrenamiento): '{classes[top_idx]}' "
        f"({train_values[top_idx]} images)"
    )
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return save_path


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    seed = 42
    show = True

    rng = random.Random(seed)
    train_counts = count_per_class(TRAIN_DIR)
    test_counts = count_per_class(TEST_DIR)

    print("Imágenes en TRAIN por clase:")
    for cls, n in train_counts.items():
        print(f"  {cls:<10} {n}")
    print(f"  {'TOTAL':<10} {sum(train_counts.values())}")
    print("Imágenes en TEST por clase")
    for cls, n in test_counts.items():
        print(f"  {cls:<10} {n}")
    print(f"  {'TOTAL':<10} {sum(test_counts.values())}")

    p1 = plot_random_sample(TRAIN_DIR, rng, RESULTS_DIR / "01_random.png", show)
    p2 = plot_one_per_class(TRAIN_DIR, rng, RESULTS_DIR / "02_una_por_clase.png", show)
    p3 = plot_class_distribution(train_counts, test_counts, RESULTS_DIR / "03_distribucion_clases.png", show)

    print("\nIMAGENES GUARDADAS:")
    for p in (p1, p2, p3):
        print(f"  {p.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
