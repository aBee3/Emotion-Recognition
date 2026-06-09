# -------------------------------------------------------
# Demostración 2: Entrenamiento + Forward Pass
# -------------------------------------------------------
#  - Aplica el pipeline (Conv → ReLU → Pool) x 3 + Dense
#    a TODAS las imágenes del dataset (no solo una).
#  - Entrena el modelo durante N épocas.
#  - Hace un forward pass de UNA imagen y muestra:
#       * los logits (salida cruda antes de softmax)
#       * las probabilidades (después de softmax)
# -------------------------------------------------------

import argparse
import os
from pathlib import Path

# Silenciar mensajes informativos de TensorFlow (deja warnings y errors).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
MODELS_DIR = REPO_ROOT / "models"
RESULTS_DIR = REPO_ROOT / "results" / "entrenamiento"

NUM_CLASSES = 7
IMG_SIZE = 48


# -------------------------------------------------------
# 1) Carga de datos preprocesados
# -------------------------------------------------------
def cargar_datos():
    X_train = np.load(PROCESSED_DIR / "X_train.npy")
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    X_test = np.load(PROCESSED_DIR / "X_test.npy")
    y_test = np.load(PROCESSED_DIR / "y_test.npy")
    classes = np.load(PROCESSED_DIR / "classes.npy")
    return X_train, y_train, X_test, y_test, classes


# -------------------------------------------------------
# 2) Definición de la arquitectura CNN 
# -------------------------------------------------------
def construir_modelo() -> keras.Model:
    """
    Arquitectura:
      Input(48, 48, 1)
        Conv2D(32) → ReLU → MaxPool   →  (24, 24, 32)
        Conv2D(64) → ReLU → MaxPool   →  (12, 12, 64)
        Conv2D(128) → ReLU → MaxPool  →  (6, 6, 128)
        Flatten                       →  4608
        Dense(128) → ReLU
        Dense(7)                      →  logits (sin activación)
    Softmax se aplica fuera del modelo, manualmente, para poder
    mostrar 'antes vs. después' en el forward pass demo.
    """
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
            keras.layers.Conv2D(32, 3, padding="same", activation="relu", name="conv1"),
            keras.layers.MaxPool2D(2, name="pool1"),
            keras.layers.Conv2D(64, 3, padding="same", activation="relu", name="conv2"),
            keras.layers.MaxPool2D(2, name="pool2"),
            keras.layers.Conv2D(128, 3, padding="same", activation="relu", name="conv3"),
            keras.layers.MaxPool2D(2, name="pool3"),
            keras.layers.Flatten(name="flatten"),
            keras.layers.Dense(128, activation="relu", name="dense1"),
            keras.layers.Dense(NUM_CLASSES, name="logits"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


# -------------------------------------------------------
# 3) Snapshot de los filtros de Conv1 (para futura demo
#    "antes vs. después de entrenamiento")
# -------------------------------------------------------
def guardar_pesos_conv1(model: keras.Model, save_path: Path) -> np.ndarray:
    pesos = model.get_layer("conv1").get_weights()[0]
    np.save(save_path, pesos)
    return pesos


# -------------------------------------------------------
# 4) Curvas de entrenamiento (loss y accuracy por época)
# -------------------------------------------------------
def plot_training_curves(history: keras.callbacks.History, save_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"], label="train", marker="o")
    axes[0].plot(history.history["val_loss"], label="val", marker="o")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("época")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(history.history["accuracy"], label="train", marker="o")
    axes[1].plot(history.history["val_accuracy"], label="val", marker="o")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("época")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------
# 5) Forward pass de UNA imagen + impresión en terminal
# -------------------------------------------------------
def forward_pass_demo(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    classes: np.ndarray,
    index: int,
):
    img = X_test[index : index + 1]
    true_idx = int(y_test[index])
    true_label = str(classes[true_idx])

    logits = model.predict(img, verbose=0)[0]
    probs = tf.nn.softmax(logits).numpy()
    pred_idx = int(np.argmax(probs))
    pred_label = str(classes[pred_idx])

    print(f"\nImagen seleccionada: test[{index}]")
    print(f"Clase verdadera:     {true_label}")

    print("\nLogits (salida CRUDA, antes de softmax):")
    for cls, l in zip(classes, logits):
        print(f"  {str(cls):<10}  {l:+.4f}")

    print("\nProbabilidades (DESPUÉS de softmax):")
    for cls, p in zip(classes, probs):
        bar = "#" * int(round(p * 40))
        marker = "  ← predicción" if str(cls) == pred_label else ""
        print(f"  {str(cls):<10}  {p * 100:5.1f}%   {bar}{marker}")

    print(f"\nSuma de probabilidades: {probs.sum():.4f}  (debe ser 1.0)")
    print(f"Predicción: {pred_label}  ({probs[pred_idx] * 100:.1f}% de confianza)")
    if pred_idx == true_idx:
        print("Predicción CORRECTA")
    else:
        print(f"Predicción INCORRECTA (verdadera: {true_label})")

    return img[0, :, :, 0], logits, probs, pred_label, true_label


# -------------------------------------------------------
# 6) Visualización del forward pass (imagen + barras)
# -------------------------------------------------------
def plot_prediction(
    img: np.ndarray,
    logits: np.ndarray,
    probs: np.ndarray,
    classes: np.ndarray,
    pred_label: str,
    true_label: str,
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"Entrada\nverdadera: {true_label}")
    axes[0].axis("off")

    class_names = [str(c) for c in classes]
    axes[1].bar(range(len(class_names)), logits, color="#3b82f6")
    axes[1].set_xticks(range(len(class_names)))
    axes[1].set_xticklabels(class_names, rotation=30)
    axes[1].set_title("Logits (antes de softmax)")
    axes[1].axhline(0, color="black", linewidth=0.6)
    axes[1].grid(axis="y", alpha=0.3)

    is_correct = pred_label == true_label
    bar_colors = [
        "#2dc653" if (str(c) == pred_label and is_correct)
        else ("#dc2626" if str(c) == pred_label else "#9ca3af")
        for c in classes
    ]
    axes[2].bar(range(len(class_names)), probs * 100, color=bar_colors)
    axes[2].set_xticks(range(len(class_names)))
    axes[2].set_xticklabels(class_names, rotation=30)
    axes[2].set_ylabel("Probabilidad (%)")
    axes[2].set_ylim(0, 100)
    axes[2].set_title(f"Softmax → Probabilidades\npredicción: {pred_label}")
    axes[2].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------
# 7) MAIN
# -------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Demo 2 — Entrenamiento + Forward Pass.")
    parser.add_argument("--epochs", type=int, default=5, help="Número de épocas de entrenamiento.")
    parser.add_argument("--batch-size", type=int, default=128, help="Tamaño de batch.")
    parser.add_argument("--retrain", action="store_true", help="Forzar reentrenamiento aunque exista el modelo guardado.")
    parser.add_argument("--test-index", type=int, default=0, help="Índice del test set para el forward pass demo.")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("===================================")
    print("DEMO 2: ENTRENAMIENTO + FORWARD PASS")
    print("===================================")

    print("\nCargando datos preprocesados...")
    X_train, y_train, X_test, y_test, classes = cargar_datos()
    print(f"  X_train: {X_train.shape}  (rango: [{X_train.min():.2f}, {X_train.max():.2f}])")
    print(f"  X_test:  {X_test.shape}")
    print(f"  Clases:  {[str(c) for c in classes]}")

    model_path = MODELS_DIR / "emotion_cnn.keras"

    if model_path.exists() and not args.retrain:
        print(f"\nModelo encontrado — cargando desde {model_path.relative_to(REPO_ROOT)}")
        print("(usar --retrain para volver a entrenar desde cero)")
        model = keras.models.load_model(model_path)
        model.summary()
    else:
        print("\nConstruyendo modelo...")
        model = construir_modelo()
        model.summary()

        print("\nGuardando pesos INICIALES de Conv1 (antes de entrenar)...")
        pesos_iniciales = guardar_pesos_conv1(model, MODELS_DIR / "conv1_filters_initial.npy")
        print(f"  shape: {pesos_iniciales.shape}  → 32 filtros 3x3")
        print(f"  ejemplo (filtro #0):\n{pesos_iniciales[:, :, 0, 0]}")

        print("\nCalculando class_weight (compensa el desbalance de clases)...")
        class_weights = compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=y_train)
        class_weight_dict = {i: float(w) for i, w in enumerate(class_weights)}
        for i, w in class_weight_dict.items():
            print(f"  {str(classes[i]):<10}  peso={w:.2f}")

        print(f"\nEntrenando por {args.epochs} épocas (batch={args.batch_size})...")
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            epochs=args.epochs,
            batch_size=args.batch_size,
            class_weight=class_weight_dict,
            verbose=2,
        )

        print("\nGuardando modelo y pesos entrenados...")
        model.save(model_path)
        pesos_finales = guardar_pesos_conv1(model, MODELS_DIR / "conv1_filters_trained.npy")
        print(f"  modelo:           {model_path.relative_to(REPO_ROOT)}")
        print(f"  filtros iniciales: models/conv1_filters_initial.npy")
        print(f"  filtros entrenados: models/conv1_filters_trained.npy")
        print(f"  ejemplo filtro #0 ENTRENADO:\n{pesos_finales[:, :, 0, 0]}")

        plot_training_curves(history, RESULTS_DIR / "training_curves.png")
        print(f"  curvas:           results/entrenamiento/training_curves.png")

    print("\nEvaluando en test set completo...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Loss:     {test_loss:.4f}")
    print(f"  Accuracy: {test_acc * 100:.2f}%")

    print("\n===================================")
    print("FORWARD PASS — UNA IMAGEN")
    print("===================================")
    img, logits, probs, pred_label, true_label = forward_pass_demo(
        model, X_test, y_test, classes, args.test_index
    )
    plot_prediction(img, logits, probs, classes, pred_label, true_label, RESULTS_DIR / "prediction.png")
    print(f"\nVisualización guardada en: results/entrenamiento/prediction.png")


if __name__ == "__main__":
    main()
