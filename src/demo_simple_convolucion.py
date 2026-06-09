# -------------------------------------------------------
# Demostración de uso de un kernel: filtro vertical
# -------------------------------------------------------

# Librerías
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Datasets
REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RESULTS_DIR = REPO_ROOT / "results" / "demo1"

# Kernel: filtro vertical
KERNEL = np.array(
    [
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1],
    ],
    dtype=np.float32,
)
# Parámetros fijos
STRIDE = 1
PADDING = 1
POOL_SIZE = 2
POOL_STRIDE = 2

# Cargar imágen preprocesada
def load_image(index: int) -> tuple[np.ndarray, str]:
    X = np.load(PROCESSED_DIR / "X_train.npy")
    y = np.load(PROCESSED_DIR / "y_train.npy")
    classes = np.load(PROCESSED_DIR / "classes.npy")
    img = X[index, :, :, 0]
    label = str(classes[y[index]])
    return img, label


def pad_image(img: np.ndarray, padding: int) -> np.ndarray:
    """
    Agregar padding
    """

    print("\nAgregando Padding")
    print("----------------")
    print(f"Tamaño original: {img.shape}")
    print(f"Padding: {padding} pixel(s)")

    # Usando np.pad
    padded_img = np.pad(
        img,
        pad_width=padding,
        mode="constant",
        constant_values=0
    )

    print(f"Nuevo tamaño: {padded_img.shape}")

    return padded_img


def convolve2d(padded: np.ndarray, kernel: np.ndarray, stride: int) -> np.ndarray:
    kh, kw = kernel.shape
    in_h, in_w = padded.shape
    # Aplicar el stride (h= alto, w= ancho)
    out_h = (in_h - kh) // stride + 1
    out_w = (in_w - kw) // stride + 1
    output = np.zeros((out_h, out_w), dtype=np.float32)

    first_step_printed = False
    for row in range(out_h):
        for col in range(out_w):
            r0 = row * stride
            c0 = col * stride
            region = padded[r0 : r0 + kh, c0 : c0 + kw]
            value = float(np.sum(region * kernel))
            output[row, col] = value
            if not first_step_printed:
                region_int = (region * 255).round().astype(int)
                result_int = int(np.sum(region_int * kernel.astype(int)))
                print("\nPrimera convolución (esquina superior izq. 3x3):")
                print("Region (valores de cada pixel 0-255):")
                print(region_int)
                print("Kernel:")
                print(kernel.astype(int))
                print(f"Resultado (escala 0-255 ): {result_int}")
                print(f"Resultado (escala 0-1), usado por la  CNN): {value:.4f}")
                first_step_printed = True
    return output

# POOLING: usando Max Pool
def max_pool2d(fmap: np.ndarray, pool_size: int, stride: int) -> np.ndarray:
    # fmap = feature map
    # pool_size = tamaño del pool
    # stride = desplazamiento

    in_h, in_w = fmap.shape

    out_h = (in_h - pool_size) // stride + 1
    out_w = (in_w - pool_size) // stride + 1

    # Inicializar matriz de "resultado"
    output = np.zeros((out_h, out_w), dtype=np.float32)

    for row in range(out_h):
        for col in range(out_w):
            r0 = row * stride
            c0 = col * stride
            region = fmap[
                r0 : r0 + pool_size, 
                c0 : c0 + pool_size]
            # Del pool: obtiene el valor máximo
            output[row, col] = float(np.max(region))
    return output

# FLATTEN: aplanar el feature map (matriz 2D) a un vector 1D
# Convierte un tensor 2D (alto, ancho) en un vector 1D de tamaño alto*ancho.
# Es la entrada que recibe la primera capa Dense de la red.
def flatten(fmap: np.ndarray) -> np.ndarray:
    return fmap.flatten()

# VISUALIZACIÓN: Observa la imágen original
def save_original(img: np.ndarray, label: str, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"Imágen original\nclase: {label}   tamaño: {img.shape}")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

# ILUSTRACIÓN DEL KERNEL
# Función para ilustrar y mostrar el kernel utilizado
def save_kernel(kernel: np.ndarray, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(kernel, cmap="gray")
    ax.set_title("Filtro de la convolución\n(detector de ejes verticales)")
    ax.set_xticks(range(kernel.shape[1]))
    ax.set_yticks(range(kernel.shape[0]))
    for r in range(kernel.shape[0]):
        for c in range(kernel.shape[1]):
            val = int(kernel[r, c])
            color = "blue"
            ax.text(c, r, f"{val}", ha="center", va="center", color=color, fontsize=18, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

# PADDING: ilustración del padding utilizado
def save_padding(img: np.ndarray, padded: np.ndarray, save_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"Original\n{img.shape}")
    axes[0].axis("off")
    axes[1].imshow(padded, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"Con padding (borde de 0's))\n{padded.shape}")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

# FEATURE MAP: Plot del fmap generado
def save_feature_map(fmap: np.ndarray, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(fmap, cmap="gray")
    ax.set_title(f"Feature Map después de la convolución\ntamaño: {fmap.shape}")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

# POOLING: Ilustración del resultado
def save_pooled(pooled: np.ndarray, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(pooled, cmap="gray")
    ax.set_title(f"Feature Map + Pooling (2x2 max)\ntamaño: {pooled.shape}")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

# FLATTEN: Ilustración del aplanamiento (cubo -> vector)
# aplana un vector 1D de cubo a vector.
def save_flatten(pooled: np.ndarray, flat: np.ndarray, save_path: Path) -> None:

    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.4)

    # Arriba: feature map 2D
    ax_top = fig.add_subplot(gs[0])
    ax_top.imshow(pooled, cmap="gray", aspect="equal")
    ax_top.set_title(f"Feature Map (2D)   tamaño: {pooled.shape}   →   {pooled.size} valores")
    ax_top.axis("off")

    # Abajo: vector 1D (mismo contenido leído fila por fila)
    ax_bot = fig.add_subplot(gs[1])
    ax_bot.imshow(flat[np.newaxis, :], cmap="gray", aspect="auto")
    ax_bot.set_title(f"Vector aplanado (1D)   tamaño: {flat.shape}   →   {flat.size} neuronas")
    ax_bot.set_yticks([])
    ax_bot.set_xticks([0, flat.size // 4, flat.size // 2, 3 * flat.size // 4, flat.size - 1])

    fig.suptitle("Flatten: del cubo al vector", fontsize=13)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

# Plot de las imágenes combinadas
def save_combined(
    img: np.ndarray,
    padded: np.ndarray,
    fmap: np.ndarray,
    pooled: np.ndarray,
    flat: np.ndarray,
    label: str,
    save_path: Path,
) -> None:
    fig = plt.figure(figsize=(15, 6.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[3.5, 1], hspace=0.5)

    # Fila superior: 4 pasos espaciales
    panels_top = [
        (img, f"1. Original\n{img.shape}\nclase: {label}", {"vmin": 0, "vmax": 1}),
        (padded, f"2. Padded\n{padded.shape}", {"vmin": 0, "vmax": 1}),
        (fmap, f"3. Feature Map\n{fmap.shape}", {}),
        (pooled, f"4. Pooled\n{pooled.shape}", {}),
    ]
    for col, (data, title, kwargs) in enumerate(panels_top):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(data, cmap="gray", **kwargs)
        ax.set_title(title)
        ax.axis("off")

    # Fila inferior: flatten ocupando las 4 columnas
    ax_flat = fig.add_subplot(gs[1, :])
    ax_flat.imshow(flat[np.newaxis, :], cmap="gray", aspect="auto")
    ax_flat.set_title(f"5. Flatten   {flat.shape}   →   {flat.size} neuronas (entrada de la capa Dense)")
    ax_flat.set_yticks([])

    fig.suptitle(
        "CNN Demostración 1 con filtro detector de bordes verticales\nConvolución manual + padding + max pooling + flatten",
        fontsize=12,
    )
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# MAIN: Demostración del algoritmo en terminal
def main() -> None:
    parser = argparse.ArgumentParser(description="CNN Demonstración 1 con convolución de un único filtro.")
    parser.add_argument("--index", type=int, default=10, help="Index de X_train.")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("===================================")
    print("CNN DEMONSTRACIÓN 1")
    print("===================================")

    img, label = load_image(args.index)
    print(f"\nImágen seleccionada: {args.index}")
    print(f"Clase:    {label}")
    print(f"Tamaño original: {img.shape}")

    print("\n=========================")
    print("PARÁMETROS DE LA CONVOLUCIÓN")
    print("=========================")
    print("\nKernel:")
    print(KERNEL.astype(int))
    print(f"\nStride:  {STRIDE}")
    print(f"Padding: {PADDING}")
    print(f"Pooling: {POOL_SIZE}x{POOL_SIZE} Max Pooling (stride={POOL_STRIDE})")

    padded = pad_image(img, PADDING)
    print(f"\nTamaño de la imagen con Padding: {padded.shape}")

    print("\n--- APLICANDO CONVOLUCIÓN ---")
    fmap = convolve2d(padded, KERNEL, STRIDE)
    print(f"\nTamaño Feature Map: {fmap.shape}")
    print(f"Rango Feature Map: [{fmap.min():.3f}, {fmap.max():.3f}]")

    print("\n--- APLICANDO MAX POOLING ---")
    pooled = max_pool2d(fmap, POOL_SIZE, POOL_STRIDE)
    print(f"\nTamaño Pooled: {pooled.shape}")
    print(f"Rango Pooled: [{pooled.min():.3f}, {pooled.max():.3f}]")

    print("\n--- APLICANDO FLATTEN ---")
    flat = flatten(pooled)
    print(f"\nAntes de flatten (cubo 2D): {pooled.shape}  →  {pooled.size} valores")
    print(f"Después de flatten (vector 1D): {flat.shape}  →  {flat.size} neuronas")
    print(f"Primeros 8 valores: {np.array2string(flat[:8], precision=3, separator=', ')}")

    save_original(img, label, RESULTS_DIR / "original.png")
    save_kernel(KERNEL, RESULTS_DIR / "kernel.png")
    save_padding(img, padded, RESULTS_DIR / "padding.png")
    save_feature_map(fmap, RESULTS_DIR / "feature_map.png")
    save_pooled(pooled, RESULTS_DIR / "pooling.png")
    save_flatten(pooled, flat, RESULTS_DIR / "flatten.png")
    save_combined(img, padded, fmap, pooled, flat, label, RESULTS_DIR / "demo1.png")

    print("\nTodas las visualizaciones fueron guardadas:")
    for p in sorted(RESULTS_DIR.iterdir()):
        print(f"  {p.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
