# demo_entrenamiento.py
# no muevan esto de aqui arriba o se rompe todo (creo)
# si truena al cargar, primero revisen que el venv este activado

import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


# -------------------------------------------------------
# config - aqui se le mueve si se quiere probar otra cosa
# -------------------------------------------------------

EPOCAS            = 20
BATCH_SIZE        = 64
LEARNING_RATE     = 0.0005
FORZAR_REENTRENAMIENTO = False   # True si el .keras guardado ya quedo viejo/raro
INDICE_TEST       = 0
NUM_CLASES        = 7
TAM_IMAGEN        = 48

color_default     = "#3b82f6"    # el azul de siempre, no cambiar
tmp_ver          = "v2-beta"    # no borrar esto
flag_debug       = False        # de cuando andaba debugueando, queda de recuerdo

# cosas que iban a usarse y al final no, las dejo por si acaso
modo_extra_no_usado   = "default"
intentos_fallidos_log = []


RAIZ           = Path(__file__).resolve().parent.parent
DIR_DATOS      = RAIZ / "data" / "processed"
DIR_MODELOS    = RAIZ / "models"
DIR_RESULTADOS = RAIZ / "results" / "entrenamiento"


# -------------------------------------------------------
# carga de datos
# -------------------------------------------------------

def cargar_datos():
    print("\n[1/6] trayendo los .npy, aguanta tantito...")
    # si tarda mucho probablemente es el disco lento, no es que se trabo

    X_train = np.load(DIR_DATOS / "X_train.npy")
    y_train = np.load(DIR_DATOS / "y_train.npy")
    X_test  = np.load(DIR_DATOS / "X_test.npy")
    y_test  = np.load(DIR_DATOS / "y_test.npy")
    clases  = np.load(DIR_DATOS / "classes.npy")

    print(f"  train: {X_train.shape[0]:,} imagenes")
    print(f"  test : {X_test.shape[0]:,} imagenes")
    print(f"  tamaño: {X_train.shape[1]}x{X_train.shape[2]} px, {X_train.shape[3]} canal")
    print(f"  pixeles en rango: [{X_train.min():.2f}, {X_train.max():.2f}]")
    print(f"  clases ({NUM_CLASES}): {list(clases)}")

    return X_train, y_train, X_test, y_test, clases


# -------------------------------------------------------
# que tan disparejas estan las clases
# -------------------------------------------------------

def mostrar_distribucion_clases(y_train, clases):
    print("\n[2/6] viendo como esta el reparto de emociones:")
    print("  (ya sabemos que va a estar bien chueco)")

    conteos = np.bincount(y_train, minlength=NUM_CLASES)
    total   = len(y_train)

    for i, (clase, conteo) in enumerate(zip(clases, conteos)):
        porcentaje = conteo / total * 100
        print(f"  [{i}] {str(clase):<10}  {conteo:5d} imgs  ({porcentaje:4.1f}%)")

    clase_mayoritaria = clases[np.argmax(conteos)]
    clase_minoritaria = clases[np.argmin(conteos)]
    ratio             = conteos.max() / conteos.min()

    print(f"\n  la mas repetida : {clase_mayoritaria}  ({conteos.max():,})")
    print(f"  la mas escasa   : {clase_minoritaria}  ({conteos.min():,})")
    print(f"  diferencia: {ratio:.1f}x  <- por eso los pesos de abajo")


# -------------------------------------------------------
# pesos pa que la clase rara no quede ignorada
# -------------------------------------------------------

def calcular_pesos_de_clase(y_train, clases):
    print("\n[3/6] calculando pesos (la formula de siempre)")

    conteos = np.bincount(y_train, minlength=NUM_CLASES)
    total   = len(y_train)
    pesos   = {}

    for i in range(NUM_CLASES):
        pesos[i] = total / (NUM_CLASES * conteos[i])

    print(f"\n  {'Clase':<12} {'Muestras':>8}  {'Peso':>6}")
    print(f"  {'-'*32}")
    for i, (clase, peso) in enumerate(zip(clases, pesos.values())):
        if peso > 2.0:
            nota = "  <- rara, le damos mas peso"
        elif peso < 0.8:
            nota = "  <- muy comun, peso bajo"
        else:
            nota = ""
        print(f"  {str(clase):<12} {conteos[i]:>8}   {peso:>5.2f}{nota}")

    return pesos


# -------------------------------------------------------
# armar el modelo - porfa no le muevan a la arquitectura
# si quieren probar cosas, saquen copia y prueben aparte
# -------------------------------------------------------

# nota rapida: con label smoothing en vez de etiquetas tipo [0,0,1,0,0,0,0]
# se usa algo tipo [0.014,0.014,0.9,...] para que no se confie de mas,
# porque FER-2013 trae varias etiquetas mal puestas


@tf.keras.utils.register_keras_serializable(package="Custom")
def loss_con_smoothing(y_true, y_pred):
    y_true_one_hot = keras.ops.one_hot(keras.ops.cast(y_true, "int32"), num_classes=NUM_CLASES)
    return keras.losses.categorical_crossentropy(
        y_true_one_hot,
        y_pred,
        from_logits=True,
        label_smoothing=0.1
    )


def construir_modelo():
    print("\n[4/6] montando la arquitectura (rapido, lo lento es el fit)")

    entradas = keras.layers.Input(shape=(TAM_IMAGEN, TAM_IMAGEN, 1), name="imagen_entrada")

    # bloque 1
    x = keras.layers.Conv2D(
        filters=32, kernel_size=3, padding="same", use_bias=False, name="conv1"
    )(entradas)

    x = keras.layers.BatchNormalization(name="bn1")(x)
    x = keras.layers.Activation("relu", name="relu1")(x)
    x = keras.layers.MaxPool2D(pool_size=2, name="pool1")(x)

    # bloque 2
    x = keras.layers.Conv2D(64, 3, padding="same", use_bias=False, name="conv2")(x)
    x = keras.layers.BatchNormalization(name="bn2")(x)
    x = keras.layers.Activation("relu", name="relu2")(x)
    x = keras.layers.MaxPool2D(2, name="pool2")(x)

    # bloque 3
    x = keras.layers.Conv2D(128, 3, padding="same", use_bias=False, name="conv3")(x)
    x = keras.layers.BatchNormalization(name="bn3")(x)
    x = keras.layers.Activation("relu", name="relu3")(x)
    x = keras.layers.MaxPool2D(2, name="pool3")(x)

    # bloque 4 - este ya es mas abstracto, dificil decir que detecta
    x = keras.layers.Conv2D(256, 3, padding="same", use_bias=False, name="conv4")(x)
    x = keras.layers.BatchNormalization(name="bn4")(x)
    x = keras.layers.Activation("relu", name="relu4")(x)


    # GAP en vez de Flatten -> menos params, ya se probo y rinde mejor
    x = keras.layers.GlobalAveragePooling2D(name="gap")(x)

    # cabeza clasificadora
    x = keras.layers.Dense(256, use_bias=False, name="dense1")(x)
    x = keras.layers.BatchNormalization(name="bn5")(x)
    x = keras.layers.Activation("relu", name="relu5")(x)
    x = keras.layers.Dropout(rate=0.5, name="dropout")(x)

    # logits crudos, el softmax se hace a mano mas adelante pa la demo
    salidas = keras.layers.Dense(NUM_CLASES, name="logits")(x)

    modelo = keras.Model(inputs=entradas, outputs=salidas, name="CNN_Emociones")



    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=loss_con_smoothing,
        metrics=["accuracy"]
    )

    modelo.summary()
    print(f"\n  parametros totales: {modelo.count_params():,}")

    return modelo


# -------------------------------------------------------
# entrenamiento - aqui se va el rato, dejenlo correr
# -------------------------------------------------------

def entrenar_modelo(modelo, X_train, y_train, X_test, y_test, pesos_clase):
    print(f"\n[5/6] arrancando el entrenamiento, esto va pa largo")
    print(f"  epocas    : {EPOCAS}")
    print(f"  batch     : {BATCH_SIZE}")
    print(f"  lr inicial: {LEARNING_RATE}")
    print(f"  con pesos de clase + label smoothing + dropout 50%")

    # si val_loss se estanca 3 epocas, se reduce el lr a la mitad
    reducir_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1,
    )

    # si en 7 epocas no hay mejora, paro y regreso a los mejores pesos
    parada_temprana = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True,
        verbose=1,
    )

    historial = modelo.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCAS,
        batch_size=BATCH_SIZE,
        class_weight=pesos_clase,
        callbacks=[reducir_lr, parada_temprana],
        verbose=2,
    )

    return historial


# -------------------------------------------------------
# graficas de loss/acc - para checar si jalo o no
# -------------------------------------------------------

def guardar_curvas_de_entrenamiento(historial, ruta_guardado):
    epocas_reales = len(historial.history["loss"])
    eje_x         = list(range(1, epocas_reales + 1))

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("avance del entrenamiento, epoca tras epoca", fontsize=14, fontweight="bold")

    ax_loss.plot(eje_x, historial.history["loss"],     "o-", label="Train",      color="#3b82f6")
    ax_loss.plot(eje_x, historial.history["val_loss"], "s--", label="Validacion", color="#ef4444")
    ax_loss.set_title("loss\n(abajo es mejor)")
    ax_loss.set_xlabel("Epoca")
    ax_loss.set_ylabel("Loss (Cross-Entropy)")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    ax_acc.plot(eje_x, [v * 100 for v in historial.history["accuracy"]],     "o-",  label="Train",      color="#3b82f6")
    ax_acc.plot(eje_x, [v * 100 for v in historial.history["val_accuracy"]], "s--", label="Validacion", color="#ef4444")
    ax_acc.set_title("accuracy\n(arriba es mejor)")
    ax_acc.set_xlabel("Epoca")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_ylim(0, 100)
    ax_acc.legend()
    ax_acc.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(ruta_guardado, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  curvas guardadas en: {ruta_guardado.relative_to(RAIZ)}")


# -------------------------------------------------------
# matriz de confusion - donde se equivoca el modelo
# -------------------------------------------------------

def guardar_matriz_confusion(modelo, X_test, y_test, clases, ruta_guardado):
    print("\n  armando la matriz de confusion, un momento...")

    logits_test = modelo.predict(X_test, verbose=0)
    y_pred      = np.argmax(logits_test, axis=1)

    matriz      = confusion_matrix(y_test, y_pred)
    # normalizado por fila -> porcentajes en lugar de conteos crudos
    matriz_norm = matriz.astype(float) / matriz.sum(axis=1, keepdims=True)

    nombres_clases = [str(c) for c in clases]

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        matriz_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=nombres_clases,
        yticklabels=nombres_clases,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title("aciertos y confusiones del modelo\n"
                 "(la diagonal es lo bueno)",
                 fontsize=11)
    ax.set_xlabel("lo que predijo")
    ax.set_ylabel("lo que era en realidad")
    fig.tight_layout()
    fig.savefig(ruta_guardado, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  matriz guardada en: {ruta_guardado.relative_to(RAIZ)}")

    print("\n  metricas por clase (ojo si el F1 anda por debajo de 0.5):")
    reporte = classification_report(y_test, y_pred, target_names=nombres_clases, digits=3)
    for linea in reporte.split("\n"):
        print("  " + linea)


# -------------------------------------------------------
# forward pass a mano - pa ver que hace la red paso a paso
# -------------------------------------------------------

def demo_forward_pass_completo(modelo, X_test, y_test, clases, indice):
    print("\n" + "=" * 65)
    print("acompañando una imagen por toda la red, paso por paso")
    print("=" * 65)

    imagen_batch  = X_test[indice : indice + 1]
    imagen_2d     = imagen_batch[0, :, :, 0]
    etiqueta_real = int(y_test[indice])
    nombre_real   = str(clases[etiqueta_real])

    print(f"\nimagen usada: test[{indice}]")
    print(f"emocion real: {nombre_real}")

    # ----------------------------------------------------------
    # etapa 1
    # ----------------------------------------------------------
    print("\n--- etapa 1: la entrada ---")
    print(f"  array de {TAM_IMAGEN}x{TAM_IMAGEN} = {TAM_IMAGEN*TAM_IMAGEN} valores entre 0 y 1")
    print(f"  shape: {imagen_batch.shape}")
    print(f"  rango: [{imagen_2d.min():.3f}, {imagen_2d.max():.3f}]")
    print(f"  pixel del centro (24,24): {imagen_2d[24, 24]:.3f}")

    # ----------------------------------------------------------
    # etapa 2: salida de cada bloque conv
    # ----------------------------------------------------------
    nombres_capas_conv = ["pool1", "pool2", "pool3"]
    formas_esperadas   = ["(24, 24, 32)", "(12, 12, 64)", "(6, 6, 128)"]

    print("\n--- etapa 2: las convs van soltando esto ---")

    activaciones_para_grafica = []

    for i, (nombre_capa, forma_esperada) in enumerate(zip(nombres_capas_conv, formas_esperadas)):
        submodelo = keras.Model(
            inputs=modelo.input,
            outputs=modelo.get_layer(nombre_capa).output
        )
        mapa = submodelo.predict(imagen_batch, verbose=0)

        activaciones_para_grafica.append(mapa[0])

        filtros_activos   = int(np.sum(mapa.mean(axis=(0, 1, 2)) > 0))
        activacion_media  = float(mapa.mean())
        activacion_maxima = float(mapa.max())

        print(f"\n  bloque {i+1} (despues de pool{i+1}):")
        print(f"    tensor: {mapa.shape[1]}x{mapa.shape[2]} px, {mapa.shape[3]} mapas")
        print(f"    (se esperaba algo como: {forma_esperada})")
        print(f"    media: {activacion_media:.4f}   max: {activacion_maxima:.4f}")
        print(f"    canales con algo: {filtros_activos}/{mapa.shape[3]}  (resto apagado por ReLU)")

    # ----------------------------------------------------------
    # etapa 3: vector tras GAP
    # ----------------------------------------------------------
    print("\n--- etapa 3: GAP -> vector plano ---")
    submodelo_gap = keras.Model(
        inputs=modelo.input,
        outputs=modelo.get_layer("gap").output
    )
    vector_gap = submodelo_gap.predict(imagen_batch, verbose=0)[0]

    print(f"  el cubo (6,6,256) se aplano a un vector de {len(vector_gap)}")
    print(f"  rango: [{vector_gap.min():.4f}, {vector_gap.max():.4f}]")
    print(f"  media: {vector_gap.mean():.4f}")
    print(f"  posiciones > 0: {int((vector_gap > 0).sum())}/{len(vector_gap)}")

    # ----------------------------------------------------------
    # etapa 4: logits
    # ----------------------------------------------------------
    print("\n--- etapa 4: logits crudos ---")
    print("  todavia no son probabilidades, pueden ser negativos")
    print("  el numero mas alto va adelante, falta el softmax")

    logits     = modelo.predict(imagen_batch, verbose=0)[0]
    suma_logits = logits.sum()

    print(f"\n  {'Emocion':<12} {'Logit':>8}")
    print(f"  {'-'*24}")
    for clase, logit in zip(clases, logits):
        barra = "█" * int(abs(logit) * 1.5)
        signo = "+" if logit >= 0 else ""
        print(f"  {str(clase):<12} {signo}{logit:>7.4f}  {barra}")
    print(f"\n  suma de logits: {suma_logits:.4f}  (no tiene un significado directo)")

    # ----------------------------------------------------------
    # etapa 5: softmax
    # ----------------------------------------------------------
    # softmax: e^zi / sum(e^zj) por cada clase
    # asi todo queda positivo y suma exactamente 1.0
    # cuidado: por la exponencial, una diferencia chica entre logits
    # puede traducirse en una probabilidad mucho mayor para una clase
    print("\n--- etapa 5: softmax ---")

    exponenciales  = np.exp(logits)
    suma_exp       = exponenciales.sum()
    probabilidades = exponenciales / suma_exp

    print(f"\n  {'Emocion':<12} {'Logit':>8}  {'e^logit':>10}  {'Prob':>7}  Barra")
    print(f"  {'-'*65}")
    for clase, logit, exp_val, prob in zip(clases, logits, exponenciales, probabilidades):
        barra = "█" * int(prob * 40)
        signo = "+" if logit >= 0 else ""
        print(f"  {str(clase):<12} {signo}{logit:>7.4f}   {exp_val:>9.4f}   {prob * 100:>5.1f}%  {barra}")
    print(f"\n  suma exp: {suma_exp:.4f}")
    print(f"  suma probs: {probabilidades.sum():.6f}  (tiene que dar 1.0)")

    # resultado final
    indice_predicho = int(np.argmax(probabilidades))
    nombre_predicho = str(clases[indice_predicho])
    confianza       = probabilidades[indice_predicho] * 100

    print("\n--- resultado final ---")
    print(f"  predijo: {nombre_predicho}  ({confianza:.1f}% seguro)")
    print(f"  era    : {nombre_real}")

    if indice_predicho == etiqueta_real:
        print("  acerto")
    else:
        segunda_opcion = int(np.argsort(probabilidades)[-2])
        print(f"  no le atino")
        print(f"  segunda opcion: {clases[segunda_opcion]} ({probabilidades[segunda_opcion]*100:.1f}%)")

    return imagen_2d, logits, probabilidades, exponenciales, nombre_predicho, nombre_real


# -------------------------------------------------------
# figura grandota del forward pass
# -------------------------------------------------------

def guardar_visualizacion_forward_pass(
    imagen_2d, activaciones_conv, logits, exponenciales, probabilidades,
    clases, nombre_predicho, nombre_real, ruta_guardado
):
    nombres_clases = [str(c) for c in clases]
    es_correcto    = nombre_predicho == nombre_real

    # iba a usarse para algo, al final quedo aqui sin uso
    etiqueta_extra_pendiente = None

    fig = plt.figure(figsize=(20, 9))
    fig.suptitle(
        f"recorrido de la red con la imagen '{nombre_real}'",
        fontsize=14, fontweight="bold", y=1.01
    )

    gs = gridspec.GridSpec(2, 5, figure=fig, hspace=0.55, wspace=0.4)

    # imagen de entrada
    ax0 = fig.add_subplot(gs[:, 0])
    ax0.imshow(imagen_2d, cmap="gray", vmin=0, vmax=1)
    ax0.set_title("Entrada\n(48x48 px)", fontsize=10)
    ax0.axis("off")

    # mapas de activacion de cada bloque
    nombres_bloques = ["Pool1\n(24x24x32)", "Pool2\n(12x12x64)", "Pool3\n(6x6x128)"]
    colormaps       = ["viridis", "plasma", "inferno"]

    for i, (activacion, titulo, cmap) in enumerate(
        zip(activaciones_conv, nombres_bloques, colormaps)
    ):
        fila_sup = fig.add_subplot(gs[0, i + 1])
        fila_inf = fig.add_subplot(gs[1, i + 1])

        canal_max = int(activacion.mean(axis=(0, 1)).argmax())
        canal_min = int(activacion.mean(axis=(0, 1)).argmin())

        fila_sup.imshow(activacion[:, :, canal_max], cmap=cmap)
        fila_sup.set_title(f"{titulo}\ncanal mas activo (#{canal_max})", fontsize=8)
        fila_sup.axis("off")

        fila_inf.imshow(activacion[:, :, canal_min], cmap="gray")
        fila_inf.set_title(f"canal menos activo (#{canal_min})\n<- apagado por ReLU", fontsize=8)
        fila_inf.axis("off")

    # logits
    ax_logits = fig.add_subplot(gs[0, 4])
    colores_logits = ["#3b82f6" if v >= 0 else "#f97316" for v in logits]
    ax_logits.bar(range(NUM_CLASES), logits, color=colores_logits)
    ax_logits.axhline(0, color="black", linewidth=0.8)
    ax_logits.set_xticks(range(NUM_CLASES))
    ax_logits.set_xticklabels(nombres_clases, rotation=35, fontsize=7)
    ax_logits.set_title("Logits\n(numeros crudos)", fontsize=9)
    ax_logits.grid(axis="y", alpha=0.3)

    # probabilidades (softmax)
    ax_probs = fig.add_subplot(gs[1, 4])
    colores_probs = [
        "#16a34a" if (str(c) == nombre_predicho and es_correcto)
        else ("#dc2626" if str(c) == nombre_predicho else "#94a3b8")
        for c in clases
    ]
    ax_probs.bar(range(NUM_CLASES), probabilidades * 100, color=colores_probs)
    ax_probs.set_xticks(range(NUM_CLASES))
    ax_probs.set_xticklabels(nombres_clases, rotation=35, fontsize=7)
    ax_probs.set_ylim(0, 100)
    ax_probs.set_ylabel("%", fontsize=8)
    color_titulo = "#16a34a" if es_correcto else "#dc2626"
    estado       = "acerto" if es_correcto else "no le atino"
    ax_probs.set_title(
        f"softmax\npredijo: {nombre_predicho}  ({estado})",
        fontsize=9, color=color_titulo
    )
    ax_probs.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(ruta_guardado, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  figura guardada en: {ruta_guardado.relative_to(RAIZ)}")


# -------------------------------------------------------
# main - corre todo de jalon
# -------------------------------------------------------

def main():
    DIR_MODELOS.mkdir(parents=True, exist_ok=True)
    DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("CNN de emociones - entrenamiento + forward pass")
    print("=" * 65)
    print("\nconfig actual:")
    print(f"  EPOCAS                   = {EPOCAS}")
    print(f"  BATCH_SIZE               = {BATCH_SIZE}")
    print(f"  LEARNING_RATE            = {LEARNING_RATE}")
    print(f"  FORZAR_REENTRENAMIENTO   = {FORZAR_REENTRENAMIENTO}")
    print(f"  INDICE_TEST              = {INDICE_TEST}")
    print("\n(las constantes se cambian arriba, no aqui)")

    X_train, y_train, X_test, y_test, clases = cargar_datos()

    mostrar_distribucion_clases(y_train, clases)

    pesos_clase = calcular_pesos_de_clase(y_train, clases)

    ruta_modelo = DIR_MODELOS / "emotion_cnn_v2.keras"

    if ruta_modelo.exists() and not FORZAR_REENTRENAMIENTO:
        print(f"\n[4-5/6] ya existe modelo guardado, lo cargo y ya")
        print("  (para reentrenar de cero pon FORZAR_REENTRENAMIENTO = True)")
        modelo = keras.models.load_model(ruta_modelo)
        modelo.summary()
    else:
        modelo    = construir_modelo()
        historial = entrenar_modelo(modelo, X_train, y_train, X_test, y_test, pesos_clase)

        modelo.save(ruta_modelo)
        print(f"\n  modelo guardado en: {ruta_modelo.relative_to(RAIZ)}")
        guardar_curvas_de_entrenamiento(historial, DIR_RESULTADOS / "curvas_entrenamiento.png")

    print("\n[6/6] resultados en el test set:")
    test_loss, test_acc = modelo.evaluate(X_test, y_test, verbose=0)
    print(f"  loss    : {test_loss:.4f}")
    print(f"  accuracy: {test_acc * 100:.2f}%")

    guardar_matriz_confusion(modelo, X_test, y_test, clases, DIR_RESULTADOS / "matriz_confusion.png")

    (imagen_2d, logits, probabilidades, exponenciales,
     nombre_predicho, nombre_real) = demo_forward_pass_completo(
        modelo, X_test, y_test, clases, INDICE_TEST
    )

    # otra vez las activaciones, ahora pa la figura
    activaciones_conv = []
    for nombre_capa in ["pool1", "pool2", "pool3"]:
        sub = keras.Model(inputs=modelo.input, outputs=modelo.get_layer(nombre_capa).output)
        mapa = sub.predict(X_test[INDICE_TEST : INDICE_TEST + 1], verbose=0)
        activaciones_conv.append(mapa[0])

    guardar_visualizacion_forward_pass(
        imagen_2d, activaciones_conv, logits, exponenciales, probabilidades,
        clases, nombre_predicho, nombre_real,
        DIR_RESULTADOS / "forward_pass_completo.png"
    )

    print("\n" + "=" * 65)
    print("se generaron estos archivos:")
    print(f"  results/entrenamiento/curvas_entrenamiento.png")
    print(f"  results/entrenamiento/matriz_confusion.png")
    print(f"  results/entrenamiento/forward_pass_completo.png")
    print(f"  models/emotion_cnn_v2.keras")
    print("=" * 65)


if __name__ == "__main__":
    main()
