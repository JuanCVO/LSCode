r"""Prueba del entorno del Traductor LSC.

Qué hace, en orden:
  1. Confirma que las librerías del proyecto se importan sin error.
  2. Abre la cámara web con OpenCV y muestra el video en una ventana.
  3. Pasa cada imagen por MediaPipe (detector de manos) y dibuja encima los
     21 puntos de cada mano, en tiempo real.
  4. Se cierra con la tecla 'q' (también con Esc o con la X de la ventana).

Cómo correrlo (desde la carpeta lsc-traductor, con el entorno virtual activado):
    .\venv\Scripts\Activate.ps1
    python test_setup.py              # usa la cámara por defecto (la UGREEN)
    python test_setup.py --camara 0   # usa la cámara de la laptop
"""
import argparse
import importlib
import os
import sys
import time
import urllib.request
import warnings
from pathlib import Path

# --- Avisos conocidos e inofensivos: se silencian SOLO en este script de prueba ---
# Estas variables hay que definirlas ANTES de importar las librerías.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # TensorFlow: oculta mensajes informativos y avisos del motor
# Oculta el aviso de oneDNN. En esta prueba da igual, pero en los scripts de
# entrenamiento NO lo pongas: oneDNN acelera TensorFlow en la CPU.
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")   # OpenCV: solo muestra errores
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")  # abre la cámara más rápido en Windows
# Aviso de tensorflow-hub sobre una pieza vieja de setuptools (ya fijamos setuptools<81).
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*", category=UserWarning)

print("Cargando librerías (la primera vez tarda unos segundos)...", flush=True)

try:
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
except Exception as error:  # noqa: BLE001 - cualquier fallo de importación se explica igual
    sys.exit(
        f"\nNo pude importar una librería ({type(error).__name__}: {error}).\n"
        "¿Activaste el entorno virtual? Desde la carpeta lsc-traductor ejecuta:\n"
        "    .\\venv\\Scripts\\Activate.ps1\n"
        "y vuelve a correr:  python test_setup.py\n"
    )

TITULO_VENTANA = "Prueba de entorno - Traductor LSC (q para salir)"

# Número de la cámara que se usa si no pasas --camara. En este equipo, con
# DirectShow (el sistema de video de Windows), son:
#   0 = USB2.0 VGA UVC WebCam (la de la laptop)     1 = UGREEN Camera 2K
# Si desconectas las cámaras o las cambias de puerto USB, el orden puede cambiar:
# en ese caso usa --camara N o edita este número.
CAMARA_POR_DEFECTO = 1

# Modelo de MediaPipe que sabe encontrar manos (oficial de Google, ~8 MB).
# Se descarga solo la primera vez y queda guardado en la carpeta modelos/.
URL_MODELO = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
RUTA_MODELO = Path(__file__).resolve().parent / "modelos" / "hand_landmarker.task"

# Parejas de puntos que se unen con una línea para formar el "esqueleto" de la mano.
CONEXIONES = [(c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]

# Colores en formato BGR (así los pide OpenCV: azul, verde, rojo).
COLOR_LINEA = (255, 255, 255)
COLOR_PUNTO = (0, 0, 255)
COLOR_TEXTO = (0, 255, 0)

LIBRERIAS = [
    ("numpy", "numpy"),
    ("OpenCV", "cv2"),
    ("scikit-learn", "sklearn"),
    ("matplotlib", "matplotlib"),
    ("TensorFlow", "tensorflow"),
    ("MediaPipe", "mediapipe"),
    ("tensorflowjs", "tensorflowjs"),
]

AYUDA_CAMARA = """
  No pude abrir la cámara. Revisa lo siguiente:
    - Que ninguna otra aplicación la esté usando (Zoom, Teams, el navegador, la app Cámara).
    - Que Windows deje usarla: Configuración > Privacidad > Cámara >
      "Permitir que las aplicaciones de escritorio accedan a la cámara".
    - Si tienes más de una cámara, prueba con otro número:  python test_setup.py --camara 0
"""


def verificar_librerias():
    """Importa cada librería, muestra su versión y devuelve True si todas cargaron."""
    todo_bien = True
    for nombre, modulo in LIBRERIAS:
        try:
            cargada = importlib.import_module(modulo)
            print(f"  [OK]    {nombre:<13} {getattr(cargada, '__version__', '?')}")
            if modulo == "tensorflow":
                # Oculta los avisos de "función antigua" que tf_keras imprime al cargarse.
                cargada.get_logger().setLevel("ERROR")
                gpus = len(cargada.config.list_physical_devices("GPU"))
                print(f"          (usa la CPU; GPUs detectadas: {gpus})")
        except Exception as error:  # noqa: BLE001 - queremos mostrar cualquier fallo
            todo_bien = False
            print(f"  [FALLO] {nombre:<13} {type(error).__name__}: {error}")
    return todo_bien


def obtener_modelo():
    """Devuelve el contenido del modelo de manos (bytes); lo descarga la primera vez."""
    if not RUTA_MODELO.exists() or RUTA_MODELO.stat().st_size < 1_000_000:
        RUTA_MODELO.parent.mkdir(parents=True, exist_ok=True)
        print(f"  Descargando el modelo de manos de MediaPipe (unos 8 MB) en {RUTA_MODELO} ...")
        temporal = RUTA_MODELO.with_name(RUTA_MODELO.name + ".descarga")
        urllib.request.urlretrieve(URL_MODELO, temporal)
        temporal.replace(RUTA_MODELO)
    # Se pasa como bytes (no como ruta) para evitar problemas con tildes en la ruta de Windows.
    return RUTA_MODELO.read_bytes()


def crear_detector(modelo_bytes):
    """Crea el detector de manos.

    Modo VIDEO: después de encontrar la mano en un cuadro, la sigue en los
    siguientes en vez de buscarla desde cero cada vez (más rápido y más estable).
    """
    opciones = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=modelo_bytes),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )
    return vision.HandLandmarker.create_from_options(opciones)


def dibujar_mano(imagen, puntos):
    """Dibuja el esqueleto y los 21 puntos de UNA mano sobre la imagen (en el mismo arreglo).

    MediaPipe entrega cada punto como x, y entre 0 y 1: la proporción del ancho
    y del alto de la imagen. Para dibujar hay que convertirlos a píxeles.
    """
    alto, ancho = imagen.shape[:2]
    pixeles = [(int(p.x * ancho), int(p.y * alto)) for p in puntos]
    for inicio, fin in CONEXIONES:
        cv2.line(imagen, pixeles[inicio], pixeles[fin], COLOR_LINEA, 2)
    for centro in pixeles:
        cv2.circle(imagen, centro, 5, COLOR_PUNTO, -1)  # relleno
        cv2.circle(imagen, centro, 5, COLOR_LINEA, 1)   # borde


def procesar_cuadro(cuadro, detector, marca_ms):
    """Busca manos en un cuadro (BGR), dibuja sus puntos encima y devuelve cuántas vio.

    `marca_ms` es la hora del cuadro en milisegundos; tiene que crecer en cada llamada.
    """
    rgb = cv2.cvtColor(cuadro, cv2.COLOR_BGR2RGB)  # OpenCV trabaja en BGR; MediaPipe espera RGB
    imagen_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    resultado = detector.detect_for_video(imagen_mp, marca_ms)
    for puntos in resultado.hand_landmarks:
        dibujar_mano(cuadro, puntos)
    return len(resultado.hand_landmarks)


def abrir_camara(indice):
    """Abre la cámara probando dos "motores" de video de Windows; devuelve None si ninguno sirve."""
    motores = [("DirectShow", cv2.CAP_DSHOW), ("Media Foundation", cv2.CAP_MSMF)]
    for nombre, motor in motores:
        camara = cv2.VideoCapture(indice, motor)
        if camara.isOpened():
            for _ in range(15):  # las primeras lecturas pueden fallar mientras la cámara "despierta"
                hay_imagen, _cuadro = camara.read()
                if hay_imagen:
                    print(f"  Cámara {indice} abierta con {nombre}.")
                    return camara
                time.sleep(0.1)
        camara.release()
    return None


def ventana_cerrada():
    """True si la persona cerró la ventana con la X."""
    try:
        return cv2.getWindowProperty(TITULO_VENTANA, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def main():
    lector = argparse.ArgumentParser(description="Prueba del entorno del Traductor LSC")
    lector.add_argument("--camara", type=int, default=CAMARA_POR_DEFECTO,
                        help="número de la cámara (por defecto %(default)s: la UGREEN; 0 = la de la laptop)")
    argumentos = lector.parse_args()

    print("=== Prueba del entorno - Traductor LSC ===")
    en_venv = sys.prefix != sys.base_prefix
    print(f"Python {sys.version.split()[0]} | entorno virtual: {'sí' if en_venv else 'NO (actívalo)'}")

    print("\n1) Librerías")
    if not verificar_librerias():
        print("\nHay librerías que no cargan; no sigo con la cámara.")
        return 1

    print("\n2) Modelo de manos de MediaPipe")
    try:
        modelo = obtener_modelo()
    except OSError as error:
        print(f"  No pude descargar el modelo: {error}\n  Revisa tu conexión a internet y vuelve a intentar.")
        return 1
    # El motor interno de MediaPipe imprime por su cuenta líneas técnicas y no hay
    # forma oficial de apagarlas. No ocultamos la salida de errores a la fuerza
    # porque también taparía errores de verdad.
    print("  Nota: MediaPipe puede imprimir líneas que empiezan con INFO, WARNING o W0000.")
    print("        Son avisos internos normales; no son errores.")
    detector = crear_detector(modelo)
    print("  Detector de manos listo.")

    print("\n3) Cámara")
    camara = abrir_camara(argumentos.camara)
    if camara is None:
        detector.close()
        print(AYUDA_CAMARA)
        return 1

    print("\nSe abre una ventana con tu cámara. Muestra la mano: deberías ver 21 puntos siguiéndola.")
    print("Presiona 'q' para salir.")

    inicio = time.perf_counter()
    ultima_marca = -1
    hora_anterior = inicio
    fps = 0.0
    try:
        while True:
            hay_imagen, cuadro = camara.read()
            if not hay_imagen:
                print("La cámara dejó de entregar imagen; cierro el programa.")
                break
            cuadro = cv2.flip(cuadro, 1)  # efecto espejo: se siente natural al mover la mano

            # MediaPipe exige que la hora de cada cuadro sea mayor que la del anterior.
            marca_ms = max(int((time.perf_counter() - inicio) * 1000), ultima_marca + 1)
            ultima_marca = marca_ms
            manos = procesar_cuadro(cuadro, detector, marca_ms)

            ahora = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(ahora - hora_anterior, 1e-6)  # promedio suavizado
            hora_anterior = ahora
            cv2.putText(cuadro, f"Manos: {manos}   FPS: {fps:.0f}   [q] salir", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXTO, 2)

            cv2.imshow(TITULO_VENTANA, cuadro)
            tecla = cv2.waitKey(1) & 0xFF
            if tecla in (ord("q"), ord("Q"), 27) or ventana_cerrada():  # 27 = tecla Esc
                break
    except KeyboardInterrupt:
        print("\nInterrumpido con Ctrl+C.")
    finally:
        camara.release()
        detector.close()
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # deja que Windows termine de cerrar la ventana

    print("\nListo: la ventana se cerró correctamente.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
