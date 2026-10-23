# Traductor de Lengua de Señas Colombiana (LSC)

Trabajo de grado de Ingeniería de Sistemas (Universidad del Valle, Colombia): un traductor automático de Lengua de Señas Colombiana que corre en el navegador. Reconoce dos cosas:

1. **El alfabeto dactilológico** (posturas estáticas de la mano), con un modelo denso simple (MLP).
2. **Un vocabulario básico de palabras con movimiento**, con un modelo de secuencias (LSTM).

## Cómo funciona

```text
Video de la cámara
  → MediaPipe: encuentra la mano y sus 21 puntos (coordenadas numéricas)
  → Modelo de IA: MLP para letras, LSTM para palabras (entrenado en Python con TensorFlow)
  → Conversión a TensorFlow.js
  → Inferencia en el navegador, dentro de una app Next.js
```

Los modelos se entrenan con los **21 puntos de la mano** y no con las imágenes completas. Son 63 números por mano (21 puntos × x, y, z) en lugar de cientos de miles de píxeles. Los puntos describen solo la forma de la mano y descartan el fondo, la luz y el color de piel; por eso bastan pocos ejemplos, el entrenamiento corre en CPU y el modelo final es lo bastante pequeño para el navegador.

## Estado del proyecto

- [x] Entorno de desarrollo de Python listo y verificado (cámara + detección de hasta 2 manos).
- [ ] Extraer los puntos del dataset y entrenar el modelo del alfabeto.
- [ ] Grabar las letras que faltan en el dataset (g, h, j, s, z).
- [ ] Modelo de palabras con movimiento.
- [ ] Convertir los modelos a TensorFlow.js e integrarlos en la app web (hoy la app es la plantilla de Next.js).

## Requisitos

Probado únicamente en **Windows 10 (64 bits)**; los comandos son de PowerShell.

| Requisito | Versión | Notas |
|---|---|---|
| Python | 3.12 (probado con 3.12.10) | **No uses la 3.14**: TensorFlow 2.21 solo publica instaladores para Python 3.10 a 3.13. |
| Microsoft Visual C++ Redistributable 2015-2022 (x64) | 14.x | Lo exige TensorFlow en Windows. Sin él, `import tensorflow` falla con `Could not find the DLL(s) 'msvcp140.dll...'`. |
| Node.js | 20.9 o superior (probado con 24.15) | Para la app web (Next.js 16.4.2). |
| Git | reciente | Para clonar el repositorio. |
| Cámara web | cualquiera que Windows reconozca (UVC) | Probado con una UGREEN Camera 2K y con la cámara integrada de una laptop. |
| Internet | | Primera instalación (unos 600 MB de descargas) y descarga del modelo de manos (unos 8 MB). |
| Espacio en disco | unos 3 GB | ~2,3 GB el entorno de Python, ~0,4 GB `node_modules`, ~0,05 GB el dataset. |

Hardware con el que se desarrolló: AMD Ryzen 5 8600G (sin GPU dedicada: todo corre en CPU), 32 GB de RAM y SSD NVMe.

Si te falta Python 3.12 o el runtime de Visual C++, se instalan con `winget`:

```powershell
winget install --id Python.Python.3.12 --scope user
winget install --id "Microsoft.VCRedist.2015+.x64"    # pide permisos de administrador
```

### Librerías de Python (versiones exactas)

| Librería | Versión | Para qué sirve |
|---|---|---|
| TensorFlow | 2.21.0 | Entrenar los modelos (en Windows el paquete normal ya es solo CPU). |
| MediaPipe | 1.1.1 | Encontrar la mano y sus 21 puntos. |
| opencv-contrib-python | 4.14.0.94 | Cámara y ventanas (se importa como `cv2`). |
| NumPy | 2.5.4 | Cálculo con arreglos numéricos. |
| scikit-learn | 1.9.1 | Partir los datos y calcular métricas. |
| Matplotlib | 3.11.2 | Gráficas. |
| tensorflowjs | 4.22.0 | Convertir los modelos a TensorFlow.js. |

La lista completa (58 paquetes con versión exacta) está en [`lsc-traductor/requirements.txt`](lsc-traductor/requirements.txt).

## Instalación

### 1. Clonar el repositorio

```powershell
git clone https://github.com/TU_USUARIO/TU_REPOSITORIO.git
cd TU_REPOSITORIO
```

### 2. App web (Next.js)

```powershell
npm install
npm run dev        # abre http://localhost:3000
```

Otros comandos: `npm run build`, `npm start` y `npm run lint`. Stack: Next.js 16.4.2 (App Router), React 19.3.0, Tailwind CSS 4 y TypeScript 5.

### 3. Entorno de Python

Los comandos se ejecutan **dentro de `lsc-traductor/`** (el `requirements.txt` referencia por ruta relativa un paquete de `vendor/`):

```powershell
cd lsc-traductor
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install --no-deps -r requirements.txt
```

La instalación lleva `--no-deps` a propósito (ver [Notas técnicas](#notas-técnicas)): `requirements.txt` ya trae todas las dependencias con su versión exacta. Después, `pip check` debe mostrar **una sola línea**, que es esperada:

```text
tensorflowjs 4.22.0 requires flax, which is not installed.
```

Si PowerShell no deja ejecutar `Activate.ps1`, mira [Problemas frecuentes](#problemas-frecuentes).

### 4. Dataset del alfabeto (no incluido en el repositorio)

Las imágenes pesan y tienen su propia licencia, así que no se suben a git. Descárgalas desde Kaggle (requiere cuenta): [`oscarstep/dataset-lsc-modelo`](https://www.kaggle.com/datasets/oscarstep/dataset-lsc-modelo), basado en el trabajo de Ortiz-Farfán y Camargo-Mendoza (2020), publicado en la revista TecnoLógicas. Déjalas así:

```text
lsc-traductor/datasets/LSC/
├── a/    ← 200 imágenes .jpg (400×400) por letra
├── b/
└── ...   ← 21 letras: a b c d e f i k l m n o p q r t u v w x y  (faltan g, h, j, s, z)
```

## Verificar la instalación

Con el entorno activado y desde `lsc-traductor/`:

```powershell
python test_setup.py
```

El script comprueba que las 7 librerías cargan, abre la cámara, muestra el video en una ventana y dibuja encima los 21 puntos de cada mano en tiempo real. Se cierra con `q` (también con Esc o con la X de la ventana).

- La primera vez tarda unos 15-20 segundos en cargar TensorFlow y MediaPipe, y descarga el modelo de manos (~8 MB) a `lsc-traductor/modelos/`.
- Usa por defecto la cámara número 1 (la UGREEN del autor). Con una sola cámara usa `python test_setup.py --camara 0`, o cambia `CAMARA_POR_DEFECTO` dentro del script.
- MediaPipe imprime líneas `INFO`, `WARNING` y `W0000`: son avisos internos normales, no errores.

## Estructura del repositorio

```text
.
├── app/                    App web (Next.js, App Router)
├── public/                 Archivos estáticos de la app
├── AGENTS.md               Notas para asistentes de código (Next.js 16 cambió respecto a versiones anteriores)
├── package.json            Dependencias y comandos de la app web
└── lsc-traductor/          Parte de Python: modelos de IA
    ├── requirements.txt    Dependencias con versión exacta
    ├── test_setup.py       Prueba del entorno: cámara + 21 puntos de la mano
    ├── vendor/             Paquete vacío que reemplaza a tensorflow-decision-forests (ver Notas técnicas)
    ├── datasets/           (no versionado) imágenes del alfabeto
    ├── modelos/            (no versionado) modelo de manos de MediaPipe
    └── venv/               (no versionado) entorno virtual de Python
```

## Notas técnicas

Estas decisiones salen de problemas reales de compatibilidad; si cambias una versión, vuelve a probarlas.

1. **Python 3.12.** TensorFlow 2.21 solo trae instaladores hasta Python 3.13.
2. **`opencv-contrib-python` en lugar de `opencv-python`.** MediaPipe exige el primero y ambos instalan el mismo módulo `cv2`, así que no deben convivir. El paquete *contrib* incluye todo lo de `opencv-python`.
3. **`tensorflowjs` se instala con `--no-deps`.** La versión 4.22.0 declara dos dependencias que en Windows dan problemas y que este proyecto no usa:
   - `tensorflow-decision-forests`: no existe para Windows y se importa al arrancar. Por eso `lsc-traductor/vendor/tensorflow_decision_forests_stub/` es un paquete vacío con ese nombre y versión `1.12.0+stub`, solo para que `import tensorflowjs` no falle.
   - `flax`: arrastra `orbax-checkpoint`, cuyas carpetas de prueba superan el límite de 260 caracteres de rutas de Windows y rompen la instalación. Solo sirve para convertir modelos de JAX.
4. **`setuptools==80.9.0`.** `tensorflow-hub` todavía usa `pkg_resources`, que las versiones nuevas de setuptools ya no incluyen.
5. **MediaPipe 1.x ya no tiene `mp.solutions`.** Se usa la API *Tasks* (`HandLandmarker`) con el modelo `hand_landmarker.task`, que `test_setup.py` descarga solo. Es el mismo modelo que usa la versión de MediaPipe para el navegador (`@mediapipe/tasks-vision`), así que los puntos de entrenamiento y los de la app son del mismo tipo.
6. **Keras para los modelos que irán al navegador.** Define `TF_USE_LEGACY_KERAS=1` **antes** de importar TensorFlow:

   ```python
   import os
   os.environ["TF_USE_LEGACY_KERAS"] = "1"   # antes de importar tensorflow
   import tensorflow as tf
   ```

   Con Keras 3 (el valor por defecto) el `model.json` sale con `batch_shape` y TensorFlow.js 4.22 no lo carga (`An InputLayer should be passed either a batchInputShape or an inputShape`). Con Keras legacy (`tf_keras`) cargaron un MLP y un LSTM por las tres vías de conversión (verificado con `@tensorflow/tfjs` 4.22.0 en Node).
7. **Detector en imágenes sueltas.** Para fotos independientes usa `RunningMode.IMAGE`; el modo `VIDEO` asume que cada cuadro continúa al anterior. En una muestra de 15 imágenes del dataset (3 de cada una de las letras a, b, l, y, w), con los umbrales por defecto el detector encontró mano en 9 y falló en las 3 de la `a` y las 3 de la `l`. Es solo una muestra: mídelo bien antes de entrenar y prueba añadir margen alrededor de la imagen o bajar `min_hand_detection_confidence`.

## Problemas frecuentes

| Síntoma | Causa y solución |
|---|---|
| `ImportError: Could not find the DLL(s) 'msvcp140.dll or msvcp140_1.dll'` | Falta el Visual C++ Redistributable. Instálalo con el comando de `winget` de la sección Requisitos o desde <https://aka.ms/vs/17/release/vc_redist.x64.exe>. |
| `No matching distribution found for tensorflow` | Estás usando una versión de Python no soportada (por ejemplo la 3.14). Crea el entorno con `py -3.12 -m venv venv`. |
| `python` abre la Microsoft Store o dice "no se encontró Python" | Es el atajo de la Store. Activa el entorno virtual (`.\venv\Scripts\Activate.ps1`) o ejecuta `.\venv\Scripts\python.exe test_setup.py`. |
| `Activate.ps1 no se puede cargar porque la ejecución de scripts está deshabilitada` | Ejecuta una vez `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` (responde `S` o `Y` si pregunta) y vuelve a intentar. |
| `WinError 206: El nombre del archivo o la extensión es demasiado largo` al instalar un paquete nuevo | Límite de 260 caracteres de Windows. Opción: habilitar las rutas largas con `New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force` (PowerShell como administrador; conviene reiniciar). |
| `ModuleNotFoundError: No module named 'pkg_resources'` | Un `setuptools` demasiado nuevo. Ejecuta `pip install setuptools==80.9.0`. |
| `No pude abrir la cámara` | Cierra las apps que la usen (Zoom, Teams, el navegador). Revisa *Configuración > Privacidad > Cámara > Permitir que las aplicaciones de escritorio accedan a la cámara*. Prueba otro número con `--camara`. |
| `An InputLayer should be passed either a batchInputShape or an inputShape` en el navegador | El modelo se guardó con Keras 3. Define `TF_USE_LEGACY_KERAS=1` antes de importar TensorFlow (nota técnica 6). |
| `pip check` avisa de `flax` | Es esperado (nota técnica 3). |

## Créditos y referencias

- [MediaPipe](https://ai.google.dev/edge/mediapipe) (Google): detección de la mano y sus 21 puntos.
- [TensorFlow](https://www.tensorflow.org) y [TensorFlow.js](https://www.tensorflow.org/js): entrenamiento e inferencia en el navegador.
- [Next.js](https://nextjs.org): aplicación web.
- Dataset del alfabeto: [`oscarstep/dataset-lsc-modelo`](https://www.kaggle.com/datasets/oscarstep/dataset-lsc-modelo) en Kaggle, basado en Ortiz-Farfán y Camargo-Mendoza (2020), *TecnoLógicas*. Revisa su licencia antes de redistribuirlo.
- Para las palabras con movimiento se evalúan los datasets académicos LSC50, LSC54 y LSC70 (video + sensores inerciales + landmarks).
