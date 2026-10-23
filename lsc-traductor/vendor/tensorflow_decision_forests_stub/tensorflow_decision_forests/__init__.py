"""Reemplazo vacío ("stub") de tensorflow-decision-forests para Windows.

La librería real solo existe para Linux y macOS. Aun así, tensorflowjs hace
`import tensorflow_decision_forests` apenas arranca (la usa solo para convertir
modelos de bosques de decisión), y sin ella ni siquiera se puede importar
tensorflowjs en Windows.

Este módulo no contiene nada de la librería real: existe únicamente para que
ese `import` no falle. Si algún código intenta usar algo de ella, recibe un
error claro en lugar de un comportamiento extraño.
"""

__version__ = "1.12.0+stub"


def __getattr__(nombre):
    raise AttributeError(
        f"tensorflow_decision_forests es un reemplazo vacío para Windows y no tiene '{nombre}'. "
        "La librería real solo existe para Linux y macOS."
    )
