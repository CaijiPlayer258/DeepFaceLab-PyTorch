"""Qt binding compatibility layer.

DeepFaceLab upstream uses PyQt5. On some environments (especially newer Python
versions), PyQt5 wheels may be unavailable while PySide6 is.

This module provides star-import friendly symbols compatible with the existing
codebase:

- Prefer PyQt5
- Fall back to PySide6

No UX changes; this is purely an import shim.
"""

# Prefer PyQt5 (upstream)
try:  # pragma: no cover
    from PyQt5.QtCore import *  # noqa: F401,F403
    from PyQt5.QtGui import *  # noqa: F401,F403
    from PyQt5.QtWidgets import *  # noqa: F401,F403

    QT_BINDING = "PyQt5"

except Exception:  # pragma: no cover
    from PySide6.QtCore import *  # type: ignore # noqa: F401,F403
    from PySide6.QtGui import *  # type: ignore # noqa: F401,F403
    from PySide6.QtWidgets import *  # type: ignore # noqa: F401,F403

    QT_BINDING = "PySide6"

    # Provide PyQt5-style aliases commonly used by the codebase.
    try:
        pyqtSignal
    except NameError:  # PySide6 exports Signal
        try:
            pyqtSignal = Signal  # type: ignore # noqa: N816
        except Exception:
            pass

    try:
        pyqtSlot
    except NameError:
        try:
            pyqtSlot = Slot  # type: ignore # noqa: N816
        except Exception:
            pass

    try:
        pyqtProperty
    except NameError:
        try:
            pyqtProperty = Property  # type: ignore # noqa: N816
        except Exception:
            pass
