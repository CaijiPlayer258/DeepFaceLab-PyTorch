try:  # pragma: no cover
	from .qtex import *
except Exception:
	# Allow importing submodules (e.g. qt_selftest) even when Qt bindings
	# are not installed in the current environment.
	pass

try:  # pragma: no cover
	from .QSubprocessor import *
	from .QXIconButton import *
except Exception:
	pass