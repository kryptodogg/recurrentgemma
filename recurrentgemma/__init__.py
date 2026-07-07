# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""RecurrentGemma public library.

Synesthesia fork change: JAX submodules (``complex_lib``, ``layers``,
``scan``) are imported lazily via ``__getattr__`` to avoid triggering
JAX initialization when only the torch path is used.  The upstream
``__init__.py`` eagerly imports JAX, which forces JAX to initialize
(and emit a ``No GPU/TPU found, falling back to CPU`` warning) even
for pure-PyTorch workloads like the gfx1031 dtype benchmark.

The JAX symbols remain accessible via direct submodule import
(``from recurrentgemma.jax import complex_lib``) or attribute access
(``recurrentgemma.complex_lib``) — the import just happens on first
access rather than at package-load time.

``common`` is pure Python (no JAX dependency) but is included in the
lazy table for consistency so that ``from recurrentgemma import common``
works through the same ``__getattr__`` path used by
``recurrentgemma/torch/__init__.py``.

Note: ``__version__`` is the only eager binding; the module appears
empty under ``help()`` / ``dir()`` until a lazy export is first accessed.
``__all__`` lists the lazy exports so ``from recurrentgemma import *``
matches the upstream behavior (and triggers all lazy loads).
"""

__version__ = "1.0.1"

# `__all__` lists the lazy exports so `from recurrentgemma import *` and
# `dir(recurrentgemma)` see them (matching the upstream behavior).  Note
# that `import *` will trigger all lazy loads including JAX init — the
# same trade-off as the upstream eager-import version.
__all__ = ("common", "complex_lib", "layers", "scan")

# Lazy export table: name -> fully-qualified module path.  Modules are
# imported on first attribute access, not at package-load time.
_LAZY_EXPORTS = {
    "common": "recurrentgemma.common",
    "complex_lib": "recurrentgemma.jax.complex_lib",
    "layers": "recurrentgemma.jax.layers",
    "scan": "recurrentgemma.jax.scan",
}

# Hoisted to module scope so `__getattr__` doesn't pay the import-lookup
# cost on every call (importlib is always available on Python 3+).
import importlib as _importlib  # noqa: E402


def __getattr__(name):
    """Lazy module attribute access — triggers JAX init only when needed."""
    if name in _LAZY_EXPORTS:
        module = _importlib.import_module(_LAZY_EXPORTS[name])
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
