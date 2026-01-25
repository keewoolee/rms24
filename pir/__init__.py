"""
PIR (Private Information Retrieval) library.

This package provides implementations of Piano-like PIR schemes
(client-dependent preprocessing model).

Modules:
- primitives: Cryptographic primitives (PRF)
- protocols: Protocol interfaces for Piano-like PIR schemes
- keyword_pir: Keyword PIR using cuckoo hashing
- rms24: RMS24 PIR scheme implementation
"""

from . import primitives
from . import protocols
from . import keyword_pir
from . import rms24

__all__ = [
    "primitives",
    "protocols",
    "keyword_pir",
    "rms24",
]
