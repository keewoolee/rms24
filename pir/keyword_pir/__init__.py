"""
Keyword PIR (KPIR) using Cuckoo Hashing.

Generic PIR-to-KPIR conversion based on Section 5 of eprint 2019/1483
("Simple and Practical Amortized Sublinear Private Information Retrieval").

Uses cuckoo hashing to map sparse keywords to a dense index space,
enabling keyword-based lookups on top of any index-based PIR scheme.
"""

from .params import KPIRParams
from .client import KPIRClient
from .server import KPIRServer
from .cuckoo import CuckooParams, CuckooHash, CuckooTable

__all__ = [
    "KPIRParams",
    "KPIRClient",
    "KPIRServer",
    "CuckooParams",
    "CuckooHash",
    "CuckooTable",
]
