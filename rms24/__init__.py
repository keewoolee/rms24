"""
RMS24: Simple and Practical Amortized Sublinear Private Information Retrieval

A Python implementation of the single-server PIR scheme from:
"Simple and Practical Amortized Sublinear Private Information Retrieval using Dummy Subsets"
by Ling Ren, Muhammad Haris Mughees, and I Sun (CCS 2024)

Reference: https://eprint.iacr.org/2023/1072
"""

from .params import Params
from .protocol import Query, Response, EntryUpdate
from .client import Client
from .server import Server
from .cuckoo import CuckooParams, CuckooTable
from .keyword_pir import KeywordParams, KeywordClient, KeywordServer

__version__ = "0.1.0"
__all__ = [
    "Params",
    "Query",
    "Response",
    "EntryUpdate",
    "Client",
    "Server",
    "CuckooParams",
    "CuckooTable",
    "KeywordParams",
    "KeywordClient",
    "KeywordServer",
]
