"""
Cryptographic primitives for PIR schemes.

This module defines interfaces for cryptographic primitives.
Concrete implementations are in scheme-specific modules.
"""

from .prf import PRFProtocol

__all__ = ["PRFProtocol"]
