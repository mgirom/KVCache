"""MSCC -- mid-stack context codec.

Read a document once, store the model's state of having read it at one depth, and
let every downstream consumer start near the top of the stack instead of at token 1.

This package is the enforced half: the frame container (format) and the conditions
that must hold for a frame to mean anything (guard). The measurement half lives in
auditor/.
"""
from .format import (FORMAT_VERSION, Frame, FrameError, FrameHeader,
                     codebook_fingerprint, model_fingerprint, read_frame,
                     read_header, write_frame)
from .guard import FrameRejected, GuardResult, check, require

__all__ = ["FORMAT_VERSION", "Frame", "FrameError", "FrameHeader", "FrameRejected",
           "GuardResult", "check", "codebook_fingerprint", "model_fingerprint",
           "read_frame", "read_header", "require", "write_frame"]
