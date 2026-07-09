"""Pisces-style private RAG retrieval scaffolding.

This package starts with NssMPClib-native building blocks and deliberately keeps
the paper-specific protocols behind small interfaces so they can be replaced as
we port the full Pisces construction.
"""

from .config import PiscesConfig
from .pir import (
    ShareToHEAudit,
    ShareToHEResult,
    SudaBFVPolynomialBackend,
    SudaEncryptedOPROPEOPIBackend,
    SudaLFHEPolynomialBackend,
    SudaPIRToShareAudit,
    SudaPIRToShareResult,
    decrypt_bfv_ciphertext_to_tensor,
    SudaPIRToSharePlaintextProtocolBackend,
    SudaPolynomialPlaintextBackend,
    encode_database_as_polynomials,
    evaluate_polynomial_database,
    evaluate_polynomial_coefficients,
    shares_to_bfv_ciphertext,
    suda_ope_mask_polynomials,
    suda_opi_interpolate_share_polynomials,
    suda_opr_reduce_polynomials,
    suda_pir_to_share,
)
from .protocol1 import (
    Protocol1CandidateResult,
    Protocol1Client,
    Protocol1SemanticResult,
    Protocol1Server,
    protocol1_finish_from_candidate_mask,
)
from .protocol3 import (
    Protocol3Client,
    Protocol3ClientMessage,
    Protocol3Document,
    Protocol3PublicSetup,
    Protocol3Server,
)
from .protocol4 import (
    Protocol4Client,
    Protocol4PublicSetup,
    Protocol4Server,
    run_protocol4_client,
    run_protocol4_server,
)
from .psi import OKVSMultiInstanceLabeledPSI
from .retrieval import PiscesRetriever, RetrievalResult
from .secure_sorting import SecureTopKAudit, SecureTopKResult, secure_top_k_indicators

__all__ = [
    "OKVSMultiInstanceLabeledPSI",
    "ShareToHEAudit",
    "ShareToHEResult",
    "SudaBFVPolynomialBackend",
    "SudaEncryptedOPROPEOPIBackend",
    "SudaLFHEPolynomialBackend",
    "SudaPIRToShareAudit",
    "SudaPIRToShareResult",
    "SudaPIRToSharePlaintextProtocolBackend",
    "SudaPolynomialPlaintextBackend",
    "PiscesConfig",
    "PiscesRetriever",
    "RetrievalResult",
    "SecureTopKAudit",
    "SecureTopKResult",
    "Protocol1CandidateResult",
    "Protocol1Client",
    "Protocol1SemanticResult",
    "Protocol1Server",
    "Protocol4Client",
    "Protocol3Client",
    "Protocol3ClientMessage",
    "Protocol3Document",
    "Protocol3PublicSetup",
    "Protocol3Server",
    "Protocol4PublicSetup",
    "Protocol4Server",
    "run_protocol4_client",
    "run_protocol4_server",
    "protocol1_finish_from_candidate_mask",
    "secure_top_k_indicators",
    "decrypt_bfv_ciphertext_to_tensor",
    "encode_database_as_polynomials",
    "evaluate_polynomial_database",
    "evaluate_polynomial_coefficients",
    "shares_to_bfv_ciphertext",
    "suda_ope_mask_polynomials",
    "suda_opi_interpolate_share_polynomials",
    "suda_opr_reduce_polynomials",
    "suda_pir_to_share",
]
