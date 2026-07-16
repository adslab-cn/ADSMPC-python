"""PIR-to-share boundary for the Pisces retrieval pipeline.

Pisces calls for Suda/Song et al. batch PIR-to-share. Plain integer-valued
payloads now use a Pyfhel/BFV implementation of Suda's OPR/OPE/OPI flow by
default. Secret-shared payloads keep an ASS fallback because Suda PIR starts
from a server-held plaintext database and a client encrypted query.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any

import torch
import numpy as np

from .ops import select_by_indicators

DEFAULT_PLAINTEXT_MODULUS = 2_147_483_647
SUDA_NATIVE_MODULUS = 1_337_006_139_375_617
SUDA_NATIVE_HALF_POLY_MOD_DEGREE = 4096
SUDA_NATIVE_SUPPORTED_BATCH_SIZES = (1, 4, 16, 64, 256, 1024, 2048, 4096)


@dataclass(frozen=True)
class SudaPIRToShareAudit:
    """Shape and implementation metadata for a PIR-to-share call."""

    batch_size: int
    database_size: int
    record_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    implementation: str = "suda-pir-to-share-ass-indicator-backend"
    paper_backend: str = "LFHE/BFV OPR/OPE/OPI"
    paper_backend_available: bool = False
    backend_gap: str = "uses ASS/local indicator selection instead of Suda LFHE polynomial PIR"
    polynomial_modulus: int | None = None
    polynomial_degree: int | None = None
    he_scheme: str | None = None
    share_conversion: str | None = None
    opr_reduced_degree: int | None = None
    ope_masked_degree: int | None = None
    opi_iota: int | None = None
    native_host_log_n_data: int | None = None
    native_padded_database_size: int | None = None
    native_batch_size: int | None = None
    native_query_bytes: int | None = None
    native_response_bytes: int | None = None


@dataclass(frozen=True)
class SudaPIRToShareResult:
    """Selected record shares plus audit metadata."""

    records: Any
    audit: SudaPIRToShareAudit
    server_share: Any | None = None
    client_share: Any | None = None


@dataclass
class SudaNativeClientState:
    client: Any
    real_batch_size: int
    feature_num: int
    output_shape: tuple[int, ...]
    modulus: int
    dtype: torch.dtype
    device: torch.device | str


@dataclass
class SudaNativeServerAnswer:
    server_share: torch.Tensor
    response_message: dict[str, Any]
    audit: SudaPIRToShareAudit


@dataclass
class SudaNativeClientAnswer:
    client_share: torch.Tensor
    records: torch.Tensor | None = None


@dataclass(frozen=True)
class ShareToHEAudit:
    shape: tuple[int, ...]
    modulus: int
    he_scheme: str = "BFV"
    implementation: str = "pyfhel-bfv-client-share-encryption"
    paper_step: str = "Pisces Protocol 1 Step 6 / Protocol 2 Step 7"


@dataclass(frozen=True)
class ShareToHEResult:
    ciphertext: Any
    he_context: Any
    audit: ShareToHEAudit


@dataclass(frozen=True)
class SudaPolynomialDatabase:
    """Finite-field polynomial encoding of a row-addressed database.

    The encoding is intentionally simple and auditable: for each flattened
    payload coordinate it interpolates one polynomial P such that
    ``P(row_id + 1) = database[row_id]`` over ``Z_modulus``.
    """

    coefficients: torch.Tensor
    original_shape: tuple[int, ...]
    record_shape: tuple[int, ...]
    modulus: int

    @property
    def database_size(self) -> int:
        return self.original_shape[0]

    @property
    def degree(self) -> int:
        return int(self.coefficients.shape[0]) - 1


class SudaPolynomialPlaintextBackend:
    """Plaintext polynomial PIR prototype.

    This is not private: it exists to validate the Suda-style database
    polynomialization and point-evaluation path before the same encoding is
    placed behind an LFHE/BFV layer.
    """

    def __init__(self, *, modulus: int = DEFAULT_PLAINTEXT_MODULUS):
        self.modulus = int(modulus)

    def retrieve(self, indicators: Any, database: Any) -> SudaPIRToShareResult:
        _validate_plain_tensor(indicators, "indicators")
        _validate_plain_tensor(database, "database")
        _validate_pir_shapes(indicators, database)
        encoding = encode_database_as_polynomials(database, modulus=self.modulus)
        query_points = _one_hot_query_points(indicators)
        records = evaluate_polynomial_database(encoding, query_points, dtype=database.dtype, device=database.device)
        audit = SudaPIRToShareAudit(
            batch_size=int(indicators.shape[0]),
            database_size=int(indicators.shape[1]),
            record_shape=tuple(int(dim) for dim in database.shape[1:]),
            output_shape=tuple(int(dim) for dim in records.shape),
            implementation="suda-polynomial-plaintext-backend",
            paper_backend="plaintext finite-field polynomial evaluation",
            paper_backend_available=False,
            backend_gap="validates polynomial database encoding but does not hide the query with LFHE/BFV",
            polynomial_modulus=encoding.modulus,
            polynomial_degree=encoding.degree,
        )
        return SudaPIRToShareResult(records=records, audit=audit)


class SudaPIRToSharePlaintextProtocolBackend:
    """Plaintext implementation of Suda Protocols 4/5/7.

    This backend follows the algebraic flow of Suda's PIR-to-share path:
    OPR reduces the database polynomials modulo the client query polynomial,
    OPI interpolates server-side random shares, and OPE masks the client-share
    polynomials before evaluation. It is a correctness/audit backend; the
    LFHE privacy layer is still represented by ``SudaBFVPolynomialBackend`` and
    the future ``SudaLFHEPolynomialBackend``.
    """

    def __init__(self, *, modulus: int = DEFAULT_PLAINTEXT_MODULUS, seed: int = 20260708):
        self.modulus = int(modulus)
        self.seed = int(seed)

    def retrieve(self, indicators: Any, database: Any) -> SudaPIRToShareResult:
        _validate_plain_tensor(indicators, "indicators")
        _validate_plain_tensor(database, "database")
        _validate_pir_shapes(indicators, database)
        encoding = encode_database_as_polynomials(database, modulus=self.modulus)
        query_points = _one_hot_query_points(indicators)
        reduced = suda_opr_reduce_polynomials(encoding.coefficients, query_points, modulus=encoding.modulus)
        selected = evaluate_polynomial_coefficients(reduced, query_points, encoding.modulus, encoding.record_shape)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)
        server_share_field = torch.randint(
            0,
            encoding.modulus,
            selected.shape,
            dtype=torch.long,
            generator=generator,
        )
        opi_polynomials, opi_iota = suda_opi_interpolate_share_polynomials(
            server_share_field,
            query_points,
            modulus=encoding.modulus,
        )
        client_polynomials = _poly_matrix_sub_mod(reduced, opi_polynomials, encoding.modulus)
        masked_client_polynomials, masked_degree = suda_ope_mask_polynomials(
            client_polynomials,
            query_points,
            modulus=encoding.modulus,
            seed=self.seed + 1,
        )
        client_share_field = evaluate_polynomial_coefficients(
            masked_client_polynomials,
            query_points,
            encoding.modulus,
            encoding.record_shape,
        )
        records_field = (server_share_field.reshape(client_share_field.shape) + client_share_field) % encoding.modulus

        records = _field_tensor_to_signed_float(records_field, encoding.modulus).to(dtype=database.dtype, device=database.device)
        server_share = _field_tensor_to_signed_float(server_share_field, encoding.modulus).to(dtype=database.dtype, device=database.device)
        client_share = _field_tensor_to_signed_float(client_share_field, encoding.modulus).to(dtype=database.dtype, device=database.device)
        audit = SudaPIRToShareAudit(
            batch_size=int(indicators.shape[0]),
            database_size=int(indicators.shape[1]),
            record_shape=tuple(int(dim) for dim in database.shape[1:]),
            output_shape=tuple(int(dim) for dim in records.shape),
            implementation="suda-plaintext-opr-ope-opi-backend",
            paper_backend="plaintext Suda OPR/OPE/OPI algebra",
            paper_backend_available=False,
            backend_gap="implements Suda Protocol 4/5/7 algebra without LFHE ciphertext transport",
            polynomial_modulus=encoding.modulus,
            polynomial_degree=encoding.degree,
            share_conversion="plaintext random additive shares over the PIR field",
            opr_reduced_degree=_poly_matrix_degree(reduced),
            ope_masked_degree=masked_degree,
            opi_iota=opi_iota,
        )
        return SudaPIRToShareResult(records=records, audit=audit, server_share=server_share, client_share=client_share)


class SudaBFVPolynomialBackend:
    """Experimental Pyfhel/BFV polynomial backend boundary.

    This backend is only selected explicitly. It uses the plaintext polynomial
    encoding above, then evaluates it through Pyfhel when that optional
    dependency is installed. It is a functional HE prototype, not the optimized
    Suda OPR/OPE/OPI implementation.
    """

    def __init__(
        self,
        *,
        modulus: int = 65_537,
        poly_modulus_degree: int = 16_384,
        plaintext_modulus_bits: int | None = None,
    ):
        self.modulus = int(modulus)
        self.poly_modulus_degree = int(poly_modulus_degree)
        self.plaintext_modulus_bits = None if plaintext_modulus_bits is None else int(plaintext_modulus_bits)

    def retrieve(self, indicators: Any, database: Any) -> SudaPIRToShareResult:
        try:
            from Pyfhel import Pyfhel
        except ImportError as exc:
            raise ImportError(
                "SudaBFVPolynomialBackend requires Pyfhel. Install it separately "
                "or use SudaPolynomialPlaintextBackend for encoding tests."
            ) from exc

        _validate_plain_tensor(indicators, "indicators")
        _validate_plain_tensor(database, "database")
        _validate_pir_shapes(indicators, database)
        encoding = encode_database_as_polynomials(database, modulus=self.modulus)
        query_points = _one_hot_query_points(indicators)

        he = Pyfhel()
        context_kwargs = {"scheme": "bfv", "n": self.poly_modulus_degree}
        if self.plaintext_modulus_bits is None:
            context_kwargs["t"] = encoding.modulus
        else:
            context_kwargs["t_bits"] = self.plaintext_modulus_bits
        he.contextGen(**context_kwargs)
        he.keyGen()

        rows = []
        for point in query_points.tolist():
            encrypted_point = _pyfhel_encrypt_int(he, int(point))
            encrypted_values = _evaluate_encrypted_polynomials_horner(he, encoding, encrypted_point)
            values = [_pyfhel_decrypt_int(he, value) % encoding.modulus for value in encrypted_values]
            rows.append(_field_values_to_tensor(values, encoding.modulus))

        flat = torch.stack(rows, dim=0).to(dtype=database.dtype, device=database.device)
        records = flat.reshape((len(rows),) + encoding.record_shape)
        audit = SudaPIRToShareAudit(
            batch_size=int(indicators.shape[0]),
            database_size=int(indicators.shape[1]),
            record_shape=tuple(int(dim) for dim in database.shape[1:]),
            output_shape=tuple(int(dim) for dim in records.shape),
            implementation="suda-bfv-polynomial-prototype",
            paper_backend="Pyfhel BFV polynomial evaluation",
            paper_backend_available=True,
            backend_gap="uses direct encrypted polynomial evaluation, not Suda's optimized OPR/OPE/OPI packing",
            polynomial_modulus=encoding.modulus,
            polynomial_degree=encoding.degree,
            he_scheme="BFV",
            share_conversion="client-decrypts prototype result; PIR-to-share conversion is not applied",
        )
        return SudaPIRToShareResult(records=records, audit=audit)


class SudaEncryptedOPROPEOPIBackend:
    """Coefficient-encrypted BFV implementation of Suda Protocols 4/5/7.

    This backend follows the Suda PIR-to-share message flow with encrypted
    polynomial messages. Pyfhel does not expose the same packed LFHE polynomial
    ciphertext interface used in the paper, so each polynomial coefficient is
    encrypted as an individual BFV ciphertext and polynomial products are
    computed by ciphertext convolutions.
    """

    def __init__(
        self,
        *,
        modulus: int = 65_537,
        poly_modulus_degree: int = 16_384,
        seed: int = 20260708,
    ):
        self.modulus = int(modulus)
        self.poly_modulus_degree = int(poly_modulus_degree)
        self.seed = int(seed)

    def retrieve(self, indicators: Any, database: Any) -> SudaPIRToShareResult:
        try:
            from Pyfhel import Pyfhel
        except ImportError as exc:
            raise ImportError("SudaEncryptedOPROPEOPIBackend requires Pyfhel") from exc

        _validate_plain_tensor(indicators, "indicators")
        _validate_plain_tensor(database, "database")
        _validate_pir_shapes(indicators, database)
        encoding = encode_database_as_polynomials(database, modulus=self.modulus)
        query_points = _one_hot_query_points(indicators)
        points = [int(point) % encoding.modulus for point in query_points.reshape(-1).tolist()]

        he = Pyfhel()
        he.contextGen(scheme="bfv", n=self.poly_modulus_degree, t=encoding.modulus)
        he.keyGen()
        he.relinKeyGen()

        encrypted_reduced = _encrypted_opr_reduce_polynomials(he, encoding.coefficients, points, encoding.modulus)
        selected = evaluate_polynomial_coefficients(
            suda_opr_reduce_polynomials(encoding.coefficients, query_points, modulus=encoding.modulus),
            query_points,
            encoding.modulus,
            encoding.record_shape,
        )

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)
        server_share_field = torch.randint(
            0,
            encoding.modulus,
            selected.shape,
            dtype=torch.long,
            generator=generator,
        )
        encrypted_server_share_polys, opi_iota = _encrypted_opi_interpolate_share_polynomials(
            he,
            server_share_field,
            points,
            encoding.modulus,
        )
        encrypted_client_polys = _encrypted_poly_matrix_sub(he, encrypted_reduced, encrypted_server_share_polys)
        encrypted_masked_client_polys, masked_degree = _encrypted_ope_mask_polynomials(
            he,
            encrypted_client_polys,
            points,
            encoding.modulus,
            seed=self.seed + 1,
        )
        client_share_field = _decrypt_poly_matrix_and_evaluate(
            he,
            encrypted_masked_client_polys,
            query_points,
            encoding.modulus,
            encoding.record_shape,
        )
        records_field = (server_share_field.reshape(client_share_field.shape) + client_share_field) % encoding.modulus
        records = _field_tensor_to_signed_float(records_field, encoding.modulus).to(dtype=database.dtype, device=database.device)
        server_share = _field_tensor_to_signed_float(server_share_field, encoding.modulus).to(dtype=database.dtype, device=database.device)
        client_share = _field_tensor_to_signed_float(client_share_field, encoding.modulus).to(dtype=database.dtype, device=database.device)
        audit = SudaPIRToShareAudit(
            batch_size=int(indicators.shape[0]),
            database_size=int(indicators.shape[1]),
            record_shape=tuple(int(dim) for dim in database.shape[1:]),
            output_shape=tuple(int(dim) for dim in records.shape),
            implementation="suda-bfv-encrypted-opr-ope-opi-backend",
            paper_backend="Pyfhel BFV coefficient-encrypted Suda OPR/OPE/OPI",
            paper_backend_available=True,
            backend_gap="uses coefficient-wise BFV ciphertexts instead of Suda's packed LFHE polynomial ciphertexts",
            polynomial_modulus=encoding.modulus,
            polynomial_degree=encoding.degree,
            he_scheme="BFV",
            share_conversion="server/client additive shares over the PIR field",
            opr_reduced_degree=_encrypted_poly_matrix_degree(encrypted_reduced),
            ope_masked_degree=masked_degree,
            opi_iota=opi_iota,
        )
        return SudaPIRToShareResult(records=records, audit=audit, server_share=server_share, client_share=client_share)


class SudaNativeBridgeBackend:
    """Adapter for the original Suda C++ batch PIR-to-share implementation.

    The native extension is intentionally optional. When available, it should
    expose ``batch_pir_to_share(feature_major, query_ids, **kwargs)`` and return
    server/client additive shares in Suda's finite field. This Python layer owns
    the stable Pisces-facing contract: one-hot indicators in, selected payload
    shares out.
    """

    def __init__(
        self,
        *,
        module_name: str = "NssMPC.application.rag.pisces._suda_bridge",
        module: Any | None = None,
        batch_size: int | None = None,
        modulus: int = SUDA_NATIVE_MODULUS,
        mod_switch: bool = True,
        use_out_mem: bool = False,
    ):
        self.module_name = module_name
        self.module = module
        self.batch_size = None if batch_size is None else int(batch_size)
        self.modulus = int(modulus)
        self.mod_switch = bool(mod_switch)
        self.use_out_mem = bool(use_out_mem)

    def retrieve(self, indicators: Any, database: Any) -> SudaPIRToShareResult:
        _validate_plain_tensor(indicators, "indicators")
        _validate_plain_tensor(database, "database")
        _validate_pir_shapes(indicators, database)
        row_ids = _one_hot_row_ids_zero_based(indicators)
        field_database = _tensor_to_field(database, self.modulus)
        database_size = int(field_database.shape[0])
        native_batch_size = _choose_suda_native_batch_size(len(row_ids), int(field_database.shape[1]), self.batch_size)
        padded_database_size = _suda_native_padded_database_size(database_size, native_batch_size, len(row_ids))
        host_log_n_data = _log2_exact(padded_database_size)

        padded_database = torch.zeros((padded_database_size, field_database.shape[1]), dtype=torch.long)
        padded_database[:database_size] = field_database
        padded_query_ids = _pad_suda_native_query_ids(row_ids, database_size, padded_database_size, native_batch_size)
        feature_major = padded_database.transpose(0, 1).contiguous().numpy().astype(np.int64)
        query_ids_np = np.asarray(padded_query_ids, dtype=np.int64)

        module = self.module if self.module is not None else _load_suda_native_bridge(self.module_name)
        native_result = module.batch_pir_to_share(
            feature_major,
            query_ids_np,
            database_size=database_size,
            host_log_n_data=host_log_n_data,
            batch_size=native_batch_size,
            mod_switch=self.mod_switch,
            use_out_mem=self.use_out_mem,
        )
        server_share_field, client_share_field, native_meta = _normalise_suda_native_output(
            native_result,
            real_batch_size=len(row_ids),
            feature_num=int(field_database.shape[1]),
            modulus=self.modulus,
        )
        records_field = (server_share_field + client_share_field) % self.modulus
        records = _field_tensor_to_signed_float(records_field, self.modulus).to(dtype=database.dtype, device=database.device)
        output_shape = (len(row_ids),) + tuple(int(dim) for dim in database.shape[1:])
        records = records.reshape(output_shape)
        server_share = server_share_field.reshape(output_shape).to(device=database.device)
        client_share = client_share_field.reshape(output_shape).to(device=database.device)

        audit = SudaPIRToShareAudit(
            batch_size=int(indicators.shape[0]),
            database_size=database_size,
            record_shape=tuple(int(dim) for dim in database.shape[1:]),
            output_shape=tuple(int(dim) for dim in records.shape),
            implementation="suda-native-cpp-batch-pir-to-share-bridge",
            paper_backend="Suda C++ BatchPirToShareServer/Client with SEAL polynomial ciphertexts",
            paper_backend_available=True,
            backend_gap="uses the original Suda C++ implementation through a Python bridge",
            polynomial_modulus=self.modulus,
            he_scheme="SEAL BGV/BFV-style batched polynomial HE",
            share_conversion="server/client additive shares kept as finite-field integers from Suda native batch PIR-to-share",
            native_host_log_n_data=host_log_n_data,
            native_padded_database_size=padded_database_size,
            native_batch_size=native_batch_size,
            native_query_bytes=_optional_int(native_meta.get("query_bytes")),
            native_response_bytes=_optional_int(native_meta.get("response_bytes")),
        )
        return SudaPIRToShareResult(records=records, audit=audit, server_share=server_share, client_share=client_share)


def suda_native_make_layout(
    *,
    database_size: int,
    record_shape: tuple[int, ...],
    selected_count: int,
    batch_size: int | None = None,
    modulus: int = SUDA_NATIVE_MODULUS,
) -> dict[str, Any]:
    feature_num = 1
    for dim in record_shape:
        feature_num *= int(dim)
    native_batch_size = _choose_suda_native_batch_size(selected_count, feature_num, batch_size)
    padded_database_size = _suda_native_padded_database_size(database_size, native_batch_size, selected_count)
    return {
        "database_size": int(database_size),
        "record_shape": tuple(int(dim) for dim in record_shape),
        "feature_num": int(feature_num),
        "selected_count": int(selected_count),
        "native_batch_size": int(native_batch_size),
        "padded_database_size": int(padded_database_size),
        "host_log_n_data": _log2_exact(padded_database_size),
        "modulus": int(modulus),
    }


def suda_native_make_client_request(
    row_ids: Any,
    layout: dict[str, Any],
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    module_name: str = "NssMPC.application.rag.pisces._suda_bridge",
    module: Any | None = None,
) -> tuple[SudaNativeClientState, dict[str, Any]]:
    module = module if module is not None else _load_suda_native_bridge(module_name)
    real_row_ids = [int(value) for value in torch.as_tensor(row_ids).reshape(-1).tolist()]
    if len(real_row_ids) != int(layout["selected_count"]):
        raise ValueError("row_ids length must match Suda native layout selected_count")
    padded_query_ids = _pad_suda_native_query_ids(
        real_row_ids,
        int(layout["database_size"]),
        int(layout["padded_database_size"]),
        int(layout["native_batch_size"]),
    )
    client = module.BatchPirToShareClientBridge(
        int(layout["host_log_n_data"]),
        int(layout["feature_num"]),
        int(layout["native_batch_size"]),
    )
    keys = client.save_keys()
    query = client.gen_query(np.asarray(padded_query_ids, dtype=np.int64))
    state = SudaNativeClientState(
        client=client,
        real_batch_size=int(layout["selected_count"]),
        feature_num=int(layout["feature_num"]),
        output_shape=(int(layout["selected_count"]),) + tuple(layout["record_shape"]),
        modulus=int(layout["modulus"]),
        dtype=dtype,
        device=device,
    )
    request = {
        "keys": keys,
        "query": query,
        "query_bytes": int(query["query_bytes"]),
        "layout": dict(layout),
    }
    return state, request


def suda_native_server_answer(
    database: Any,
    request: dict[str, Any],
    *,
    mod_switch: bool = True,
    use_out_mem: bool = False,
    module_name: str = "NssMPC.application.rag.pisces._suda_bridge",
    module: Any | None = None,
) -> SudaNativeServerAnswer:
    module = module if module is not None else _load_suda_native_bridge(module_name)
    layout = dict(request["layout"])
    database_tensor = database if isinstance(database, torch.Tensor) else torch.as_tensor(database)
    _validate_plain_tensor(database_tensor, "database")
    field_database = _tensor_to_field(database_tensor, int(layout["modulus"]))
    flat_database = field_database.reshape(field_database.shape[0], -1)
    if int(field_database.shape[0]) != int(layout["database_size"]):
        raise ValueError("database size does not match Suda native request layout")
    if int(flat_database.shape[1]) != int(layout["feature_num"]):
        raise ValueError("database feature count does not match Suda native request layout")

    padded_database = torch.zeros((int(layout["padded_database_size"]), int(layout["feature_num"])), dtype=torch.long)
    padded_database[: int(layout["database_size"])] = flat_database
    feature_major = padded_database.transpose(0, 1).contiguous().numpy().astype(np.int64)

    server = module.BatchPirToShareServerBridge(feature_major, int(layout["native_batch_size"]), bool(use_out_mem))
    server.load_keys(request["keys"])
    response = server.gen_response(request["query"], bool(mod_switch))
    server_share_field = torch.as_tensor(server.extract_answer(), dtype=torch.long).T
    server_share = server_share_field[: int(layout["selected_count"]), : int(layout["feature_num"])]
    output_shape = (int(layout["selected_count"]),) + tuple(layout["record_shape"])
    server_share = server_share.reshape(output_shape).to(device=database_tensor.device)
    audit = SudaPIRToShareAudit(
        batch_size=int(layout["selected_count"]),
        database_size=int(layout["database_size"]),
        record_shape=tuple(layout["record_shape"]),
        output_shape=output_shape,
        implementation="suda-native-cpp-split-batch-pir-to-share-bridge",
        paper_backend="Suda C++ BatchPirToShareServer/Client with SEAL polynomial ciphertexts",
        paper_backend_available=True,
        backend_gap="split client/server bridge: server receives encrypted Suda query, not plaintext top-k ids",
        polynomial_modulus=int(layout["modulus"]),
        he_scheme="SEAL BGV/BFV-style batched polynomial HE",
        share_conversion="server/client additive shares kept as finite-field integers from Suda native batch PIR-to-share",
        native_host_log_n_data=int(layout["host_log_n_data"]),
        native_padded_database_size=int(layout["padded_database_size"]),
        native_batch_size=int(layout["native_batch_size"]),
        native_query_bytes=_optional_int(request.get("query_bytes")),
        native_response_bytes=_optional_int(response.get("response_bytes")),
    )
    return SudaNativeServerAnswer(
        server_share=server_share,
        response_message={"response": response["response"], "layout": layout},
        audit=audit,
    )


def suda_native_client_extract(
    state: SudaNativeClientState,
    response_message: dict[str, Any],
    *,
    server_share: Any | None = None,
) -> SudaNativeClientAnswer:
    client_share_field = torch.as_tensor(state.client.extract_answer(response_message["response"]), dtype=torch.long).T
    client_share = client_share_field[: state.real_batch_size, : state.feature_num]
    client_share = client_share.reshape(state.output_shape).to(device=state.device)
    records = None
    if server_share is not None:
        records_field = (torch.as_tensor(server_share, dtype=torch.long).cpu() + client_share.cpu()) % state.modulus
        records = _field_tensor_to_signed_float(records_field, state.modulus).to(dtype=state.dtype, device=state.device)
    return SudaNativeClientAnswer(client_share=client_share, records=records)


class SudaLFHEPolynomialBackend:
    """Placeholder for the paper-level Suda LFHE/BFV backend.

    This backend is intentionally not selected by default. It isolates the
    missing OPR/OPE/OPI implementation without preventing the public
    ``suda_pir_to_share`` boundary from running with the available backend.
    """

    def retrieve(self, _indicators: Any, _database: Any) -> SudaPIRToShareResult:
        raise NotImplementedError("Suda LFHE/BFV OPR/OPE/OPI backend is not implemented")


def suda_pir_to_share(indicators: Any, database: Any, *, backend: Any | None = None) -> SudaPIRToShareResult:
    """Retrieve selected database rows as shares.

    The default backend first tries the native Suda C++ BatchPirToShare bridge
    for plain integer-valued tensors. If the native extension is not built, it
    falls back to the executable Pyfhel/plaintext Suda algebra. Secret-shared
    inputs keep the ASS fallback because the Suda HE flow starts from a
    server-held plaintext database and a client encrypted query.

    ``indicators`` has shape ``[batch_size, database_size]`` and can be either
    plain or arithmetic-secret-shared. ``database`` has shape
    ``[database_size, *record_shape]``. The output has shape
    ``[batch_size, *record_shape]`` and preserves the input representation.
    """

    if backend is not None:
        return backend.retrieve(indicators, database)
    if _can_use_default_suda_he_backend(indicators, database):
        try:
            return SudaNativeBridgeBackend().retrieve(indicators, database)
        except ImportError:
            pass
        try:
            return SudaEncryptedOPROPEOPIBackend().retrieve(indicators, database)
        except ImportError:
            return SudaPIRToSharePlaintextProtocolBackend().retrieve(indicators, database)
    return _ass_indicator_pir_to_share(indicators, database)


def suda_result_to_plain_additive_shares(result: SudaPIRToShareResult) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a Suda PIR result into ordinary additive tensor shares.

    Native Suda returns server/client shares over its finite field. NssMPClib's
    secure neural-network path expects ordinary fixed-point additive shares, so
    the demo boundary keeps the server's centered field share and adjusts the
    client share so the two tensors sum exactly to ``result.records``.
    """

    if result.server_share is None or result.client_share is None:
        raise ValueError("Suda result does not contain both server and client shares")
    if result.audit.implementation == "suda-native-cpp-batch-pir-to-share-bridge":
        modulus = result.audit.polynomial_modulus
        if modulus is None:
            raise ValueError("native Suda result must report polynomial_modulus")
        server_field = torch.as_tensor(result.server_share, dtype=torch.long)
        # Keep the MPC handoff numerically small. The native Suda field shares
        # are ~1e15; feeding them directly into fixed-point neural inference
        # would overflow or lose low bits. This preserves correctness of the
        # downstream additive sharing while using the Suda share as deterministic
        # entropy for the conversion.
        server = ((server_field % 2001) - 1000).to(dtype=torch.float32)
        records = torch.as_tensor(result.records, dtype=torch.float32)
        client = records - server
        return server, client
    server = torch.as_tensor(result.server_share)
    client = torch.as_tensor(result.client_share)
    return server, client


def shares_to_bfv_ciphertext(
    server_share: torch.Tensor,
    client_share: torch.Tensor,
    *,
    modulus: int = 65_537,
    poly_modulus_degree: int = 16_384,
) -> ShareToHEResult:
    """Convert additive plaintext shares into one BFV ciphertext of their sum.

    This models Pisces' optional handoff to HE-based secure inference: the
    client encrypts its share, sends it to the server, and the server adds its
    share homomorphically to obtain ``Enc(server_share + client_share)``.
    """

    try:
        from Pyfhel import Pyfhel
    except ImportError as exc:
        raise ImportError("shares_to_bfv_ciphertext requires Pyfhel") from exc
    _validate_plain_tensor(server_share, "server_share")
    _validate_plain_tensor(client_share, "client_share")
    if tuple(server_share.shape) != tuple(client_share.shape):
        raise ValueError("server_share and client_share must have the same shape")

    he = Pyfhel()
    he.contextGen(scheme="bfv", n=int(poly_modulus_degree), t=int(modulus))
    he.keyGen()
    client_values = _tensor_to_centered_int_array(client_share, modulus)
    server_values = _tensor_to_centered_int_array(server_share, modulus)
    ciphertext = he.encryptInt(client_values)
    ciphertext = ciphertext + server_values
    return ShareToHEResult(
        ciphertext=ciphertext,
        he_context=he,
        audit=ShareToHEAudit(shape=tuple(int(dim) for dim in server_share.shape), modulus=int(modulus)),
    )


def decrypt_bfv_ciphertext_to_tensor(result: ShareToHEResult, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    values = result.he_context.decryptInt(result.ciphertext)[: int(np.prod(result.audit.shape))]
    return torch.tensor(values, dtype=dtype).reshape(result.audit.shape)


def _ass_indicator_pir_to_share(indicators: Any, database: Any) -> SudaPIRToShareResult:
    _validate_pir_shapes(indicators, database)

    records = select_by_indicators(indicators, database)
    audit = SudaPIRToShareAudit(
        batch_size=int(indicators.shape[0]),
        database_size=int(indicators.shape[1]),
        record_shape=tuple(int(dim) for dim in database.shape[1:]),
        output_shape=tuple(int(dim) for dim in records.shape),
    )
    return SudaPIRToShareResult(records=records, audit=audit)


def encode_database_as_polynomials(database: torch.Tensor, *, modulus: int = DEFAULT_PLAINTEXT_MODULUS) -> SudaPolynomialDatabase:
    """Interpolate database rows into finite-field payload polynomials."""

    _validate_plain_tensor(database, "database")
    if len(database.shape) < 1:
        raise ValueError("database must have shape [database_size, ...]")
    if modulus <= database.shape[0] + 1:
        raise ValueError("modulus must be larger than the number of database rows")
    flat_values = _tensor_to_field(database, modulus)
    database_size = int(database.shape[0])
    flat_width = int(flat_values.shape[1])
    xs = list(range(1, database_size + 1))
    coefficients = torch.zeros((database_size, flat_width), dtype=torch.long)

    for row, x_value in enumerate(xs):
        basis = [1]
        denominator = 1
        for other in xs:
            if other == x_value:
                continue
            basis = _poly_mul_linear_mod(basis, (-other) % modulus, modulus)
            denominator = (denominator * ((x_value - other) % modulus)) % modulus
        scale = pow(denominator, -1, modulus)
        row_values = flat_values[row]
        for degree, basis_coefficient in enumerate(basis):
            term = (row_values * ((basis_coefficient * scale) % modulus)) % modulus
            coefficients[degree] = (coefficients[degree] + term) % modulus

    return SudaPolynomialDatabase(
        coefficients=coefficients,
        original_shape=tuple(int(dim) for dim in database.shape),
        record_shape=tuple(int(dim) for dim in database.shape[1:]),
        modulus=modulus,
    )


def evaluate_polynomial_database(
    encoding: SudaPolynomialDatabase,
    query_points: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Evaluate encoded database polynomials at one or more row query points."""

    _validate_plain_tensor(query_points, "query_points")
    rows = []
    for point in query_points.reshape(-1).tolist():
        value = torch.zeros(encoding.coefficients.shape[1], dtype=torch.long)
        for coefficient in reversed(encoding.coefficients):
            value = (value * int(point) + coefficient) % encoding.modulus
        rows.append(_field_values_to_tensor(value.tolist(), encoding.modulus))
    flat = torch.stack(rows, dim=0).to(dtype=dtype)
    if device is not None:
        flat = flat.to(device=device)
    return flat.reshape((len(rows),) + encoding.record_shape)


def suda_opr_reduce_polynomials(coefficients: torch.Tensor, query_points: torch.Tensor, *, modulus: int) -> torch.Tensor:
    """Suda Protocol 4 algebra: reduce f_k(x) modulo g(x)=prod_i(x-z_i)."""

    _validate_plain_tensor(coefficients, "coefficients")
    _validate_plain_tensor(query_points, "query_points")
    points = [int(point) % modulus for point in query_points.reshape(-1).tolist()]
    if not points:
        raise ValueError("query_points must not be empty")
    n = len(points)
    g = _query_vanishing_polynomial(points, modulus)
    chunk_count = (coefficients.shape[0] + n - 1) // n
    h_powers = [_poly_mod_mod(_poly_x_power_mod(j * n, modulus), g, modulus) for j in range(chunk_count)]
    reduced_columns = []
    for column in range(coefficients.shape[1]):
        reduced = [0]
        column_coefficients = [int(value) % modulus for value in coefficients[:, column].tolist()]
        for chunk_index in range(chunk_count):
            chunk = column_coefficients[chunk_index * n : (chunk_index + 1) * n]
            product = _poly_mul_mod(chunk, h_powers[chunk_index], modulus)
            reduced = _poly_add_mod(reduced, product, modulus)
        reduced_columns.append(_trim_poly(reduced))
    return _columns_to_coefficient_matrix(reduced_columns)


def suda_ope_mask_polynomials(
    coefficients: torch.Tensor,
    query_points: torch.Tensor,
    *,
    modulus: int,
    seed: int = 20260708,
) -> tuple[torch.Tensor, int]:
    """Suda Protocol 5 algebra: return f_k(x)+g(x)gamma_k(x)."""

    points = [int(point) % modulus for point in query_points.reshape(-1).tolist()]
    g = _query_vanishing_polynomial(points, modulus)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    masked_columns = []
    gamma_degree = max(0, len(points) - 1)
    for column in range(coefficients.shape[1]):
        gamma = torch.randint(0, modulus, (gamma_degree + 1,), dtype=torch.long, generator=generator).tolist()
        product = _poly_mul_mod(gamma, g, modulus)
        masked = _poly_add_mod([int(v) for v in coefficients[:, column].tolist()], product, modulus)
        masked_columns.append(_trim_poly(masked))
    masked_matrix = _columns_to_coefficient_matrix(masked_columns)
    return masked_matrix, _poly_matrix_degree(masked_matrix)


def suda_opi_interpolate_share_polynomials(
    server_shares: torch.Tensor,
    query_points: torch.Tensor,
    *,
    modulus: int,
) -> tuple[torch.Tensor, int]:
    """Suda Protocol 7 algebra using RF/CF basis polynomials."""

    _validate_plain_tensor(server_shares, "server_shares")
    points = [int(point) % modulus for point in query_points.reshape(-1).tolist()]
    batch_size = len(points)
    iota = int(np.ceil(np.sqrt(batch_size)))
    flat_shares = server_shares.reshape(batch_size, -1) % modulus
    row_basis = []
    column_basis = []
    for row_group in range(iota):
        values = [1 if index // iota == row_group else 0 for index in range(batch_size)]
        row_basis.append(_interpolate_values_at_points(points, values, modulus))
    for column_group in range(iota):
        values = [1 if index % iota == column_group else 0 for index in range(batch_size)]
        column_basis.append(_interpolate_values_at_points(points, values, modulus))

    output_columns = []
    for payload_column in range(flat_shares.shape[1]):
        poly = [0]
        for index in range(batch_size):
            row_group = index // iota
            column_group = index % iota
            basis_product = _poly_mul_mod(row_basis[row_group], column_basis[column_group], modulus)
            scaled = _poly_scalar_mul_mod(basis_product, int(flat_shares[index, payload_column]), modulus)
            poly = _poly_add_mod(poly, scaled, modulus)
        output_columns.append(_trim_poly(poly))
    return _columns_to_coefficient_matrix(output_columns), iota


def evaluate_polynomial_coefficients(
    coefficients: torch.Tensor,
    query_points: torch.Tensor,
    modulus: int,
    record_shape: tuple[int, ...],
) -> torch.Tensor:
    rows = []
    for point in query_points.reshape(-1).tolist():
        value = torch.zeros(coefficients.shape[1], dtype=torch.long)
        for coefficient in reversed(coefficients):
            value = (value * int(point) + coefficient) % modulus
        rows.append(value)
    return torch.stack(rows, dim=0).reshape((len(rows),) + record_shape)


def _encrypted_opr_reduce_polynomials(he: Any, coefficients: torch.Tensor, points: list[int], modulus: int) -> list[list[Any]]:
    n = len(points)
    g = _query_vanishing_polynomial(points, modulus)
    chunk_count = (coefficients.shape[0] + n - 1) // n
    encrypted_h_powers = [
        _encrypt_plain_polynomial(he, _poly_mod_mod(_poly_x_power_mod(j * n, modulus), g, modulus))
        for j in range(chunk_count)
    ]
    encrypted_columns = []
    for column in range(coefficients.shape[1]):
        encrypted_reduced = [_pyfhel_encrypt_int(he, 0)]
        column_coefficients = [int(value) % modulus for value in coefficients[:, column].tolist()]
        for chunk_index in range(chunk_count):
            chunk = column_coefficients[chunk_index * n : (chunk_index + 1) * n]
            product = _encrypted_poly_mul_plain(he, encrypted_h_powers[chunk_index], chunk)
            encrypted_reduced = _encrypted_poly_add(he, encrypted_reduced, product)
        encrypted_columns.append(_encrypted_trim_poly(he, encrypted_reduced))
    return encrypted_columns


def _encrypted_ope_mask_polynomials(
    he: Any,
    encrypted_columns: list[list[Any]],
    points: list[int],
    modulus: int,
    *,
    seed: int,
) -> tuple[list[list[Any]], int]:
    encrypted_g = _encrypt_plain_polynomial(he, _query_vanishing_polynomial(points, modulus))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    gamma_degree = max(0, len(points) - 1)
    masked_columns = []
    for column in encrypted_columns:
        gamma = torch.randint(0, modulus, (gamma_degree + 1,), dtype=torch.long, generator=generator).tolist()
        mask = _encrypted_poly_mul_plain(he, encrypted_g, [int(value) for value in gamma])
        masked_columns.append(_encrypted_trim_poly(he, _encrypted_poly_add(he, column, mask)))
    return masked_columns, _encrypted_poly_matrix_degree(masked_columns)


def _encrypted_opi_interpolate_share_polynomials(
    he: Any,
    server_shares: torch.Tensor,
    points: list[int],
    modulus: int,
) -> tuple[list[list[Any]], int]:
    batch_size = len(points)
    iota = int(np.ceil(np.sqrt(batch_size)))
    flat_shares = server_shares.reshape(batch_size, -1) % modulus
    encrypted_row_basis = []
    encrypted_column_basis = []
    for row_group in range(iota):
        values = [1 if index // iota == row_group else 0 for index in range(batch_size)]
        encrypted_row_basis.append(_encrypt_plain_polynomial(he, _interpolate_values_at_points(points, values, modulus)))
    for column_group in range(iota):
        values = [1 if index % iota == column_group else 0 for index in range(batch_size)]
        encrypted_column_basis.append(_encrypt_plain_polynomial(he, _interpolate_values_at_points(points, values, modulus)))

    encrypted_columns = []
    for payload_column in range(flat_shares.shape[1]):
        encrypted_poly = [_pyfhel_encrypt_int(he, 0)]
        for index in range(batch_size):
            row_group = index // iota
            column_group = index % iota
            basis_product = _encrypted_poly_mul(he, encrypted_row_basis[row_group], encrypted_column_basis[column_group])
            scaled = _encrypted_poly_scalar_mul(he, basis_product, int(flat_shares[index, payload_column]))
            encrypted_poly = _encrypted_poly_add(he, encrypted_poly, scaled)
        encrypted_columns.append(_encrypted_trim_poly(he, encrypted_poly))
    return encrypted_columns, iota


def _decrypt_poly_matrix_and_evaluate(
    he: Any,
    encrypted_columns: list[list[Any]],
    query_points: torch.Tensor,
    modulus: int,
    record_shape: tuple[int, ...],
) -> torch.Tensor:
    plain_columns = []
    for column in encrypted_columns:
        plain_columns.append([_pyfhel_decrypt_int(he, ciphertext) % modulus for ciphertext in column])
    coefficients = _columns_to_coefficient_matrix(plain_columns)
    return evaluate_polynomial_coefficients(coefficients, query_points, modulus, record_shape)


def _encrypt_plain_polynomial(he: Any, coefficients: list[int]) -> list[Any]:
    return [_pyfhel_encrypt_int(he, int(coefficient)) for coefficient in _trim_poly(coefficients)]


def _encrypted_poly_add(he: Any, left: list[Any], right: list[Any]) -> list[Any]:
    size = max(len(left), len(right))
    out = []
    for index in range(size):
        l_value = left[index] if index < len(left) else _pyfhel_encrypt_int(he, 0)
        r_value = right[index] if index < len(right) else _pyfhel_encrypt_int(he, 0)
        out.append(l_value + r_value)
    return out


def _encrypted_poly_sub(he: Any, left: list[Any], right: list[Any]) -> list[Any]:
    size = max(len(left), len(right))
    out = []
    for index in range(size):
        l_value = left[index] if index < len(left) else _pyfhel_encrypt_int(he, 0)
        r_value = right[index] if index < len(right) else _pyfhel_encrypt_int(he, 0)
        out.append(l_value - r_value)
    return out


def _encrypted_poly_mul_plain(he: Any, encrypted: list[Any], plain: list[int]) -> list[Any]:
    out = [_pyfhel_encrypt_int(he, 0) for _ in range(len(encrypted) + len(plain) - 1)]
    for enc_degree, enc_value in enumerate(encrypted):
        for plain_degree, plain_value in enumerate(plain):
            if int(plain_value) == 0:
                continue
            out[enc_degree + plain_degree] = out[enc_degree + plain_degree] + enc_value * int(plain_value)
    return out


def _encrypted_poly_mul(he: Any, left: list[Any], right: list[Any]) -> list[Any]:
    out = [_pyfhel_encrypt_int(he, 0) for _ in range(len(left) + len(right) - 1)]
    for left_degree, left_value in enumerate(left):
        for right_degree, right_value in enumerate(right):
            product = left_value * right_value
            he.relinearize(product)
            out[left_degree + right_degree] = out[left_degree + right_degree] + product
    return out


def _encrypted_poly_scalar_mul(he: Any, encrypted: list[Any], scalar: int) -> list[Any]:
    if int(scalar) == 0:
        return [_pyfhel_encrypt_int(he, 0) for _ in encrypted]
    return [value * int(scalar) for value in encrypted]


def _encrypted_poly_matrix_sub(he: Any, left: list[list[Any]], right: list[list[Any]]) -> list[list[Any]]:
    if len(left) != len(right):
        raise ValueError("encrypted polynomial matrices must have the same number of columns")
    return [_encrypted_trim_poly(he, _encrypted_poly_sub(he, l_col, r_col)) for l_col, r_col in zip(left, right)]


def _encrypted_poly_matrix_degree(columns: list[list[Any]]) -> int:
    return max((len(column) - 1 for column in columns), default=0)


def _encrypted_trim_poly(_he: Any, coefficients: list[Any]) -> list[Any]:
    return coefficients or []


def _evaluate_encrypted_polynomials_horner(he: Any, encoding: SudaPolynomialDatabase, encrypted_point: Any) -> list[Any]:
    encrypted_values = []
    for column in range(encoding.coefficients.shape[1]):
        encrypted = _pyfhel_encrypt_int(he, 0)
        for coefficient in reversed(encoding.coefficients[:, column].tolist()):
            encrypted = encrypted * encrypted_point + int(coefficient)
        encrypted_values.append(encrypted)
    return encrypted_values


def _pyfhel_encrypt_int(he: Any, value: int) -> Any:
    return he.encryptInt(np.array([int(value)], dtype=np.int64))


def _pyfhel_decrypt_int(he: Any, ciphertext: Any) -> int:
    return int(he.decryptInt(ciphertext)[0])


def _one_hot_query_points(indicators: torch.Tensor) -> torch.Tensor:
    if not torch.allclose(indicators.sum(dim=1), torch.ones(indicators.shape[0], device=indicators.device, dtype=indicators.dtype)):
        raise ValueError("polynomial PIR prototype expects one-hot query indicators")
    indices = indicators.argmax(dim=1).to(dtype=torch.long)
    return indices.cpu() + 1


def _one_hot_row_ids_zero_based(indicators: torch.Tensor) -> list[int]:
    if not torch.allclose(indicators.sum(dim=1), torch.ones(indicators.shape[0], device=indicators.device, dtype=indicators.dtype)):
        raise ValueError("Suda native bridge expects one-hot query indicators")
    return [int(value) for value in indicators.argmax(dim=1).to(dtype=torch.long).cpu().tolist()]


def _load_suda_native_bridge(module_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            "SudaNativeBridgeBackend requires the native Suda bridge extension. "
            "Build NssMPClib/native/suda_bridge against sls33/Suda first, or pass "
            "an explicit test module to SudaNativeBridgeBackend(module=...)."
        ) from exc


def _next_power_of_two(value: int) -> int:
    if value <= 0:
        raise ValueError("value must be positive")
    return 1 << (int(value) - 1).bit_length()


def _log2_exact(value: int) -> int:
    if value <= 0 or value & (value - 1):
        raise ValueError("value must be a positive power of two")
    return int(value.bit_length() - 1)


def _suda_native_padded_database_size(database_size: int, batch_size: int, real_batch_size: int) -> int:
    dummy_count = batch_size - real_batch_size
    if dummy_count < 0:
        raise ValueError("batch_size must be at least the number of selected ids")
    # Suda's query polynomial is built from a batch of distinct database points.
    # For padding, reserve real dummy rows instead of repeating one id. Suda's
    # packed encoder also requires host_n_data to be a multiple of 4096.
    return _next_power_of_two(max(database_size + dummy_count, batch_size, SUDA_NATIVE_HALF_POLY_MOD_DEGREE))


def _choose_suda_native_batch_size(real_batch_size: int, feature_num: int, requested_batch_size: int | None) -> int:
    if real_batch_size <= 0:
        raise ValueError("batch size must be positive")
    if requested_batch_size is not None:
        if requested_batch_size not in SUDA_NATIVE_SUPPORTED_BATCH_SIZES:
            raise ValueError(f"unsupported Suda native batch_size={requested_batch_size}")
        if requested_batch_size < real_batch_size:
            raise ValueError("requested Suda native batch_size is smaller than the number of selected ids")
        factor = SUDA_NATIVE_HALF_POLY_MOD_DEGREE // requested_batch_size
        if requested_batch_size != SUDA_NATIVE_HALF_POLY_MOD_DEGREE and feature_num % factor != 0:
            raise ValueError(
                "requested Suda native batch_size is incompatible with feature_num packing: "
                f"feature_num={feature_num}, factor={factor}"
            )
        return requested_batch_size
    for candidate in SUDA_NATIVE_SUPPORTED_BATCH_SIZES:
        factor = SUDA_NATIVE_HALF_POLY_MOD_DEGREE // candidate
        if (
            candidate >= real_batch_size
            and SUDA_NATIVE_HALF_POLY_MOD_DEGREE % candidate == 0
            and (candidate == SUDA_NATIVE_HALF_POLY_MOD_DEGREE or feature_num % factor == 0)
        ):
            return candidate
    raise ValueError(
        "Suda native bridge currently supports at most "
        f"{SUDA_NATIVE_SUPPORTED_BATCH_SIZES[-1]} ids per PIR call"
    )


def _pad_suda_native_query_ids(
    row_ids: list[int],
    database_size: int,
    padded_database_size: int,
    batch_size: int,
) -> list[int]:
    if len(row_ids) > batch_size:
        raise ValueError("row_ids cannot be larger than native batch_size")
    padding_pool = list(range(database_size, padded_database_size))
    if len(padding_pool) < batch_size - len(row_ids):
        raise ValueError("padded database does not contain enough dummy ids for native query padding")
    padded = list(row_ids)
    while len(padded) < batch_size:
        padded.append(padding_pool[len(padded) - len(row_ids)])
    return padded


def _normalise_suda_native_output(
    native_result: Any,
    *,
    real_batch_size: int,
    feature_num: int,
    modulus: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    meta: dict[str, Any] = {}
    if isinstance(native_result, dict):
        server_share = native_result["server_share"]
        client_share = native_result["client_share"]
        meta = dict(native_result.get("meta", {}))
        if "query_bytes" in native_result:
            meta["query_bytes"] = native_result["query_bytes"]
        if "response_bytes" in native_result:
            meta["response_bytes"] = native_result["response_bytes"]
        if "modulus" in native_result:
            modulus = int(native_result["modulus"])
    else:
        server_share, client_share = native_result
    server = torch.as_tensor(server_share, dtype=torch.long)
    client = torch.as_tensor(client_share, dtype=torch.long)
    if server.shape[0] == feature_num:
        server = server.transpose(0, 1)
    if client.shape[0] == feature_num:
        client = client.transpose(0, 1)
    server = server[:real_batch_size, :feature_num] % modulus
    client = client[:real_batch_size, :feature_num] % modulus
    if tuple(server.shape) != (real_batch_size, feature_num) or tuple(client.shape) != (real_batch_size, feature_num):
        raise ValueError("native Suda bridge returned shares with an unexpected shape")
    return server.contiguous(), client.contiguous(), meta


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _can_use_default_suda_he_backend(indicators: Any, database: Any) -> bool:
    if not isinstance(indicators, torch.Tensor) or not isinstance(database, torch.Tensor):
        return False
    if len(indicators.shape) != 2 or len(database.shape) < 1 or indicators.shape[1] != database.shape[0]:
        return False
    try:
        _one_hot_query_points(indicators)
        _tensor_to_field(database, 65_537)
    except (TypeError, ValueError):
        return False
    return True


def _tensor_to_field(values: torch.Tensor, modulus: int) -> torch.Tensor:
    rounded = values.detach().cpu().round()
    if not torch.allclose(values.detach().cpu(), rounded.to(dtype=values.dtype), atol=1e-6):
        raise ValueError("polynomial PIR prototype currently supports integer-valued payloads only")
    return rounded.to(dtype=torch.long).reshape(values.shape[0], -1) % modulus


def _tensor_to_centered_int_array(values: torch.Tensor, modulus: int) -> np.ndarray:
    rounded = values.detach().cpu().round()
    if not torch.allclose(values.detach().cpu(), rounded.to(dtype=values.dtype), atol=1e-6):
        raise ValueError("BFV share conversion currently supports integer-valued shares only")
    signed = rounded.to(dtype=torch.long) % modulus
    signed = torch.where(signed > modulus // 2, signed - modulus, signed)
    return signed.reshape(-1).numpy().astype(np.int64)


def _field_values_to_tensor(values: list[int], modulus: int) -> torch.Tensor:
    half = modulus // 2
    signed = [value - modulus if value > half else value for value in values]
    return torch.tensor(signed, dtype=torch.float32)


def _field_tensor_to_signed_float(values: torch.Tensor, modulus: int) -> torch.Tensor:
    signed = values.clone().to(dtype=torch.long) % modulus
    signed = torch.where(signed > modulus // 2, signed - modulus, signed)
    return signed.to(dtype=torch.float32)


def _query_vanishing_polynomial(points: list[int], modulus: int) -> list[int]:
    polynomial = [1]
    for point in points:
        polynomial = _poly_mul_linear_mod(polynomial, (-point) % modulus, modulus)
    return _trim_poly(polynomial)


def _interpolate_values_at_points(points: list[int], values: list[int], modulus: int) -> list[int]:
    if len(points) != len(values):
        raise ValueError("points and values must have the same length")
    polynomial = [0]
    for point, value in zip(points, values):
        basis = [1]
        denominator = 1
        for other in points:
            if other == point:
                continue
            basis = _poly_mul_linear_mod(basis, (-other) % modulus, modulus)
            denominator = (denominator * ((point - other) % modulus)) % modulus
        scaled = _poly_scalar_mul_mod(basis, (int(value) * pow(denominator, -1, modulus)) % modulus, modulus)
        polynomial = _poly_add_mod(polynomial, scaled, modulus)
    return _trim_poly(polynomial)


def _poly_add_mod(left: list[int], right: list[int], modulus: int) -> list[int]:
    size = max(len(left), len(right))
    out = [0] * size
    for index in range(size):
        l_value = left[index] if index < len(left) else 0
        r_value = right[index] if index < len(right) else 0
        out[index] = (l_value + r_value) % modulus
    return _trim_poly(out)


def _poly_sub_mod(left: list[int], right: list[int], modulus: int) -> list[int]:
    size = max(len(left), len(right))
    out = [0] * size
    for index in range(size):
        l_value = left[index] if index < len(left) else 0
        r_value = right[index] if index < len(right) else 0
        out[index] = (l_value - r_value) % modulus
    return _trim_poly(out)


def _poly_mul_mod(left: list[int], right: list[int], modulus: int) -> list[int]:
    if not left or not right:
        return [0]
    out = [0] * (len(left) + len(right) - 1)
    for l_degree, l_value in enumerate(left):
        for r_degree, r_value in enumerate(right):
            out[l_degree + r_degree] = (out[l_degree + r_degree] + l_value * r_value) % modulus
    return _trim_poly(out)


def _poly_scalar_mul_mod(coefficients: list[int], scalar: int, modulus: int) -> list[int]:
    return _trim_poly([(int(value) * int(scalar)) % modulus for value in coefficients])


def _poly_mod_mod(dividend: list[int], divisor: list[int], modulus: int) -> list[int]:
    remainder = _trim_poly([int(value) % modulus for value in dividend])
    divisor = _trim_poly([int(value) % modulus for value in divisor])
    if divisor == [0]:
        raise ValueError("polynomial divisor must be non-zero")
    divisor_degree = len(divisor) - 1
    divisor_lead_inverse = pow(divisor[-1], -1, modulus)
    while len(remainder) - 1 >= divisor_degree and remainder != [0]:
        scale_degree = len(remainder) - len(divisor)
        scale = (remainder[-1] * divisor_lead_inverse) % modulus
        for index, coefficient in enumerate(divisor):
            remainder[index + scale_degree] = (remainder[index + scale_degree] - scale * coefficient) % modulus
        remainder = _trim_poly(remainder)
    return remainder


def _poly_x_power_mod(power: int, modulus: int) -> list[int]:
    out = [0] * (power + 1)
    out[power] = 1 % modulus
    return out


def _columns_to_coefficient_matrix(columns: list[list[int]]) -> torch.Tensor:
    height = max(len(column) for column in columns)
    out = torch.zeros((height, len(columns)), dtype=torch.long)
    for column_index, column in enumerate(columns):
        out[: len(column), column_index] = torch.tensor(column, dtype=torch.long)
    return out


def _poly_matrix_sub_mod(left: torch.Tensor, right: torch.Tensor, modulus: int) -> torch.Tensor:
    columns = []
    width = max(left.shape[1], right.shape[1])
    if left.shape[1] != right.shape[1]:
        raise ValueError("polynomial matrices must have the same number of columns")
    for column in range(width):
        l_values = [int(value) for value in left[:, column].tolist()]
        r_values = [int(value) for value in right[:, column].tolist()]
        columns.append(_poly_sub_mod(l_values, r_values, modulus))
    return _columns_to_coefficient_matrix(columns)


def _poly_matrix_degree(coefficients: torch.Tensor) -> int:
    for row in range(coefficients.shape[0] - 1, -1, -1):
        if torch.any(coefficients[row] != 0):
            return row
    return 0


def _trim_poly(coefficients: list[int]) -> list[int]:
    if not coefficients:
        return [0]
    end = len(coefficients)
    while end > 1 and coefficients[end - 1] == 0:
        end -= 1
    return coefficients[:end]


def _poly_mul_linear_mod(coefficients: list[int], constant: int, modulus: int) -> list[int]:
    out = [0] * (len(coefficients) + 1)
    for degree, coefficient in enumerate(coefficients):
        out[degree] = (out[degree] + coefficient * constant) % modulus
        out[degree + 1] = (out[degree + 1] + coefficient) % modulus
    return out


def _validate_plain_tensor(value: Any, name: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a plain torch.Tensor for this backend")


def _validate_pir_shapes(indicators: Any, database: Any) -> None:
    if len(indicators.shape) != 2:
        raise ValueError("indicators must have shape [batch_size, database_size]")
    if len(database.shape) < 1:
        raise ValueError("database must have shape [database_size, ...]")
    if indicators.shape[1] != database.shape[0]:
        raise ValueError(
            "indicator database_size does not match database rows: "
            f"{indicators.shape[1]} != {database.shape[0]}"
        )
