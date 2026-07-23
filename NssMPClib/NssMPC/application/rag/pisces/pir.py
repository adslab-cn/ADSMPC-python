"""Native Suda PIR-to-share boundary for the Pisces retrieval pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
import threading
from contextlib import contextmanager
from typing import Any

import torch
import numpy as np

SUDA_NATIVE_MODULUS = 1_337_006_139_375_617
SUDA_NATIVE_HALF_POLY_MOD_DEGREE = 4096
SUDA_NATIVE_SUPPORTED_BATCH_SIZES = (1, 4, 16, 64, 256, 1024, 2048, 4096)
_NATIVE_LOG_LOCK = threading.Lock()


@dataclass(frozen=True)
class SudaPIRToShareAudit:
    """Shape and implementation metadata for a PIR-to-share call."""

    batch_size: int
    database_size: int
    record_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    implementation: str = "suda-native-cpp-batch-pir-to-share-bridge"
    paper_backend: str = "Suda C++ BatchPirToShareServer/Client with SEAL polynomial ciphertexts"
    paper_backend_available: bool = True
    backend_gap: str = "native Suda bridge"
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
class SudaNativeServerState:
    """Reusable server-side Suda native state for one encoded payload DB."""

    server: Any
    layout_key: tuple[Any, ...]
    database_device: torch.device | str
    database_dtype: torch.dtype
    keys_loaded: bool = False


@dataclass
class SudaNativeServerAnswer:
    server_share: torch.Tensor
    response_message: dict[str, Any]
    audit: SudaPIRToShareAudit
    state: SudaNativeServerState


@dataclass
class SudaNativeClientAnswer:
    client_share: torch.Tensor
    records: torch.Tensor | None = None


@dataclass
class SudaNativePIRServerShare:
    """Server-side result of a split Suda PIR-to-share invocation."""

    share: Any
    audit: SudaPIRToShareAudit
    request: dict[str, Any]
    response_message: dict[str, Any]
    state: SudaNativeServerState | None = None

    @property
    def request_keys(self) -> Any | None:
        return self.request.get("keys")


@dataclass
class SudaNativePIRClientShare:
    """Client-side result of a split Suda PIR-to-share invocation."""

    share: Any
    audit: Any
    state: SudaNativeClientState
    request: dict[str, Any]
    response_message: dict[str, Any]

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


@contextmanager
def suppress_native_output(enabled: bool | None = None):
    """Hide noisy native C++ backend logs unless explicitly enabled."""

    if enabled is None:
        enabled = os.environ.get("PISCES_RAG_NATIVE_LOGS") != "1"
    if not enabled:
        yield
        return
    with _NATIVE_LOG_LOCK:
        old_stdout = os.dup(1)
        old_stderr = os.dup(2)
        try:
            with open(os.devnull, "w") as devnull:
                os.dup2(devnull.fileno(), 1)
                os.dup2(devnull.fileno(), 2)
                yield
        finally:
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)
            os.close(old_stdout)
            os.close(old_stderr)


def finite_field_share_to_float_ass(field_share: Any, modulus: int, *, device: Any | None = None):
    """Convert fixed-point signed Suda Z_p shares to NssMPClib ASS shares."""

    from NssMPC import ArithmeticSecretSharing, RingTensor
    from NssMPC.config import DEVICE

    target_device = DEVICE if device is None else device
    modulus = int(modulus)
    local = torch.as_tensor(field_share, dtype=torch.long, device=target_device)
    summed_share = ArithmeticSecretSharing(RingTensor(local, dtype="int", device=target_device))
    carry_share = summed_share >= modulus
    value_share = summed_share - carry_share * modulus
    sign_share = value_share >= ((modulus + 1) // 2)
    signed_share = value_share - sign_share * modulus
    return ArithmeticSecretSharing(RingTensor(signed_share.item.tensor.to(dtype=torch.long), dtype="float", device=target_device))


def suda_native_pir_to_share_server(
    party: Any,
    database: Any,
    *,
    selected_count: int,
    server_state: SudaNativeServerState | None = None,
    suppress_output: bool | None = None,
) -> SudaNativePIRServerShare:
    """Run the server side of one paper-level Suda batch PIR-to-share call."""

    layout = suda_native_make_layout(
        database_size=int(database.shape[0]),
        record_shape=tuple(int(dim) for dim in database.shape[1:]),
        selected_count=int(selected_count),
    )
    party.send(layout)
    request = party.receive()
    with suppress_native_output(suppress_output):
        answer = suda_native_server_answer(database, request, server_state=server_state)
    party.send(answer.response_message)
    party.send(audit_to_message(answer.audit))
    share = finite_field_share_to_float_ass(answer.server_share, answer.audit.polynomial_modulus)
    return SudaNativePIRServerShare(
        share=share,
        audit=answer.audit,
        request=request,
        response_message=answer.response_message,
        state=answer.state,
    )


def suda_native_pir_to_share_client(
    party: Any,
    row_ids: Any,
    *,
    dtype: torch.dtype = torch.float32,
    device: Any = "cpu",
    previous_state: SudaNativeClientState | None = None,
    suppress_output: bool | None = None,
) -> SudaNativePIRClientShare:
    """Run the client side of one paper-level Suda batch PIR-to-share call."""

    layout = party.receive()
    with suppress_native_output(suppress_output):
        if previous_state is None:
            state, request = suda_native_make_client_request(
                row_ids,
                layout,
                dtype=dtype,
                device=device,
            )
        else:
            state, request = suda_native_make_client_followup_request(
                previous_state,
                row_ids,
                layout,
            )
    party.send(request)
    response_message = party.receive()
    audit = party.receive()
    with suppress_native_output(suppress_output):
        answer = suda_native_client_extract(state, response_message)
    modulus = audit.get("polynomial_modulus") if isinstance(audit, dict) else audit.polynomial_modulus
    share = finite_field_share_to_float_ass(answer.client_share, modulus, device=device)
    return SudaNativePIRClientShare(
        share=share,
        audit=audit,
        state=state,
        request=request,
        response_message=response_message,
    )


def audit_to_message(audit: SudaPIRToShareAudit) -> dict[str, Any]:
    return {
        "implementation": audit.implementation,
        "paper_backend": audit.paper_backend,
        "paper_backend_available": audit.paper_backend_available,
        "backend_gap": audit.backend_gap,
        "output_shape": audit.output_shape,
        "polynomial_modulus": audit.polynomial_modulus,
        "native_batch_size": audit.native_batch_size,
        "native_padded_database_size": audit.native_padded_database_size,
        "native_query_bytes": audit.native_query_bytes,
        "native_response_bytes": audit.native_response_bytes,
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


def suda_native_make_client_followup_request(
    state: SudaNativeClientState,
    row_ids: Any,
    layout: dict[str, Any],
) -> tuple[SudaNativeClientState, dict[str, Any]]:
    """Generate another encrypted query with an existing Suda client/keypair."""

    real_row_ids = [int(value) for value in torch.as_tensor(row_ids).reshape(-1).tolist()]
    if len(real_row_ids) != int(layout["selected_count"]):
        raise ValueError("row_ids length must match Suda native layout selected_count")
    if int(layout["feature_num"]) != state.feature_num:
        raise ValueError("follow-up layout feature_num must match the existing Suda client")
    if int(layout["native_batch_size"]) < len(real_row_ids):
        raise ValueError("follow-up layout batch size is smaller than selected row count")

    padded_query_ids = _pad_suda_native_query_ids(
        real_row_ids,
        int(layout["database_size"]),
        int(layout["padded_database_size"]),
        int(layout["native_batch_size"]),
    )
    query = state.client.gen_query(np.asarray(padded_query_ids, dtype=np.int64))
    followup_state = SudaNativeClientState(
        client=state.client,
        real_batch_size=int(layout["selected_count"]),
        feature_num=int(layout["feature_num"]),
        output_shape=(int(layout["selected_count"]),) + tuple(layout["record_shape"]),
        modulus=int(layout["modulus"]),
        dtype=state.dtype,
        device=state.device,
    )
    request = {
        "query": query,
        "query_bytes": int(query["query_bytes"]),
        "layout": dict(layout),
    }
    return followup_state, request


def suda_native_server_answer(
    database: Any,
    request: dict[str, Any],
    *,
    mod_switch: bool = True,
    use_out_mem: bool = False,
    module_name: str = "NssMPC.application.rag.pisces._suda_bridge",
    module: Any | None = None,
    server_state: SudaNativeServerState | None = None,
) -> SudaNativeServerAnswer:
    module = module if module is not None else _load_suda_native_bridge(module_name)
    layout = dict(request["layout"])
    database_tensor = database if isinstance(database, torch.Tensor) else torch.as_tensor(database)
    if server_state is None:
        server_state = suda_native_make_server_state(
            database_tensor,
            layout,
            keys=request.get("keys"),
            use_out_mem=use_out_mem,
            module_name=module_name,
            module=module,
        )
    else:
        _validate_suda_native_server_state(server_state, layout)
        if "keys" in request:
            server_state.server.load_keys(request["keys"])
            server_state.keys_loaded = True
    if not server_state.keys_loaded:
        raise ValueError("Suda native server state has no loaded client keys")

    server = server_state.server
    response = server.gen_response(request["query"], bool(mod_switch))
    server_share_field = torch.as_tensor(server.extract_answer(), dtype=torch.long).T
    server_share = server_share_field[: int(layout["selected_count"]), : int(layout["feature_num"])]
    output_shape = (int(layout["selected_count"]),) + tuple(layout["record_shape"])
    server_share = server_share.reshape(output_shape).to(device=server_state.database_device)
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
        state=server_state,
    )


def suda_native_make_server_state(
    database: Any,
    layout: dict[str, Any],
    *,
    keys: Any | None = None,
    use_out_mem: bool = False,
    module_name: str = "NssMPC.application.rag.pisces._suda_bridge",
    module: Any | None = None,
) -> SudaNativeServerState:
    """Build a reusable Suda native server object for a payload database."""

    module = module if module is not None else _load_suda_native_bridge(module_name)
    layout = dict(layout)
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
    keys_loaded = keys is not None
    if keys_loaded:
        server.load_keys(keys)
    return SudaNativeServerState(
        server=server,
        layout_key=_suda_native_server_layout_key(layout),
        database_device=database_tensor.device,
        database_dtype=database_tensor.dtype,
        keys_loaded=keys_loaded,
    )


def _suda_native_server_layout_key(layout: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(layout["database_size"]),
        tuple(int(dim) for dim in layout["record_shape"]),
        int(layout["feature_num"]),
        int(layout["native_batch_size"]),
        int(layout["padded_database_size"]),
        int(layout["host_log_n_data"]),
        int(layout["modulus"]),
    )


def _validate_suda_native_server_state(state: SudaNativeServerState, layout: dict[str, Any]) -> None:
    if state.layout_key != _suda_native_server_layout_key(layout):
        raise ValueError("Suda native server state layout does not match request layout")


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


def _tensor_to_field(values: torch.Tensor, modulus: int) -> torch.Tensor:
    rounded = values.detach().cpu().round()
    if not torch.allclose(values.detach().cpu(), rounded.to(dtype=values.dtype), atol=1e-6):
        raise ValueError("native Suda bridge expects fixed-point integer payloads")
    return rounded.to(dtype=torch.long).reshape(values.shape[0], -1) % modulus


def _field_tensor_to_signed_float(values: torch.Tensor, modulus: int) -> torch.Tensor:
    signed = values.clone().to(dtype=torch.long) % modulus
    signed = torch.where(signed > modulus // 2, signed - modulus, signed)
    return signed.to(dtype=torch.float32)


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
