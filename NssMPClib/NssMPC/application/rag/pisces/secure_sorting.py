"""Secure sorting/top-k building block for Pisces.

The public entry point follows the Pisces/Panther direction: a data-independent
Panther-style ApproxTopK followed by exact top-k over the bin winners.

The Panther paper uses the optimized network inside a GC backend. NssMPClib does
not currently provide a GC runtime, so the implementation here keeps the same
data-independent compare-swap structure but executes comparisons through the
available secret-sharing backend when scores are ``ArithmeticSecretSharing``.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import subprocess
import tempfile
import threading
import time
from typing import Any

import torch

try:
    from NssMPC import ArithmeticSecretSharing, RingTensor
    from NssMPC.config import DEVICE
    from NssMPC.config.runtime import PartyRuntime
except ImportError:  # Allows plain unit tests outside the NssMPC runtime.
    ArithmeticSecretSharing = None
    RingTensor = None
    PartyRuntime = None
    DEVICE = "cpu"


PANTHER_APPROX_DELTA = 0.01
PANTHER_APPROX_SEED = 20260708


@dataclass(frozen=True)
class SecureTopKAudit:
    num_items: int
    top_k: int
    comparisons: int
    compare_swaps: int
    indicator_shape: tuple[int, int]
    backend: str
    network: str
    padded_width: int | None = None
    padded_items: int | None = None
    algorithm: str = "panther-approx-topk-then-exact-topk"
    k_prime: int | None = None
    delta: float | None = None
    bin_count: int | None = None
    max_bin_size: int | None = None
    bin_comparisons: int = 0
    exact_comparisons: int = 0
    paper_backend: str = "SS bin selection + GC exact top-k"
    paper_backend_available: bool = False
    backend_gap: str = "exact top-k uses ASS comparisons instead of Panther GC backend"
    gc_binary: str | None = None
    gc_value_bits: int | None = None
    gc_port: int | None = None
    gc_communication_bytes: int | None = None
    topk_ids_owner: str = "public"


@dataclass(frozen=True)
class SecureTopKResult:
    indicators: Any
    audit: SecureTopKAudit


_PANTHER_GC_COUNTER_BY_PARTY: dict[int, int] = {}
_PANTHER_GC_COUNTER_LOCK = threading.Lock()


def secure_top_k_indicators(
    scores: Any,
    k: int,
    *,
    return_audit: bool = False,
    pad_value: float = -1_000_000_000.0,
) -> Any | SecureTopKResult:
    """Return a [k, num_items] indicator matrix using the Pisces top-k path.

    Plain tensors use the local data-independent Panther-style reference path.
    Arithmetic-secret-shared scores must use the OpenPanther GC bridge; if that
    backend is unavailable the paper-level path is not available and this
    function fails instead of silently falling back to ASS comparisons.
    """

    if k < 1:
        raise ValueError("k must be positive")
    num_items = int(scores.shape[-1])
    if k > num_items:
        raise ValueError("k cannot exceed the number of scores")

    if _is_ass(scores):
        return _panther_gc_top_k_indicators(scores, k, return_audit=return_audit)

    score_items, indicator_items, backend = _make_items(scores)
    pairs = list(zip(score_items, indicator_items))
    k_prime = _panther_k_prime(num_items, k)
    bins = _partition_pairs(pairs, k_prime=k_prime, seed=PANTHER_APPROX_SEED)
    winners: list[tuple[Any, Any]] = []
    bin_comparisons = 0
    max_bin_size = 0
    for bin_pairs in bins:
        max_bin_size = max(max_bin_size, len(bin_pairs))
        winner, used = _bin_max(bin_pairs)
        winners.append(winner)
        bin_comparisons += used

    top_pairs, comparisons, padded_width, padded_items = _panther_top_k_network(
        winners,
        k,
        pad_value=pad_value,
    )
    total_comparisons = bin_comparisons + comparisons
    audit = SecureTopKAudit(
        num_items,
        k,
        total_comparisons,
        total_comparisons,
        (k, num_items),
        backend,
        "panther-approx-topk-ass-exact-ass",
        padded_width,
        padded_items,
        k_prime=k_prime,
        delta=PANTHER_APPROX_DELTA,
        bin_count=len(bins),
        max_bin_size=max_bin_size,
        bin_comparisons=bin_comparisons,
        exact_comparisons=comparisons,
    )
    indicators = _stack_indicators([indicator for _, indicator in top_pairs])
    return SecureTopKResult(indicators, audit) if return_audit else indicators


def _panther_k_prime(num_items: int, k: int) -> int:
    return min(num_items, max(k, int((k / PANTHER_APPROX_DELTA) + 0.999999999)))


def _partition_pairs(pairs: list[tuple[Any, Any]], *, k_prime: int, seed: int) -> list[list[tuple[Any, Any]]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    order = torch.randperm(len(pairs), generator=generator).tolist()
    bins = [[] for _ in range(k_prime)]
    for offset, item_index in enumerate(order):
        bins[offset % k_prime].append(pairs[item_index])
    return bins


def _bin_max(pairs: list[tuple[Any, Any]]) -> tuple[tuple[Any, Any], int]:
    if not pairs:
        raise ValueError("ApproxTopK bin must not be empty")
    winner = pairs[0]
    comparisons = 0
    for candidate in pairs[1:]:
        winner, _ = _compare_swap_pair_desc(winner, candidate)
        comparisons += 1
    return winner, comparisons


def _make_items(scores: Any) -> tuple[list[Any], list[Any], str]:
    num_items = int(scores.shape[-1])
    if _is_ass(scores):
        flat_scores = scores.reshape(-1)
        score_items = [flat_scores[index] for index in range(num_items)]
        return score_items, _public_one_hot_ass_indicators(num_items), "arithmetic_secret_sharing"

    flat_scores = scores.reshape(-1)
    score_items = [flat_scores[index] for index in range(num_items)]
    eye = torch.eye(num_items, device=scores.device, dtype=scores.dtype)
    return score_items, [eye[index] for index in range(num_items)], "torch-network"


def _stack_indicators(indicators: list[Any]) -> Any:
    if indicators and _is_ass(indicators[0]):
        return ArithmeticSecretSharing.cat([indicator.unsqueeze(0) for indicator in indicators], dim=0)
    return torch.stack(indicators, dim=0)


def _panther_top_k_network(
    pairs: list[tuple[Any, Any]],
    k: int,
    *,
    pad_value: float,
) -> tuple[list[tuple[Any, Any]], int, int, int]:
    width = _next_power_of_two(k)
    pad_pair = _padding_pair(pairs[0], pad_value=pad_value)
    runs: list[list[tuple[Any, Any]]] = []
    comparisons = 0

    for start in range(0, len(pairs), width):
        run = list(pairs[start : start + width])
        while len(run) < width:
            run.append(pad_pair)
        run, used = _odd_even_merge_sort(run, descending=True)
        comparisons += used
        runs.append(run)

    while len(runs) > 1:
        merged_runs: list[list[tuple[Any, Any]]] = []
        for index in range(0, len(runs), 2):
            if index + 1 >= len(runs):
                merged_runs.append(runs[index])
                continue
            merged, used = _top_k_merge_sorted_desc(
                runs[index],
                runs[index + 1],
                width,
            )
            comparisons += used
            merged_runs.append(merged)
        runs = merged_runs

    return runs[0][:k], comparisons, width, len(runs[0])


def _top_k_merge_sorted_desc(
    left: list[tuple[Any, Any]],
    right: list[tuple[Any, Any]],
    width: int,
) -> tuple[list[tuple[Any, Any]], int]:
    pairs = list(left) + list(reversed(right))
    comparisons = 0

    for index in range(width):
        pairs[index], pairs[index + width] = _compare_swap_pair_desc(pairs[index], pairs[index + width])
        comparisons += 1

    first, used = _bitonic_merge(pairs[:width], descending=True)
    comparisons += used
    return first, comparisons


def _odd_even_merge_sort(pairs: list[tuple[Any, Any]], *, descending: bool) -> tuple[list[tuple[Any, Any]], int]:
    out = list(pairs)
    return out, _odd_even_merge_sort_range(out, 0, len(out), descending)


def _odd_even_merge_sort_range(
    pairs: list[tuple[Any, Any]],
    start: int,
    size: int,
    descending: bool,
) -> int:
    if size <= 1:
        return 0
    half = size // 2
    comparisons = _odd_even_merge_sort_range(pairs, start, half, descending)
    comparisons += _odd_even_merge_sort_range(pairs, start + half, half, descending)
    comparisons += _odd_even_merge(pairs, start, size, 1, descending)
    return comparisons


def _odd_even_merge(
    pairs: list[tuple[Any, Any]],
    start: int,
    size: int,
    step: int,
    descending: bool,
) -> int:
    double_step = step * 2
    comparisons = 0
    if double_step < size:
        comparisons += _odd_even_merge(pairs, start, size, double_step, descending)
        comparisons += _odd_even_merge(pairs, start + step, size, double_step, descending)
        for index in range(start + step, start + size - step, double_step):
            pairs[index], pairs[index + step] = _compare_swap_pair(
                pairs[index],
                pairs[index + step],
                descending,
            )
            comparisons += 1
    else:
        pairs[start], pairs[start + step] = _compare_swap_pair(
            pairs[start],
            pairs[start + step],
            descending,
        )
        comparisons += 1
    return comparisons


def _bitonic_merge(pairs: list[tuple[Any, Any]], *, descending: bool) -> tuple[list[tuple[Any, Any]], int]:
    size = len(pairs)
    if size <= 1:
        return list(pairs), 0
    distance = size // 2
    out = list(pairs)
    comparisons = 0
    for index in range(distance):
        out[index], out[index + distance] = _compare_swap_pair(out[index], out[index + distance], descending)
        comparisons += 1

    left, left_used = _bitonic_merge(out[:distance], descending=descending)
    right, right_used = _bitonic_merge(out[distance:], descending=descending)
    return left + right, comparisons + left_used + right_used


def _compare_swap_pair(
    left: tuple[Any, Any],
    right: tuple[Any, Any],
    descending: bool,
) -> tuple[tuple[Any, Any], tuple[Any, Any]]:
    high, low = _compare_swap_pair_desc(left, right)
    if descending:
        return high, low
    return low, high


def _compare_swap_pair_desc(
    left: tuple[Any, Any],
    right: tuple[Any, Any],
) -> tuple[tuple[Any, Any], tuple[Any, Any]]:
    left_score, left_indicator = left
    right_score, right_indicator = right
    choose_right = right_score > left_score
    if isinstance(choose_right, torch.Tensor):
        high_score = torch.where(choose_right, right_score, left_score)
        low_score = torch.where(choose_right, left_score, right_score)
        high_indicator = torch.where(choose_right.reshape(()).bool(), right_indicator, left_indicator)
        low_indicator = torch.where(choose_right.reshape(()).bool(), left_indicator, right_indicator)
        return (high_score, high_indicator), (low_score, low_indicator)
    score_delta = right_score - left_score
    indicator_delta = right_indicator - left_indicator
    score_swap = choose_right * score_delta
    indicator_swap = choose_right * indicator_delta
    return (left_score + score_swap, left_indicator + indicator_swap), (right_score - score_swap, right_indicator - indicator_swap)


def _padding_pair(example: tuple[Any, Any], *, pad_value: float) -> tuple[Any, Any]:
    example_score, example_indicator = example
    if _is_ass(example_score):
        return _public_ass_scalar(pad_value), ArithmeticSecretSharing(RingTensor.zeros_like(example_indicator.item))
    return torch.as_tensor(pad_value, dtype=example_score.dtype, device=example_score.device), torch.zeros_like(example_indicator)


def _public_ass_scalar(value: float) -> Any:
    public_ring = RingTensor.convert_to_ring(torch.tensor(value, device=DEVICE))
    if _runtime_party_id() not in (None, 0):
        public_ring = RingTensor.zeros_like(public_ring)
    return ArithmeticSecretSharing(public_ring)


def _public_one_hot_ass_indicators(num_items: int) -> list[Any]:
    public_ring = RingTensor.convert_to_ring(torch.eye(num_items, device=DEVICE))
    if _runtime_party_id() not in (None, 0):
        public_ring = RingTensor.zeros_like(public_ring)
    return [ArithmeticSecretSharing(public_ring[index]) for index in range(num_items)]


def _panther_gc_topk_enabled() -> bool:
    return Path(_panther_gc_topk_binary()).exists()


def _panther_gc_topk_binary() -> str:
    return os.environ.get(
        "PANTHER_GC_TOPK_BIN",
        "/tmp/OpenPanther/bazel-bin/experimental/panther/pisces_gc_topk_cli",
    )


def _panther_gc_top_k_indicators(scores: Any, k: int, *, return_audit: bool) -> Any | SecureTopKResult:
    party_id = _runtime_party_id()
    if party_id not in (0, 1):
        raise RuntimeError("OpenPanther GC top-k requires a two-party NssMPClib runtime")
    binary = _panther_gc_topk_binary()
    if not Path(binary).exists():
        raise FileNotFoundError(binary)

    flat = scores.reshape(-1)
    num_items = int(flat.shape[-1])
    value_bits = int(os.environ.get("PANTHER_GC_TOPK_VALUE_BITS", "31"))
    if not 2 <= value_bits <= 31:
        raise ValueError("PANTHER_GC_TOPK_VALUE_BITS must be in [2, 31]")
    id_bits = max(1, (num_items - 1).bit_length())
    score_upper_bound = float(os.environ.get("PANTHER_GC_TOPK_SCORE_UPPER_BOUND", "10000"))
    offset = int(round(score_upper_bound * int(flat.scale)))
    if offset <= 0 or offset >= (1 << (value_bits - 1)):
        raise ValueError("PANTHER_GC_TOPK_SCORE_UPPER_BOUND is outside the GC signed range")

    raw_share = flat.item.tensor.detach().cpu().reshape(-1)
    mask = (1 << value_bits) - 1
    if party_id == 0:
        distance_share = [((offset - int(value)) & mask) for value in raw_share.tolist()]
    else:
        distance_share = [((-int(value)) & mask) for value in raw_share.tolist()]

    counter = _next_panther_gc_counter(party_id)
    port = int(os.environ.get("PANTHER_GC_TOPK_PORT_BASE", "19000")) + counter
    bin_count = _panther_k_prime(num_items, k)
    emp_party = party_id + 1
    prefix = f"pisces_gc_topk_{os.getpid()}_{counter}_{party_id}_"
    timeout = float(os.environ.get("PANTHER_GC_TOPK_TIMEOUT", "120"))

    with tempfile.TemporaryDirectory(prefix=prefix) as tmpdir:
        input_path = Path(tmpdir) / "input.txt"
        output_path = Path(tmpdir) / "output.txt"
        input_path.write_text("\n".join(str(value) for value in distance_share) + "\n", encoding="ascii")
        if emp_party == 2:
            time.sleep(float(os.environ.get("PANTHER_GC_TOPK_CLIENT_DELAY", "0.2")))
        completed = subprocess.run(
            [
                binary,
                str(emp_party),
                str(port),
                str(k),
                str(value_bits),
                str(id_bits),
                str(input_path),
                str(output_path),
                str(bin_count),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "OpenPanther GC top-k failed: "
                + (completed.stderr.strip() or completed.stdout.strip() or f"exit {completed.returncode}")
            )
        ids = [int(line.strip()) for line in output_path.read_text(encoding="ascii").splitlines() if line.strip()]

    if party_id == 0:
        ids = []
    elif len(ids) != k or any(index < 0 or index >= num_items for index in ids):
        raise RuntimeError(f"OpenPanther GC top-k returned invalid ids: {ids}")

    indicators = _client_owned_ass_indicators_from_ids(ids, k, num_items)
    communication_bytes = _parse_gc_communication_bytes(completed.stderr)
    exact_comparisons = num_items * k
    audit = SecureTopKAudit(
        num_items,
        k,
        exact_comparisons,
        exact_comparisons,
        (k, num_items),
        "arithmetic_secret_sharing",
        "openpanther-gc-approx-topk",
        padded_width=k,
        padded_items=num_items,
        k_prime=bin_count,
        delta=PANTHER_APPROX_DELTA,
        bin_count=bin_count,
        max_bin_size=(num_items + bin_count - 1) // bin_count,
        bin_comparisons=max(0, num_items - bin_count),
        exact_comparisons=exact_comparisons,
        paper_backend="OpenPanther EMP GC Approximate_topk",
        paper_backend_available=True,
        backend_gap="score-to-distance conversion is done as a public affine transform on the two local shares",
        gc_binary=binary,
        gc_value_bits=value_bits,
        gc_port=port,
        gc_communication_bytes=communication_bytes,
        topk_ids_owner="client",
    )
    return SecureTopKResult(indicators, audit) if return_audit else indicators


def _next_panther_gc_counter(party_id: int) -> int:
    with _PANTHER_GC_COUNTER_LOCK:
        counter = _PANTHER_GC_COUNTER_BY_PARTY.get(party_id, 0)
        _PANTHER_GC_COUNTER_BY_PARTY[party_id] = counter + 1
    return counter


def _client_owned_ass_indicators_from_ids(ids: list[int], k: int, num_items: int) -> Any:
    plain = torch.zeros((k, num_items), device=DEVICE, dtype=torch.float32)
    if _runtime_party_id() == 1:
        for row, index in enumerate(ids):
            plain[row, index] = 1.0
    public_ring = RingTensor.convert_to_ring(plain)
    return ArithmeticSecretSharing(public_ring)


def _parse_gc_communication_bytes(stderr: str) -> int | None:
    match = re.search(r"communication_bytes=(\d+)", stderr)
    return int(match.group(1)) if match else None


def _panther_network_comparisons(num_items: int, k: int) -> int:
    width = _next_power_of_two(k)
    runs = (num_items + width - 1) // width
    comparisons = runs * _odd_even_merge_sort_comparisons(width)
    active = runs
    while active > 1:
        merges = active // 2
        comparisons += merges * (width + _bitonic_merge_comparisons(width))
        active = merges + (active % 2)
    return comparisons


def _odd_even_merge_sort_comparisons(size: int) -> int:
    if size <= 1:
        return 0
    half = size // 2
    return (
        2 * _odd_even_merge_sort_comparisons(half)
        + _odd_even_merge_comparisons(size, 1)
    )


def _odd_even_merge_comparisons(size: int, step: int) -> int:
    double_step = step * 2
    if double_step >= size:
        return 1
    merge_comparisons = (
        _odd_even_merge_comparisons(size, double_step)
        + _odd_even_merge_comparisons(size, double_step)
    )
    cross_comparisons = len(range(step, size - step, double_step))
    return merge_comparisons + cross_comparisons


def _bitonic_merge_comparisons(size: int) -> int:
    if size <= 1:
        return 0
    return size // 2 + 2 * _bitonic_merge_comparisons(size // 2)


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _is_ass(value: Any) -> bool:
    return ArithmeticSecretSharing is not None and isinstance(value, ArithmeticSecretSharing)


def _runtime_party_id() -> int | None:
    if PartyRuntime is None:
        return None
    try:
        return PartyRuntime.party.party_id
    except RuntimeError:
        return None
