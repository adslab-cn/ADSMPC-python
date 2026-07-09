import torch

from NssMPC.application.rag.pisces.ops import indicator_top_k
from NssMPC.application.rag.pisces.secure_sorting import secure_top_k_indicators


def selected_indices(indicators):
    return [int(row.argmax().item()) for row in indicators]


def test_secure_top_k_plain_network_matches_torch_topk():
    scores = torch.tensor([0.5, -1.0, 3.0, 2.0, 3.5, 0.0], dtype=torch.float32)
    result = secure_top_k_indicators(scores, 3, return_audit=True)

    assert selected_indices(result.indicators) == torch.topk(scores, 3).indices.tolist()
    assert result.audit.num_items == 6
    assert result.audit.top_k == 3
    assert result.audit.indicator_shape == (3, 6)
    assert result.audit.backend == "torch-network"
    assert result.audit.network == "panther-approx-topk-ass-exact-ass"
    assert result.audit.algorithm == "panther-approx-topk-then-exact-topk"
    assert result.audit.k_prime == scores.numel()
    assert result.audit.bin_comparisons == 0
    assert result.audit.paper_backend_available is False
    assert result.audit.padded_width == 4


def test_panther_top_k_plain_network_matches_torch_topk():
    generator = torch.Generator().manual_seed(20260707)
    scores = torch.randn(33, generator=generator)
    result = secure_top_k_indicators(scores, 7, return_audit=True)

    assert selected_indices(result.indicators) == torch.topk(scores, 7).indices.tolist()
    assert result.audit.network == "panther-approx-topk-ass-exact-ass"
    assert result.audit.padded_width == 8
    assert result.audit.comparisons > 0


def test_panther_approx_top_k_reduces_to_bin_winners_before_exact_topk():
    scores = torch.linspace(-1.0, 1.0, steps=640)
    result = secure_top_k_indicators(scores, 5, return_audit=True)

    assert result.indicators.shape == (5, 640)
    assert result.audit.algorithm == "panther-approx-topk-then-exact-topk"
    assert result.audit.k_prime == 500
    assert result.audit.bin_count == 500
    assert result.audit.max_bin_size == 2
    assert result.audit.bin_comparisons == 140
    assert result.audit.exact_comparisons > 0
    assert result.audit.comparisons == result.audit.bin_comparisons + result.audit.exact_comparisons


def test_indicator_top_k_uses_secure_sorting_frontend():
    scores = torch.tensor([4.0, 1.0, 2.0, 9.0], dtype=torch.float32)
    indicators = indicator_top_k(scores, 2)

    assert indicators.shape == (2, 4)
    assert selected_indices(indicators) == [3, 0]
    assert torch.equal(indicators.sum(dim=1), torch.ones(2))


def test_secure_top_k_rejects_invalid_k():
    scores = torch.tensor([1.0, 2.0])
    try:
        secure_top_k_indicators(scores, 3)
    except ValueError as exc:
        assert "k cannot exceed" in str(exc)
    else:
        raise AssertionError("expected ValueError for k > num_items")


if __name__ == "__main__":
    test_secure_top_k_plain_network_matches_torch_topk()
    test_panther_top_k_plain_network_matches_torch_topk()
    test_panther_approx_top_k_reduces_to_bin_winners_before_exact_topk()
    test_indicator_top_k_uses_secure_sorting_frontend()
    test_secure_top_k_rejects_invalid_k()
    print("pisces secure sorting tests ok")
