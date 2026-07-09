import queue
import threading

import torch

from NssMPC import ArithmeticSecretSharing, RingTensor
from NssMPC.application.rag.pisces.secure_sorting import secure_top_k_indicators
from NssMPC.config.runtime import PartyRuntime
from NssMPC.secure_model.mpc_party import SemiHonestCS


def _party_worker(party, cases, out_queue):
    try:
        party.set_multiplication_provider()
        party.set_comparison_provider()
        party.online()
        with PartyRuntime(party):
            outputs = []
            for name, score_share, top_k in cases:
                result = secure_top_k_indicators(score_share, top_k, return_audit=True)
                restored = result.indicators.restore().convert_to_real_field()
                outputs.append((name, restored.cpu(), result.audit))
            out_queue.put((party.party_id, outputs, None))
    except BaseException as exc:
        out_queue.put((party.party_id, exc, None))
    finally:
        try:
            party.close()
        except BaseException:
            pass


def test_secure_top_k_ass_two_party_runtime():
    scores_panther = torch.tensor([-0.3560, -0.2780, -1.1423, -1.7070, -0.5160, -0.9072], dtype=torch.float32)
    expected_panther = torch.zeros(5, scores_panther.numel(), dtype=torch.float32)
    for row, index in enumerate(torch.topk(scores_panther, 5).indices.tolist()):
        expected_panther[row, index] = 1.0

    shares_panther = ArithmeticSecretSharing.share(RingTensor.convert_to_ring(scores_panther), 2)
    expected_by_name = {
        "panther": expected_panther,
    }
    server = SemiHonestCS(type="server")
    client = SemiHonestCS(type="client")
    out_queue = queue.Queue()
    server_cases = [
        ("panther", shares_panther[0], 5),
    ]
    client_cases = [
        ("panther", shares_panther[1], 5),
    ]

    threads = [
        threading.Thread(target=_party_worker, args=(server, server_cases, out_queue), daemon=True),
        threading.Thread(target=_party_worker, args=(client, client_cases, out_queue), daemon=True),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert not any(thread.is_alive() for thread in threads), "secure top-k ASS runtime test timed out"

    outputs = [out_queue.get_nowait() for _ in range(out_queue.qsize())]
    errors = [value for _, value, _ in outputs if isinstance(value, BaseException)]
    if errors:
        raise errors[0]

    party_outputs = [value for _, value, _ in outputs]
    assert len(party_outputs) == 2
    for case_outputs in party_outputs:
        for name, restored, audit in case_outputs:
            assert torch.equal(restored, expected_by_name[name])
            assert audit.backend == "arithmetic_secret_sharing"
            assert audit.network == "panther-approx-topk-ass-exact-ass"
            assert audit.algorithm == "panther-approx-topk-then-exact-topk"
            assert audit.num_items == 6
            assert audit.top_k == 5
            assert audit.k_prime == 6
            assert audit.bin_comparisons == 0
            assert audit.exact_comparisons == 19
            assert audit.comparisons == 19


if __name__ == "__main__":
    test_secure_top_k_ass_two_party_runtime()
    print("pisces secure sorting ASS runtime test ok")
