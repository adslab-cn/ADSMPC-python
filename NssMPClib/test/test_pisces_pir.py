import os
import queue
import sys
import threading

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from NssMPC import ArithmeticSecretSharing, RingTensor
from NssMPC.application.rag.pisces.pir import (
    SudaBFVPolynomialBackend,
    SudaEncryptedOPROPEOPIBackend,
    SudaLFHEPolynomialBackend,
    SudaPIRToSharePlaintextProtocolBackend,
    SudaPolynomialPlaintextBackend,
    decrypt_bfv_ciphertext_to_tensor,
    encode_database_as_polynomials,
    evaluate_polynomial_coefficients,
    evaluate_polynomial_database,
    shares_to_bfv_ciphertext,
    suda_ope_mask_polynomials,
    suda_opi_interpolate_share_polynomials,
    suda_opr_reduce_polynomials,
    suda_pir_to_share,
)
from NssMPC.config.runtime import PartyRuntime
from NssMPC.secure_model.mpc_party import SemiHonestCS


def test_suda_pir_to_share_selects_rows_with_available_backend():
    database = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
            [[9.0, 10.0], [11.0, 12.0]],
        ]
    )
    indicators = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )

    result = suda_pir_to_share(indicators, database)

    assert result.audit.batch_size == 2
    assert result.audit.database_size == 3
    assert result.audit.record_shape == (2, 2)
    assert result.audit.output_shape == (2, 2, 2)
    assert result.audit.implementation in {
        "suda-bfv-encrypted-opr-ope-opi-backend",
        "suda-plaintext-opr-ope-opi-backend",
    }
    assert torch.equal(result.records, database[[1, 0]])
    if result.server_share is not None and result.client_share is not None:
        assert torch.equal(result.server_share + result.client_share, result.records)


def test_suda_lfhe_backend_is_isolated_until_implemented():
    indicators = torch.tensor([[1.0]])
    database = torch.tensor([[7.0]])

    try:
        suda_pir_to_share(indicators, database, backend=SudaLFHEPolynomialBackend())
    except NotImplementedError as exc:
        assert "LFHE/BFV" in str(exc)
    else:
        raise AssertionError("expected NotImplementedError only for explicit LFHE/BFV backend")


def test_suda_polynomial_encoding_roundtrips_integer_payloads():
    database = torch.tensor(
        [
            [3.0, -1.0, 8.0],
            [4.0, 6.0, 2.0],
            [0.0, 5.0, 7.0],
            [9.0, 1.0, -3.0],
        ],
        dtype=torch.float32,
    )

    encoding = encode_database_as_polynomials(database, modulus=65_537)
    query_points = torch.arange(1, database.shape[0] + 1)
    restored = evaluate_polynomial_database(encoding, query_points, dtype=database.dtype)

    assert encoding.degree == database.shape[0] - 1
    assert encoding.coefficients.shape == (database.shape[0], database.shape[1])
    assert torch.equal(restored, database)


def test_suda_polynomial_plaintext_backend_selects_rows_by_polynomial_evaluation():
    database = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
            [[9.0, 10.0], [11.0, 12.0]],
            [[13.0, 14.0], [15.0, 16.0]],
        ]
    )
    indicators = torch.tensor(
        [
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ]
    )

    result = suda_pir_to_share(indicators, database, backend=SudaPolynomialPlaintextBackend(modulus=65_537))

    assert torch.equal(result.records, database[[2, 0]])
    assert result.audit.implementation == "suda-polynomial-plaintext-backend"
    assert result.audit.polynomial_degree == 3
    assert result.audit.polynomial_modulus == 65_537
    assert result.audit.paper_backend_available is False


def test_suda_opr_reduces_degree_and_preserves_query_evaluations():
    database = torch.tensor(
        [
            [3.0, -1.0],
            [4.0, 6.0],
            [0.0, 5.0],
            [9.0, 2.0],
            [8.0, 1.0],
        ],
        dtype=torch.float32,
    )
    encoding = encode_database_as_polynomials(database, modulus=65_537)
    query_points = torch.tensor([2, 4])

    reduced = suda_opr_reduce_polynomials(encoding.coefficients, query_points, modulus=encoding.modulus)
    original_values = evaluate_polynomial_coefficients(
        encoding.coefficients,
        query_points,
        encoding.modulus,
        encoding.record_shape,
    )
    reduced_values = evaluate_polynomial_coefficients(reduced, query_points, encoding.modulus, encoding.record_shape)

    assert reduced.shape[0] <= 2 * query_points.numel() - 1
    assert torch.equal(reduced_values, original_values)


def test_suda_ope_masks_polynomials_without_changing_query_values():
    database = torch.tensor(
        [
            [3.0, -1.0],
            [4.0, 6.0],
            [0.0, 5.0],
            [9.0, 2.0],
        ],
        dtype=torch.float32,
    )
    encoding = encode_database_as_polynomials(database, modulus=65_537)
    query_points = torch.tensor([1, 3])
    reduced = suda_opr_reduce_polynomials(encoding.coefficients, query_points, modulus=encoding.modulus)

    masked, masked_degree = suda_ope_mask_polynomials(reduced, query_points, modulus=encoding.modulus, seed=7)
    reduced_values = evaluate_polynomial_coefficients(reduced, query_points, encoding.modulus, encoding.record_shape)
    masked_values = evaluate_polynomial_coefficients(masked, query_points, encoding.modulus, encoding.record_shape)

    assert masked_degree >= reduced.shape[0] - 1
    assert torch.equal(masked_values, reduced_values)


def test_suda_opi_interpolates_server_share_basis_polynomials():
    modulus = 65_537
    query_points = torch.tensor([1, 2, 4])
    server_shares = torch.tensor(
        [
            [10, 20],
            [30, 40],
            [50, 60],
        ],
        dtype=torch.long,
    )

    share_polynomials, iota = suda_opi_interpolate_share_polynomials(server_shares, query_points, modulus=modulus)
    restored = evaluate_polynomial_coefficients(share_polynomials, query_points, modulus, (2,))

    assert iota == 2
    assert torch.equal(restored, server_shares)


def test_suda_plaintext_protocol_backend_outputs_additive_shares():
    database = torch.tensor(
        [
            [3.0, -1.0],
            [4.0, 6.0],
            [0.0, 5.0],
            [9.0, 2.0],
        ],
        dtype=torch.float32,
    )
    indicators = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    result = suda_pir_to_share(
        indicators,
        database,
        backend=SudaPIRToSharePlaintextProtocolBackend(modulus=65_537, seed=11),
    )

    assert torch.equal(result.records, database[[1, 3]])
    assert torch.equal(result.server_share + result.client_share, result.records)
    assert result.audit.implementation == "suda-plaintext-opr-ope-opi-backend"
    assert result.audit.opr_reduced_degree <= 2 * indicators.shape[0] - 2
    assert result.audit.opi_iota == 2


def test_suda_bfv_backend_reports_missing_optional_dependency_cleanly():
    indicators = torch.tensor([[1.0]])
    database = torch.tensor([[7.0]])

    try:
        import Pyfhel  # noqa: F401
    except ImportError:
        try:
            suda_pir_to_share(indicators, database, backend=SudaBFVPolynomialBackend())
        except ImportError as exc:
            assert "Pyfhel" in str(exc)
        else:
            raise AssertionError("expected ImportError when Pyfhel is unavailable")


def test_suda_bfv_backend_selects_rows_when_pyfhel_is_available():
    try:
        import Pyfhel  # noqa: F401
    except ImportError:
        return

    database = torch.tensor(
        [
            [3.0, -1.0],
            [4.0, 6.0],
            [0.0, 5.0],
        ],
        dtype=torch.float32,
    )
    indicators = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    result = suda_pir_to_share(
        indicators,
        database,
        backend=SudaBFVPolynomialBackend(modulus=65_537, poly_modulus_degree=8_192),
    )

    assert torch.equal(result.records, database[[1, 2]])
    assert result.audit.implementation == "suda-bfv-polynomial-prototype"
    assert result.audit.paper_backend_available is True
    assert result.audit.he_scheme == "BFV"


def test_share_to_he_conversion_encrypts_reconstructed_payload():
    try:
        import Pyfhel  # noqa: F401
    except ImportError:
        return

    database = torch.tensor(
        [
            [3.0, -1.0],
            [4.0, 6.0],
            [0.0, 5.0],
        ],
        dtype=torch.float32,
    )
    indicators = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    pir_result = suda_pir_to_share(
        indicators,
        database,
        backend=SudaEncryptedOPROPEOPIBackend(modulus=65_537, poly_modulus_degree=8_192, seed=13),
    )
    he_result = shares_to_bfv_ciphertext(
        pir_result.server_share,
        pir_result.client_share,
        modulus=65_537,
        poly_modulus_degree=8_192,
    )
    restored = decrypt_bfv_ciphertext_to_tensor(he_result, dtype=database.dtype)

    assert torch.equal(restored, pir_result.records)
    assert he_result.audit.paper_step == "Pisces Protocol 1 Step 6 / Protocol 2 Step 7"


def _party_worker(party, indicator_share, database_share, out_queue):
    try:
        party.set_multiplication_provider()
        party.online()
        with PartyRuntime(party):
            result = suda_pir_to_share(indicator_share, database_share)
            restored = result.records.restore().convert_to_real_field()
            out_queue.put((party.party_id, restored.cpu(), result.audit, None))
    except BaseException as exc:
        out_queue.put((getattr(party, "party_id", None), None, None, exc))
    finally:
        try:
            party.close()
        except BaseException:
            pass


def test_suda_pir_to_share_ass_two_party_runtime():
    database = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ],
        dtype=torch.float32,
    )
    indicators = torch.tensor(
        [
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    expected = database[[2, 0, 3]]

    indicator_shares = ArithmeticSecretSharing.share(RingTensor.convert_to_ring(indicators), 2)
    database_shares = ArithmeticSecretSharing.share(RingTensor.convert_to_ring(database), 2)

    server = SemiHonestCS(type="server")
    client = SemiHonestCS(type="client")
    out_queue = queue.Queue()
    threads = [
        threading.Thread(target=_party_worker, args=(server, indicator_shares[0], database_shares[0], out_queue), daemon=True),
        threading.Thread(target=_party_worker, args=(client, indicator_shares[1], database_shares[1], out_queue), daemon=True),
    ]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert not any(thread.is_alive() for thread in threads), "PIR-to-share ASS runtime test timed out"

    outputs = [out_queue.get_nowait() for _ in range(out_queue.qsize())]
    errors = [error for _, _, _, error in outputs if error is not None]
    if errors:
        raise errors[0]

    assert len(outputs) == 2
    for _, restored, audit, _ in outputs:
        assert torch.equal(restored, expected)
        assert audit.batch_size == 3
        assert audit.database_size == 4
        assert audit.record_shape == (3,)
        assert audit.output_shape == (3, 3)
        assert audit.implementation == "suda-pir-to-share-ass-indicator-backend"
        assert audit.paper_backend_available is False


if __name__ == "__main__":
    test_suda_pir_to_share_selects_rows_with_available_backend()
    test_suda_lfhe_backend_is_isolated_until_implemented()
    test_suda_polynomial_encoding_roundtrips_integer_payloads()
    test_suda_polynomial_plaintext_backend_selects_rows_by_polynomial_evaluation()
    test_suda_opr_reduces_degree_and_preserves_query_evaluations()
    test_suda_ope_masks_polynomials_without_changing_query_values()
    test_suda_opi_interpolates_server_share_basis_polynomials()
    test_suda_plaintext_protocol_backend_outputs_additive_shares()
    test_suda_bfv_backend_reports_missing_optional_dependency_cleanly()
    test_suda_bfv_backend_selects_rows_when_pyfhel_is_available()
    test_share_to_he_conversion_encrypts_reconstructed_payload()
    test_suda_pir_to_share_ass_two_party_runtime()
    print("pisces pir tests ok")
