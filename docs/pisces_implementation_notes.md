# Pisces implementation notes

## Reproduction target

The Pisces paper evaluates three RAG datasets with granite-embedding-small-english-r2 embeddings
and BERT tokenization. The paper's dataset sizes are:

- ClapNQ: dev answerable 1,990 chunks, train answerable 14,010 chunks, train single answerable 71,363 chunks.
- SQuAD: dev v2.0 35 documents / 1,204 chunks, train v2.0 442 documents / 19,029 chunks.
- HotpotQA: dev distractor 66,581 documents / 269,602 chunks, dev fullwiki 66,573 documents / 276,013 chunks, training 482,021 documents / 1,795,146 chunks.

The paper runs 300 queries for the Dev answerable dataset and 1,000 queries for the other datasets.
Its key efficiency tables report:

- Protocol 1 semantic coarse-to-fine vs fine-only: time, upload, download, and accuracy.
- Protocol 4 multi-instance labeled PSI vs labeled PSI: time, upload, and download.

## Current implementation status

- Protocol 4 is a real interactive multi-instance labeled PSI implementation:
  - server setup: PRF/OPRF-derived labels, KDF0/KDF1, AES-CTR label encryption, binary OKVS encode.
  - online: finite-field DH-OPRF request/response, OKVS decode, AES label validation, TF matrix recovery.
  - verified on synthetic communication tests and real SQuAD smoke runs.
- Protocol 3 is a real oblivious-filter-style implementation:
  - SimHash projection masks, additive Paillier encryption, Shamir degree-1 interpolation, OKVS buckets, shuffled encrypted candidate secrets.
  - implemented with bucket padding so hash collisions are handled without dropping candidates.
  - current smoke parameters are intentionally small unless explicitly raised: `simhash_bits=16`, `p3_projections=8`, `p3_threshold=2`, `he_key_size=64`.
- Protocol 2 is wired through Protocol 4 plus BM25 scoring:
  - client recovers query term frequencies through Protocol 4.
  - BM25 is computed from recovered TF and document lengths.
  - top-k goes through `secure_sorting.secure_top_k_indicators`, the single public Panther-style
    top-k path.
  - the ASS top-k branch is covered by a two-party `SemiHonestCS` runtime test.
- Protocol 1 is wired end to end:
  - Protocol 3 candidate filtering.
  - semantic scoring on filtered candidates.
  - top-k now uses the same secure top-k baseline.
  - payload retrieval goes through `suda_pir_to_share`; plain integer payloads use encrypted Suda
    OPR/OPE/OPI, while ASS payloads record the remaining ASS fallback.
- Real dataset runner exists at `NssMPClib/test/pisces_real_dataset_eval.py`:
  - supports `squad_dev_v2`, `squad_train_v2`, and `hotpotqa_dev_distractor`.
  - uses cached granite embeddings and BERT term-frequency caches.
  - supports CUDA for embedding preselection where applicable.
  - has offline setup caching for Protocol 3/4.
  - emits timing and wire-size estimates, and optional JSONL records via `--report-jsonl`.

## Verified local environment

- CPU: Intel i7-14700F class machine, 28 threads.
- Memory: 93 GiB total, about 88 GiB available during inspection.
- Disk: about 492 GiB free.
- GPU: NVIDIA GeForce RTX 4060 Ti, 8 GiB VRAM.
- Conda env: `/home/adslab/anaconda3/envs/nssmpc`.
- CUDA works only outside the filesystem/process sandbox. Use the VSCode/debug launcher or escalated shell execution.

## Latest smoke result

Command:

```bash
env DEVICE=cuda PYTHONUNBUFFERED=1 PYTHONPATH=/home/adslab/pazika/ADSMPC-python/NssMPClib \
  /home/adslab/anaconda3/envs/nssmpc/bin/python NssMPClib/test/pisces_real_dataset_eval.py \
  --dataset squad_dev_v2 --corpus-limit 80 --protocol-docs 80 \
  --limit-queries 3 --top-k 5 --verbose-queries 1 \
  --device cuda --setup-mode offline \
  --report-jsonl /tmp/pisces_squad_smoke.jsonl
```

Result with setup cache hit:

- P3 setup wire estimate: 366,784 bytes.
- P4 setup wire estimate: 346,764 bytes.
- Total offline setup wire estimate: 713,548 bytes.
- Average per-query online wire estimate: 1,896,158.67 bytes.
- Top-5 semantic hit: 0.3333.
- Top-5 lexical hit: 0.6667.
- Top-5 dual-union hit: 0.6667.
- Top-5 fused hit: 0.3333.
- Elapsed time: 21.204 seconds for 3 queries.
- Main online bottleneck under smoke parameters:
  - P3 client filter: about 4.07 seconds/query.
  - P3 server recover: about 2.07 seconds/query.
  - P4 OPRF + TF recovery: about 0.58 seconds/query total.

## Gap to paper-level reproduction

### Dataset coverage

- SQuAD dev/train and HotpotQA dev distractor caches are present.
- SQuAD original JSON files are present under `data/squad/`.
- HotpotQA original files are present under `/home/adslab/pazika/pisces/hotpot/`.
- ClapNQ is not currently connected in this repo.
- Full Hotpot training scale is not yet connected as a benchmark target.

Meaning: we can already run real SQuAD dev/train and HotpotQA dev distractor, but the current
script has only been smoke-tested at small `protocol_docs`. Full paper-size runs still need long-run
validation and probably setup/query caching per dataset.

### Protocol parameters

- Paper Protocol 3 uses projection count `T = 160`, projection weight `ceil(sqrt(t * L))`, and full SimHash bit length `L`.
- Current real-data smoke uses `T = 8`, `L = 16`, and `t = 2` to keep debugging fast.
- Paper security levels for HE/OPRF/AES should use production-size parameters.
- Current smoke uses 64-bit Paillier for speed; this is not a security-level reproduction.

Meaning: correctness of data flow is represented, but smoke timings/communication are not paper
numbers until parameters are raised.

### Secure sorting/top-k

- Paper invokes Panther/Li et al.-style secure sorting in Protocol 1 and Protocol 2.
- Current code exposes one public top-k path at `NssMPC.application.rag.pisces.secure_sorting`:
  - scores are compared with the NssMPClib ASS comparison backend when scores are secret shares.
  - swaps use arithmetic multiplexing rather than revealing branch decisions.
  - the network is data-independent and reports deterministic comparison counts.
  - it follows Panther's main ApproxTopK shape: public random partition into `k'` bins, SS/ASS
    winner selection inside each bin, then exact top-k over the bin winners.
  - `k'` is derived from Panther's experiment setting `k' = k / delta` with `delta = 0.01`, capped
    at `n` for small inputs.
  - the exact top-k stage uses an odd-even merge sort bootstrap plus the existing pruned top-k merge.
  - plain tensors run through the same compare-swap network for tests and baselines.
  - `NssMPClib/test/test_pisces_secure_sorting_ass.py` verifies the ASS branch with two local
    `SemiHonestCS` parties, real comparison auxiliary parameters, and local TCP communication.

Remaining gap: this is still not Panther's full mixed SS/GC implementation. The ApproxTopK structure
from Panther Section 6.1 and exact top-k network shape from Section 6.2 are implemented, but the
exact top-k stage runs over the existing ASS comparison backend instead of emp-sh2pc/GC.

Meaning: top-k is no longer a local plaintext ranking step for ASS inputs, and the algorithmic flow
now matches Panther's ApproxTopK-then-ExactTopK design. Time/communication are still not
paper-faithful until a GC backend or equivalent mixed-primitive implementation exists.

### PIR-to-share

- Paper uses Suda/Song et al. batch PIR-to-share in Protocol 1 and Protocol 2.
- Current code exposes:
  - `suda_pir_to_share`: the paper-intended entry point. It runs with the available ASS/local
    indicator-selection backend and records the LFHE/BFV gap in audit metadata.
  - `SudaPolynomialPlaintextBackend`: a non-private finite-field polynomial backend. It encodes
    every payload coordinate as a polynomial over row points and verifies that point evaluation
    recovers selected rows. This validates the Suda-style database polynomialization path, but it
    does not hide the query.
  - `SudaEncryptedOPROPEOPIBackend`: the default path for plain integer-valued tensors. It follows
    Suda's OPR/OPE/OPI message flow using Pyfhel BFV ciphertexts for polynomial coefficients:
    encrypted OPR reduction, encrypted OPI share interpolation, and encrypted OPE masking.
  - `SudaBFVPolynomialBackend`: an explicit older Pyfhel/BFV prototype. It encrypts query points and
    directly evaluates database polynomials; tests keep it as a smaller HE sanity check, but the
    OPR/OPE/OPI backend is the closer Suda path.
  - `SudaPIRToSharePlaintextProtocolBackend`: a correctness backend for Suda Protocols 4/5/7. It
    implements OPR (`f mod g` degree reduction), OPE (`f + g * gamma` masking), and OPI
    (`RF/CF` basis-polynomial interpolation of server random shares) over the finite field, then
    returns server/client additive shares whose sum is the selected payload.
  - `SudaLFHEPolynomialBackend`: isolated placeholder for the paper-level OPR/OPE/OPI backend. It
    raises `NotImplementedError` only if explicitly selected.

Remaining gap: the default plain-tensor path now uses encrypted OPR/OPE/OPI, but Pyfhel does not
expose Suda's packed LFHE polynomial-ciphertext interface directly. The implementation therefore
encrypts polynomial coefficients individually and performs coefficient-wise homomorphic convolution.
This is cryptographically real BFV transport and follows the Protocol 4/5/7 algebra, but its
time/communication are not the paper's packed LFHE numbers. ASS secret-share payloads still use the
ASS fallback because Suda PIR starts from a server-held plaintext database and client encrypted query.

Meaning: the PIR-to-share boundary runs end to end, but PIR communication/time are not
paper-faithful until an LFHE backend or equivalent polynomial-ciphertext layer is added.

### Share-to-HE conversion

- Pisces Protocol 1 Step 6 / Protocol 2 Step 7 converts retrieved additive shares to an HE
  ciphertext before private generation: the client encrypts its share, sends the ciphertext to the
  server, and the server homomorphically adds its own share.
- Current code exposes `shares_to_bfv_ciphertext` and `decrypt_bfv_ciphertext_to_tensor`.
  `shares_to_bfv_ciphertext` takes integer-valued server/client tensor shares, encrypts the client
  share with Pyfhel BFV, adds the server share as plaintext, and returns `Enc(server_share +
  client_share)` plus audit metadata.

Remaining gap: this models the Pisces share-to-HE handoff for tensor shares, but it is not yet wired
into `rag.py`'s ASS runtime path and it is not a packed HE inference backend. The end-to-end demo can
run retrieval, and PIR output shares can now be converted to BFV ciphertexts, but full private
generation remains outside the current implementation.

### Performance engineering

- Protocol 3 is currently Python-heavy and dominates smoke query time.
- Protocol 4 setup is expensive for larger corpora but now cacheable.
- Protocol 4 online cost is reasonable at smoke scale, but full Hotpot scale will still require careful
batching and possibly multiprocessing.

Meaning: the machine has enough RAM/disk and usable GPU, but Python cryptographic loops will be
the main obstacle to matching paper runtime.

## Immediate next steps

1. Run SQuAD dev full-size reproduction candidate:
   - `--dataset squad_dev_v2 --corpus-limit 1204 --protocol-docs 1204 --limit-queries 300`
   - first with smoke security parameters to validate functionality and metrics.
   - then raise Protocol 3 parameters toward paper settings.
2. Add a dedicated benchmark profile layer:
   - `smoke`: fast debug.
   - `squad-dev-paper-size`: 1204 chunks / 300 queries.
   - `squad-train-paper-size`: 19029 chunks / 1000 queries.
   - `hotpot-distractor-paper-size`: 269602 chunks / 1000 queries.
3. Add long-run resume/checkpoint support:
   - append JSONL per query.
   - skip already completed query ids.
   - keep setup cache separate from result cache.
4. Add a GC backend for exact top-k / ApproxTopK's winner ranking to match Panther's mixed-primitive
   communication/round behavior.
5. Optimize the current encrypted Suda PIR-to-share backend toward paper performance: replace
   coefficient-wise BFV ciphertexts with packed LFHE polynomial ciphertexts, add Suda's
   batching/packing, and add communication accounting matching Protocol 4/5/7.
6. Add paper-comparison scripts that load JSONL and print deltas against Table 4/Table 5 metrics.
