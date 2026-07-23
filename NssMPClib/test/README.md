# Pisces Test Entry Points

## Recommended end-to-end run

Use this for the current two-party Pisces RAG demo:

```bash
cd /home/adslab/pazika/ADSMPC-python

SKIP_GEN_PARAMS=1 \
PISCES_RAG_PROFILE=squad_dev_q8 \
PISCES_RAG_NUM_DOCS=1204 \
PISCES_RAG_TOP_K=10 \
PISCES_RAG_DOC_LEN=8 \
PISCES_RAG_P3_PAILLIER_KEY_SIZE=64 \
PISCES_RAG_REPORT_JSON=data/pisces_reports/rag_squad_1204_q8_e2e.json \
PYTHONPATH="$PWD/NssMPClib" \
/home/adslab/anaconda3/envs/nssmpc/bin/python NssMPClib/test/pisces_e2e.py
```

`pisces_e2e.py` is a stable wrapper around `rag.py`.

With `SKIP_GEN_PARAMS=1`, the runner skips unconditional regeneration but still
checks the run-specific auxiliary parameter requirements and regenerates missing
`DivKey`, `B2AKey`, `GeLUKey`, or `SigmaDICFKey` by default. Set
`PISCES_RAG_AUTO_GEN_PARAMS=0` for a strict no-write parameter check, or
`PISCES_RAG_CHECK_PARAMS=0` for a fastest-path run that assumes local parameters
are already sufficient.

## Files kept in this directory

- `pisces_e2e.py`: recommended end-to-end entry point.
- `rag.py`: current implementation of the two-party NssMPClib Pisces RAG flow.
- `bert_tiny_weights.pth`: tiny BERT weights loaded by `rag.py`.

## Subdirectories

- `protocol_checks/`: component-level tests for Protocol 1/2/3/4, PIR, sorting, and MHA.
- `research_tools/`: older demos and paper-style evaluation helpers.

The day-to-day run path should be `pisces_e2e.py`. Use the subdirectories only
when debugging a specific protocol component or reproducing paper tables.
