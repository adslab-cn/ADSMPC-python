# Panther GC Top-K bridge

This directory contains the C++ CLI used by the Pisces Python code to call
OpenPanther/EMP `emp-sh2pc` for Panther-style GC top-k.

The source is meant to be built inside an OpenPanther checkout, because it
depends on OpenPanther's Bazel-managed EMP dependencies.

```bash
cp /home/adslab/pazika/ADSMPC-python/NssMPClib/native/panther_gc_topk/panther_gc_topk_cli.cc \
  /tmp/OpenPanther/experimental/panther/pisces_gc_topk_cli.cc
```

Add a `spu_cc_binary` target in `/tmp/OpenPanther/experimental/panther/BUILD.bazel`:

```python
spu_cc_binary(
    name = "pisces_gc_topk_cli",
    srcs = ["pisces_gc_topk_cli.cc"],
    copts = AES_COPT_FLAGS,
    deps = [
        ":topk",
        "@com_github_emptoolkit_emp_sh2pc//:emp-sh2pc",
    ],
)
```

Then build:

```bash
HTTPS_PROXY=http://10.122.245.238:10808 \
HTTP_PROXY=http://10.122.245.238:10808 \
ALL_PROXY=http://10.122.245.238:10808 \
/usr/bin/bazel build -c opt //experimental/panther:pisces_gc_topk_cli \
  --action_env=PATH=/home/adslab/anaconda3/envs/nssmpc/bin:/usr/bin:/bin:/usr/local/bin
```

Set:

```bash
export PISCES_USE_PANTHER_GC_TOPK=1
export PANTHER_GC_TOPK_BIN=/tmp/OpenPanther/bazel-bin/experimental/panther/pisces_gc_topk_cli
```

The Python backend launches one process per NssMPClib party and returns public
ASS top-k indicators.

Notes:

- OpenPanther's top-k routines are written for minimum-distance search. The
  Python bridge converts Pisces scores to public-offset distances before
  launching this CLI.
- The current bridge uses 31-bit GC distance inputs by default, matching the
  32-bit masks inside OpenPanther's `Approximate_topk`/`Naive_topk` helpers.
