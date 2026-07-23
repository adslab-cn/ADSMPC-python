# Suda native bridge

This directory contains the first native bridge from the Python Pisces PIR
boundary to the original `sls33/Suda` C++ Batch PIR-to-share implementation.

The Python side calls:

```python
SudaNativeBridgeBackend()
```

which imports `NssMPC.application.rag.pisces._suda_bridge` and calls
`batch_pir_to_share(feature_major, query_ids, ...)`.

## Build outline

Install the Python-side build helpers in the `nssmpc` environment:

```bash
/home/adslab/anaconda3/envs/nssmpc/bin/python -m pip install cmake pybind11
```

Build the Suda third-party dependencies. On this machine the wrapper scripts
work for NTL, SEAL and IPCL/Paillier after the CMake policy flag is supplied;
libsodium may need the system `config.guess/config.sub` files if its helper
script stalls while downloading them.

```bash
git clone https://github.com/sls33/Suda.git /tmp/Suda
cd /tmp/Suda/src/third_party
bash ntl.get

git clone --depth 1 --branch v4.1.1 https://github.com/microsoft/SEAL.git
cd SEAL
PATH=/home/adslab/anaconda3/envs/nssmpc/bin:$PATH \
  /home/adslab/anaconda3/envs/nssmpc/bin/cmake -S . -B build \
  -DSEAL_USE_INTEL_HEXL=ON \
  -DSEAL_BUILD_EXAMPLES=ON \
  -DCMAKE_INSTALL_PREFIX=./seal_install \
  -DSEAL_THROW_ON_TRANSPARENT_CIPHERTEXT=OFF \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
PATH=/home/adslab/anaconda3/envs/nssmpc/bin:$PATH \
  /home/adslab/anaconda3/envs/nssmpc/bin/cmake --build build -j 8
PATH=/home/adslab/anaconda3/envs/nssmpc/bin:$PATH \
  /home/adslab/anaconda3/envs/nssmpc/bin/cmake --install build

cd /tmp/Suda/src/third_party
cp pailliercryptolib_cmake pailliercryptolib/CMakeLists.txt
cd pailliercryptolib
PATH=/home/adslab/anaconda3/envs/nssmpc/bin:$PATH \
  /home/adslab/anaconda3/envs/nssmpc/bin/cmake -S . -B build \
  -DCMAKE_INSTALL_PREFIX=./ext_lib \
  -DCMAKE_BUILD_TYPE=Release \
  -DIPCL_TEST=OFF \
  -DIPCL_BENCHMARK=OFF \
  -DIPCL_SHARED=OFF \
  -DIPCL_ENABLE_OMP=OFF \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
PATH=/home/adslab/anaconda3/envs/nssmpc/bin:$PATH \
  /home/adslab/anaconda3/envs/nssmpc/bin/cmake --build build -j 8
PATH=/home/adslab/anaconda3/envs/nssmpc/bin:$PATH \
  /home/adslab/anaconda3/envs/nssmpc/bin/cmake --build build --target install

cd /tmp/Suda/src/third_party
wget -c https://github.com/jedisct1/libsodium/releases/download/1.0.18-RELEASE/libsodium-1.0.18.tar.gz -O ./libsodium.tar.gz
tar zxf libsodium.tar.gz
mv libsodium-1.0.18 libsodium
rm libsodium.tar.gz
cd libsodium
sh autogen.sh
cp /usr/share/misc/config.guess build-aux/config.guess
cp /usr/share/misc/config.sub build-aux/config.sub
./configure CFLAGS='-Wall -O3 -maes -msse2 -msse4.1 -msse3 -mavx -mavx2 -mpclmul -mfma -Wfatal-errors -pthread -Wno-ignored-attributes -Wno-int-in-bool-context -Wno-sign-compare -Wno-catch-value -fopenmp'
make -j 8
```

Then build this bridge against the Suda checkout:

```bash
# Required for Suda batch sizes below 1024. The upstream file keeps only
# rotations 1024/2048/4096/8192 enabled, but packed PIR-to-share needs the
# smaller unzip rotations when Python selects batch_size=4/16/64/256.
python - <<'PY'
from pathlib import Path
path = Path("/tmp/Suda/src/include/batch_pir_basic.h")
text = path.read_text()
old = "std::vector<uint32_t> galois_eles = { 1 + 1024, 1 + 2048, 1 + 4096, 1 + 8192 };"
new = """std::vector<uint32_t> galois_eles = {
                1 + 2,
                1 + 4,
                1 + 8,
                1 + 16,
                1 + 32,
                1 + 64,
                1 + 128,
                1 + 256,
                1 + 512,
                1 + 1024,
                1 + 2048,
                1 + 4096,
                1 + 8192,
            };"""
path.write_text(text.replace(old, new))
PY

cd /home/adslab/pazika/ADSMPC-python/NssMPClib/native/suda_bridge
PYBIND11_DIR=$(/home/adslab/anaconda3/envs/nssmpc/bin/python -c 'import pybind11; print(pybind11.get_cmake_dir())')
SUDA_ROOT=/tmp/Suda PATH=/home/adslab/anaconda3/envs/nssmpc/bin:$PATH \
  /home/adslab/anaconda3/envs/nssmpc/bin/cmake -S . -B build \
  -DPython3_EXECUTABLE=/home/adslab/anaconda3/envs/nssmpc/bin/python \
  -Dpybind11_DIR="$PYBIND11_DIR" \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
PATH=/home/adslab/anaconda3/envs/nssmpc/bin:$PATH \
  /home/adslab/anaconda3/envs/nssmpc/bin/cmake --build build -j 8
```

The produced module is named `_suda_bridge`. Copy or symlink it into:

```text
/home/adslab/pazika/ADSMPC-python/NssMPClib/NssMPC/application/rag/pisces/
```

After that, Python can use the native bridge adapter:

```python
from NssMPC.application.rag.pisces import SudaNativeBridgeBackend

result = SudaNativeBridgeBackend().retrieve(indicators, database)
```

The Pisces RAG demo uses the split
`suda_native_pir_to_share_server` / `suda_native_pir_to_share_client` path so
the server receives an encrypted Suda query rather than plaintext top-k ids.

## Interface contract

Python passes a feature-major int64 matrix with shape:

```text
[feature_num, padded_database_size]
```

and zero-based query ids with shape:

```text
[native_batch_size]
```

The bridge pads the database size to a power of two, with a minimum of 4096
rows because Suda's packed encoder requires `host_n_data % 4096 == 0`. Query
ids are padded with distinct dummy row ids to a Suda-compatible batch size. It
returns Suda server/client additive shares as finite-field integers; the Python
layer reconstructs the plaintext records modulo Suda's field and cuts the
padded results back to the real selected rows.
