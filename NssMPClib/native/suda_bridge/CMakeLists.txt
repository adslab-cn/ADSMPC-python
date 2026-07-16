cmake_minimum_required(VERSION 3.18)
project(nssmpc_suda_bridge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED SUDA_ROOT)
  set(SUDA_ROOT "$ENV{SUDA_ROOT}")
endif()
if(NOT SUDA_ROOT)
  message(FATAL_ERROR "Set SUDA_ROOT to a checkout of https://github.com/sls33/Suda")
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

find_library(NTL_LIB NAMES ntl PATHS "${SUDA_ROOT}/src/third_party/ntl/src/.libs" NO_DEFAULT_PATH REQUIRED)
find_library(SEAL_LIB NAMES seal-4.1 PATHS
  "${SUDA_ROOT}/src/third_party/SEAL/seal_install/lib"
  "${SUDA_ROOT}/src/third_party/SEAL/build/lib"
  NO_DEFAULT_PATH
  REQUIRED
)
find_package(Threads REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(IPCL CONFIG REQUIRED PATHS
  "${SUDA_ROOT}/src/third_party/pailliercryptolib/ext_lib/lib/cmake/ipcl-2.0.0"
  NO_DEFAULT_PATH
)
find_library(IPPCP_LIB NAMES ippcp PATHS
  "${SUDA_ROOT}/src/third_party/pailliercryptolib/ext_lib/lib/intel64"
  "${SUDA_ROOT}/src/third_party/pailliercryptolib/build/ext_ipp-crypto/ippcrypto_install${SUDA_ROOT}/src/third_party/pailliercryptolib/ext_lib/lib/intel64"
  "${SUDA_ROOT}/src/third_party/pailliercryptolib/build/ext_ipp-crypto/src/ext_ipp-crypto-build/.build/RELEASE/lib"
  NO_DEFAULT_PATH
  REQUIRED
)
find_library(CRYPTO_MB_LIB NAMES crypto_mb PATHS
  "${SUDA_ROOT}/src/third_party/pailliercryptolib/ext_lib/lib/intel64"
  "${SUDA_ROOT}/src/third_party/pailliercryptolib/build/ext_ipp-crypto/ippcrypto_install${SUDA_ROOT}/src/third_party/pailliercryptolib/ext_lib/lib/intel64"
  "${SUDA_ROOT}/src/third_party/pailliercryptolib/build/ext_ipp-crypto/src/ext_ipp-crypto-build/.build/RELEASE/lib"
  NO_DEFAULT_PATH
  REQUIRED
)

pybind11_add_module(_suda_bridge suda_bridge_pybind.cpp)
target_include_directories(_suda_bridge PRIVATE
  "${SUDA_ROOT}/src"
  "${SUDA_ROOT}/src/include"
  "${SUDA_ROOT}/src/third_party/ntl/include"
  "${SUDA_ROOT}/src/third_party/SEAL/seal_install/include/SEAL-4.1"
  "${SUDA_ROOT}/src/third_party/pailliercryptolib/ext_lib/include/ipcl"
  "${SUDA_ROOT}/src/third_party/pailliercryptolib/ext_lib/include/ipcl/ippcrypto"
)
target_compile_options(_suda_bridge PRIVATE
  -O3
  -Wno-ignored-attributes
  -Wno-int-in-bool-context
  -Wno-sign-compare
  -Wno-catch-value
)
target_link_libraries(_suda_bridge PRIVATE
  ${NTL_LIB}
  ${SEAL_LIB}
  IPCL::ipcl
  ${IPPCP_LIB}
  ${CRYPTO_MB_LIB}
  OpenSSL::SSL
  OpenSSL::Crypto
  Threads::Threads
)

set_target_properties(_suda_bridge PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/NssMPC/application/rag/pisces"
)
