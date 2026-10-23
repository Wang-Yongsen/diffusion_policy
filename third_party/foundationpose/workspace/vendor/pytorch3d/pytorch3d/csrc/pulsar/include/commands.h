/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef PULSAR_NATIVE_COMMANDS_ROUTING_H_
#define PULSAR_NATIVE_COMMANDS_ROUTING_H_

#include <cstring>
#include "../global.h"
#if __cplusplus >= 202002L
#include <bit>
#endif

// Commands available everywhere.
#define MALLOC_HOST(VAR, TYPE, SIZE) \
  VAR = static_cast<TYPE*>(malloc(sizeof(TYPE) * (SIZE)))
#define FREE_HOST(PTR) free(PTR)

/* Include command definitions depending on CPU or GPU use. */

#ifdef __CUDACC__
// TODO: find out which compiler we're using here and use the suppression.
// #pragma push
// #pragma diag_suppress = 68
#include <ATen/cuda/CUDAContext.h>
// #pragma pop
#include "../gpu/commands.h"
#else
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#pragma clang diagnostic pop
#include "../host/commands.h"
#endif

// Host + device int <-> float bit punning. Replaces IASF / FASI macros.
IHD float pulsar_int_as_float(int v) noexcept {
#ifdef __CUDA_ARCH__
  return __int_as_float(v);
#elif defined(__HIP_DEVICE_COMPILE__)
  return __int_as_float(v);
#else
#if __cpp_lib_bit_cast >= 201806L
  return std::bit_cast<float>(v);
#else
  float f;
  std::memcpy(&f, &v, sizeof(f));
  return f;
#endif
#endif
}

IHD int pulsar_float_as_int(float v) noexcept {
#ifdef __CUDA_ARCH__
  return __float_as_int(v);
#elif defined(__HIP_DEVICE_COMPILE__)
  return __float_as_int(v);
#else
#if __cpp_lib_bit_cast >= 201806L
  return std::bit_cast<int>(v);
#else
  int i;
  std::memcpy(&i, &v, sizeof(i));
  return i;
#endif
#endif
}

#endif
