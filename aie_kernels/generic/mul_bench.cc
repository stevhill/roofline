// SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "../aie_kernel_utils.h"

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#ifndef VEC_SIZE
#define VEC_SIZE 256
#endif

template <typename T_in, typename T_out> void eltwise_mul(T_in *a, T_out *c, int R, int size)
{
    const T_in b = static_cast<T_in>(1.001f);
    for (int i = 0; i < size; i++) {
        for (int r = 0; r < R; r++) {
            c[i] = a[i] * b;
        }
    }
}

template <typename T_in, typename T_out> void eltwise_vmul(T_in *a, T_out *c, int R, int size)
{
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    event0();
    AIE_PREPARE_FOR_PIPELINING
    for (int i = 0; i < size; i += VEC_SIZE) AIE_LOOP_FLATTEN {
        auto A = aie::load_v<VEC_SIZE>(a + i);
        aie::accum acc = aie::zeros<accfloat, VEC_SIZE>();;
        AIE_LOOP_MIN_ITERATION_COUNT(16)
        for (int r = 0; r < R; r++) {
            acc = aie::mac(acc, (bfloat16)1.5f, A);
        }
        aie::store_v(c + i, acc.template to_vector<T_out>());
    }
    event1();
}

template <typename T_in, typename T_out>
void eltwise_vmul2(T_in *a, T_out *c, int R, int N) {

  constexpr int vec_factor = 16;
  event0();
  T_in *__restrict pA1 = a;
  T_out *__restrict pC1 = c;
  aie::vector<T_in, vec_factor> B0 = aie::broadcast<T_in, vec_factor>(static_cast<T_in>(1.5f));
  const int F = N / vec_factor;
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(16)
  for (int i = 0; i < F; i++) {
    aie::vector<T_in, vec_factor> A0 = aie::load_v<vec_factor>(pA1);
    pA1 += vec_factor;
    aie::accum acc = aie::zeros<accfloat, vec_factor>();
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(16)
    for (int r = 0; r < R; r++) {
        acc = aie::mac(acc, A0, B0);
    }
    aie::store_v(pC1, acc.template to_vector<T_out>());
    pC1 += vec_factor;
  }
  event1();
}

extern "C" {

void eltwise_mul_bf16_scalar(bfloat16 *a_in, bfloat16 *c_out, int R, int size)
{
    eltwise_mul<bfloat16, bfloat16>(a_in, c_out, R, size);
}
void eltwise_mul_bf16_vector(bfloat16 *a_in, bfloat16 *c_out, int R, int size)
{
    eltwise_vmul<bfloat16, bfloat16>(a_in, c_out, R, size);
}
} // extern "C"
