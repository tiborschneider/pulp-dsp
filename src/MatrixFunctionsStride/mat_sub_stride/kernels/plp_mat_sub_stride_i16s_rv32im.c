/* =====================================================================
 * Project:      PULP DSP Library
 * Title:        plp_mat_sub_stride_i16s_rv32im.c
 * Description:  16-bit strided matrix subtraction kernel for RV32IM
 *
 * $Date:        1. July 2020
 * $Revision:    V0
 *
 * Target Processor: PULP cores
 * ===================================================================== */
/*
 * Copyright (C) 2020 ETH Zurich and University of Bologna.
 *
 * Author: Tibor Schneider, ETH Zurich
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "plp_math.h"

/**
  @ingroup MatSubStride
 */

/**
  @defgroup MatSubStrideKernels Strided Matrix Subtraction Kernels
  This module contains the kernels for strided matrix subtraction.

  The Matrx Subtraction subtracts two matrices, element wise. Both matrices, and the output matrix
  have dimension MxN.

      `pDst[m, n] = pSrcA[m, n] - pSrcB[m, n]`

  For strided operations, the implementation is an optimized version of the following code:

      for (int m = 0; m < M; m++) {
          for (int n = 0; n < N; n++) {
              pDst[m * strideY + n] = pSrcA[m * strideA + n] + pSrcB[m * strideB + n];
          }
      }

  There are functions for integer 32- 16- and 8-bit data types, as well as for floating-point. These
  functions can also be used for fix-point matrices, if they have their fix-point at the same
  location. The outpt matrix will then also have the fix-point at the same location.

  The naming scheme of the functions follows the following pattern (for example
  `plp_mat_sub_stride_i32s_xpulpv2`):

      `plp_<function name>_<data type><precision><method>_<isa_extension>`

  name          | description
  ------------- | ---------------------------------------------------------
  function_name | `mat_sub_stride`
  data type     | {f, i, q} respectively for floats, integers, fixed points
  precision     | {32, 16, 8} bits
  method        | {`s`, `v`, `p`} meaning scalar, vectorized (i.e. SIMD) and parallel, respectively
  isa_extension | {`rv32im`, `xpulpv2`} respectively for ibex and riscy

  The `strideX` argument tells how many elements are in between the start of each row of the matrix.
  In other words, it is the width of the original matrix. @ref groupMatrixStride
 */

/**
  @addtogroup MatSubStrideKernels
  @{
 */

/**
  @brief strided matrix subtraction of 16-bit integer matrices kernel for RV32IM extension.
  @param[in]  pSrcA   Points to the first input matrix
  @param[in]  pSrcB   Points to the second input matrix
  @param[in]  M       Height of all matrices
  @param[in]  N       Width of all matrices
  @param[in]  strideA Stride of matrix A (elements between each row)
  @param[in]  strideB Stride of matrid B (elements between each row)
  @param[in]  strideY Stride of output matrix (elements between each row)
  @param[out] pDst    Points to the output matrix
  @return     none
 */

void plp_mat_sub_stride_i16s_rv32im(const int16_t *__restrict__ pSrcA,
                                    const int16_t *__restrict__ pSrcB,
                                    uint32_t M,
                                    uint32_t N,
                                    uint32_t strideA,
                                    uint32_t strideB,
                                    uint32_t strideY,
                                    int16_t *__restrict__ pDst) {

#ifdef PLP_MATH_LOOPUNROLL

    /*
     * The problem with strided matrix operations is that a row will not always start at a memory
     * aligned address. Therefore, loads might require multiple cycles, which causes load stalls
     * even though we apply loop unrolling. TODO: Fix this behavior.
     */

    uint32_t m, n; // loop counters

    unsigned int n_iter = N >> 1;
    unsigned int n_rem = N & 0x1;

    unsigned int step_a = strideA - N;
    unsigned int step_b = strideB - N;
    unsigned int step_y = strideY - N;

    if (n_rem) {
        for (m = 0; m < M; m++) {
            for (n = 0; n < n_iter; n++) {
                int16_t a1 = *pSrcA++;
                int16_t a2 = *pSrcA++;
                int16_t b1 = *pSrcB++;
                int16_t b2 = *pSrcB++;
                *pDst++ = a1 - b1;
                *pDst++ = a2 - b2;
            }
            *pDst++ = *pSrcA++ - *pSrcB++;
            // go to next line
            pSrcA += step_a;
            pSrcB += step_b;
            pDst += step_y;
        }
    } else {
        for (m = 0; m < M; m++) {
            for (n = 0; n < n_iter; n++) {
                int16_t a1 = *pSrcA++;
                int16_t a2 = *pSrcA++;
                int16_t b1 = *pSrcB++;
                int16_t b2 = *pSrcB++;
                *pDst++ = a1 - b1;
                *pDst++ = a2 - b2;
            }
            // go to next line
            pSrcA += step_a;
            pSrcB += step_b;
            pDst += step_y;
        }
    }

#else // PLP_MATH_LOOPUNROLL

    uint32_t m, n; // loop counters

    unsigned int step_a = strideA - N;
    unsigned int step_b = strideB - N;
    unsigned int step_y = strideY - N;

    for (m = 0; m < M; m++) {
        for (n = 0; n < N; n++) {
            int16_t a = *pSrcA++;
            int16_t b = *pSrcB++;
            *pDst++ = a - b;
        }
        pSrcA += step_a;
        pSrcB += step_b;
        pDst += step_y;
    }

#endif
}

/**
   @} end of MatSubStrideKernels group
*/
