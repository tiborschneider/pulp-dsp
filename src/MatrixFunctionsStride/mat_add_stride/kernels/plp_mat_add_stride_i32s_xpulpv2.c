/* =====================================================================
 * Project:      PULP DSP Library
 * Title:        plp_mat_add_stride_i32s_xpulpv2.c
 * Description:  32-bit integer strided matrix addition for XPULPV2
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
  @ingroup MatAddStride
 */

/**
  @addtogroup MatAddStrideKernels
  @{
 */

/**
  @brief strided matrix addition of 32-bit integer matrices kernel for XPULPV2 extension.
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

void plp_mat_add_stride_i32s_xpulpv2(const int32_t *__restrict__ pSrcA,
                                     const int32_t *__restrict__ pSrcB,
                                     uint32_t M,
                                     uint32_t N,
                                     uint32_t strideA,
                                     uint32_t strideB,
                                     uint32_t strideY,
                                     int32_t *__restrict__ pDst) {

#ifdef PLP_MATH_LOOPUNROLL

    uint32_t m, n; // loop counters

    unsigned int n_iter = N >> 1;
    unsigned int n_rem = N & 0x1;

    unsigned int step_a = strideA - N;
    unsigned int step_b = strideB - N;
    unsigned int step_y = strideY - N;

    if (n_rem) {
        for (m = 0; m < M; m++) {
            for (n = 0; n < n_iter; n++) {
                int32_t a1 = *pSrcA++;
                int32_t a2 = *pSrcA++;
                int32_t b1 = *pSrcB++;
                int32_t b2 = *pSrcB++;
                *pDst++ = a1 + b1;
                *pDst++ = a2 + b2;
            }
            *pDst++ = *pSrcA++ + *pSrcB++;

            pSrcA += step_a;
            pSrcB += step_b;
            pDst += step_y;
        }
    } else {
        for (m = 0; m < M; m++) {
            for (n = 0; n < n_iter; n++) {
                int32_t a1 = *pSrcA++;
                int32_t a2 = *pSrcA++;
                int32_t b1 = *pSrcB++;
                int32_t b2 = *pSrcB++;
                *pDst++ = a1 + b1;
                *pDst++ = a2 + b2;
            }
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
            int32_t a = *pSrcA++;
            int32_t b = *pSrcB++;
            *pDst++ = a + b;
        }
        pSrcA += step_a;
        pSrcB += step_b;
        pDst += step_y;
    }

#endif
}

/**
   @} end of MatAddStrideKernels group
*/
