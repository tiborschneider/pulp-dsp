/* =====================================================================
 * Project:      PULP DSP Library
 * Title:        plp_mat_add_stride_i8p_xpulpv2.c
 * Description:  parallel 8-bit integer strided matrix addition for XPULPV2
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
  @brief Parallel strided matrix addition of 8-bit integer matrices kernel for XPULPV2 extension.
  @param[in]  args      pointer to plp_mat_add_stride_instance_i8 struct initialized by
  plp_mat_add_stride_i8_parallel
  @return     none

  @par Exploiting SIMD instructions
  The 8 bit values are packed four each into 32 bit vectors and then the four dot products are
  performed on 32 bit vectors, with 32 bit accumulator.
*/

void plp_mat_add_stride_i8p_xpulpv2(void *args) {

    int core_id = rt_core_id();

    plp_mat_add_stride_instance_i8 *a = (plp_mat_add_stride_instance_i8 *)args;

    const int8_t *__restrict__ pSrcA = a->pSrcA;
    const int8_t *__restrict__ pSrcB = a->pSrcB;
    uint32_t M = a->M;
    uint32_t N = a->N;
    uint32_t strideA = a->strideA;
    uint32_t strideB = a->strideB;
    uint32_t strideY = a->strideY;
    uint32_t nPE = a->nPE;
    int8_t *__restrict__ pDst = a->pDst;

    uint32_t m, n; // loop counters

    unsigned int n_iter = N >> 3;
    unsigned int n_rem = N & 0b011; // how many 1-byte additions to do
    unsigned int n_blk = N & 0b100; // how many not-unrolled SIMD additions to do?

    pSrcA += strideA * core_id;
    pSrcB += strideB * core_id;
    pDst += strideY * core_id;

    unsigned int step_a = strideA * nPE - N;
    unsigned int step_b = strideB * nPE - N;
    unsigned int step_y = strideY * nPE - N;

    if (n_rem) {
        if (n_blk) {
            // n_rem >= 1
            // n_blk == 1
            for (m = core_id; m < M; m += nPE) {
                for (n = 0; n < n_iter; n++) {
                    v4s a1 = *((v4s *)pSrcA);
                    v4s b1 = *((v4s *)pSrcB);
                    pSrcA += 4;
                    pSrcB += 4;
                    v4s a2 = *((v4s *)pSrcA);
                    v4s b2 = *((v4s *)pSrcB);
                    pSrcA += 4;
                    pSrcB += 4;
                    *((v4s *)pDst) = __ADD4(a1, b1);
                    pDst += 4;
                    *((v4s *)pDst) = __ADD4(a2, b2);
                    pDst += 4;
                }
                // n_blk == 1
                v4s a = *((v4s *)pSrcA);
                v4s b = *((v4s *)pSrcB);
                pSrcA += 4;
                pSrcB += 4;
                *((v4s *)pDst) = __ADD4(a, b);
                pDst += 4;
                // n_rem >= 1
                for (n = 0; n < n_rem; n++) {
                    *pDst++ = *pSrcA++ + *pSrcB++;
                }
                pSrcA += step_a;
                pSrcB += step_b;
                pDst += step_y;
            }
        } else {
            // n_rem >= 1
            // n_blk == 0
            for (m = core_id; m < M; m += nPE) {
                for (n = 0; n < n_iter; n++) {
                    v4s a1 = *((v4s *)pSrcA);
                    v4s b1 = *((v4s *)pSrcB);
                    pSrcA += 4;
                    pSrcB += 4;
                    v4s a2 = *((v4s *)pSrcA);
                    v4s b2 = *((v4s *)pSrcB);
                    pSrcA += 4;
                    pSrcB += 4;
                    *((v4s *)pDst) = __ADD4(a1, b1);
                    pDst += 4;
                    *((v4s *)pDst) = __ADD4(a2, b2);
                    pDst += 4;
                }
                // n_rem >= 1
                for (n = 0; n < n_rem; n++) {
                    *pDst++ = *pSrcA++ + *pSrcB++;
                }
                pSrcA += step_a;
                pSrcB += step_b;
                pDst += step_y;
            }
        }
    } else {
        if (n_blk) {
            // n_rem == 0
            // n_blk == 1
            for (m = core_id; m < M; m += nPE) {
                for (n = 0; n < n_iter; n++) {
                    v4s a1 = *((v4s *)pSrcA);
                    v4s b1 = *((v4s *)pSrcB);
                    pSrcA += 4;
                    pSrcB += 4;
                    v4s a2 = *((v4s *)pSrcA);
                    v4s b2 = *((v4s *)pSrcB);
                    pSrcA += 4;
                    pSrcB += 4;
                    *((v4s *)pDst) = __ADD4(a1, b1);
                    pDst += 4;
                    *((v4s *)pDst) = __ADD4(a2, b2);
                    pDst += 4;
                }
                // n_blk == 1
                v4s a = *((v4s *)pSrcA);
                v4s b = *((v4s *)pSrcB);
                pSrcA += 4;
                pSrcB += 4;
                *((v4s *)pDst) = __ADD4(a, b);
                pDst += 4;

                pSrcA += step_a;
                pSrcB += step_b;
                pDst += step_y;
            }
        } else {
            // n_rem == 0
            // n_blk == 0
            for (m = core_id; m < M; m += nPE) {
                for (n = 0; n < n_iter; n++) {
                    v4s a1 = *((v4s *)pSrcA);
                    v4s b1 = *((v4s *)pSrcB);
                    pSrcA += 4;
                    pSrcB += 4;
                    v4s a2 = *((v4s *)pSrcA);
                    v4s b2 = *((v4s *)pSrcB);
                    pSrcA += 4;
                    pSrcB += 4;
                    *((v4s *)pDst) = __ADD4(a1, b1);
                    pDst += 4;
                    *((v4s *)pDst) = __ADD4(a2, b2);
                    pDst += 4;
                }
                pSrcA += step_a;
                pSrcB += step_b;
                pDst += step_y;
            }
        }
    }
}

/**
   @} end of MatAddStrideKernels group
*/
