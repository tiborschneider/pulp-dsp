/* =====================================================================
 * Project:      PULP DSP Library
 * Title:        plp_mat_add_stride_i16p_xpulpv2.c
 * Description:  parallel 16-bit integer strided matrix addition for XPULPV2
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
  @brief Parallel strided matrix addition of 16-bit integer matrices kernel for XPULPV2 extension.
  @param[in]  args      pointer to plp_mat_add_stride_instance_i16 struct initialized by
  plp_mat_add_stride_i16_parallel
  @return     none

  @par Exploiting SIMD instructions
  The 16 bit values are packed two each into 32 bit vectors and then the two dot products are
  performed on 32 bit vectors, with 32 bit accumulator.
*/

void plp_mat_add_stride_i16p_xpulpv2(void *args) {

    int core_id = rt_core_id();

    plp_mat_add_stride_instance_i16 *a = (plp_mat_add_stride_instance_i16 *)args;

    const int16_t *__restrict__ pSrcA = a->pSrcA;
    const int16_t *__restrict__ pSrcB = a->pSrcB;
    uint32_t M = a->M;
    uint32_t N = a->N;
    uint32_t strideA = a->strideA;
    uint32_t strideB = a->strideB;
    uint32_t strideY = a->strideY;
    uint32_t nPE = a->nPE;
    int16_t *__restrict__ pDst = a->pDst;

    uint32_t m, n; // loop counters

    unsigned int n_iter = N >> 2;
    unsigned int n_blk = N & 0b10;
    unsigned int n_rem = N & 0b01;

    pSrcA += strideA * core_id;
    pSrcB += strideB * core_id;
    pDst += strideY * core_id;

    unsigned int step_a = strideA * nPE - N;
    unsigned int step_b = strideB * nPE - N;
    unsigned int step_y = strideY * nPE - N;

    if (n_rem) {
        if (n_blk) {
            // n_rem == 1
            // n_blk == 1
            for (m = core_id; m < M; m += nPE) {
                for (n = 0; n < n_iter; n++) {
                    v2s a1 = *((v2s *)pSrcA);
                    v2s b1 = *((v2s *)pSrcB);
                    v2s a2 = *((v2s *)(pSrcA + 2));
                    v2s b2 = *((v2s *)(pSrcB + 2));
                    *((v2s *)pDst) = __ADD2(a1, b1);
                    *((v2s *)(pDst + 2)) = __ADD2(a2, b2);
                    pSrcA += 4;
                    pSrcB += 4;
                    pDst += 4;
                }
                // handle blk
                v2s a = *((v2s *)pSrcA);
                v2s b = *((v2s *)pSrcB);
                *((v2s *)pDst) = __ADD2(a, b);
                pSrcA += 2;
                pSrcB += 2;
                pDst += 2;
                // handle rem
                *pDst++ = *pSrcA++ + *pSrcB++;
                // go to next line
                pSrcA += step_a;
                pSrcB += step_b;
                pDst += step_y;
            }
        } else {
            // n_rem == 1
            // n_blk == 0
            for (m = core_id; m < M; m += nPE) {
                for (n = 0; n < n_iter; n++) {
                    v2s a1 = *((v2s *)pSrcA);
                    v2s b1 = *((v2s *)pSrcB);
                    v2s a2 = *((v2s *)(pSrcA + 2));
                    v2s b2 = *((v2s *)(pSrcB + 2));
                    *((v2s *)pDst) = __ADD2(a1, b1);
                    *((v2s *)(pDst + 2)) = __ADD2(a2, b2);
                    pSrcA += 4;
                    pSrcB += 4;
                    pDst += 4;
                }
                // handle rem
                *pDst++ = *pSrcA++ + *pSrcB++;
                // go to next line
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
                    v2s a1 = *((v2s *)pSrcA);
                    v2s b1 = *((v2s *)pSrcB);
                    v2s a2 = *((v2s *)(pSrcA + 2));
                    v2s b2 = *((v2s *)(pSrcB + 2));
                    *((v2s *)pDst) = __ADD2(a1, b1);
                    *((v2s *)(pDst + 2)) = __ADD2(a2, b2);
                    pSrcA += 4;
                    pSrcB += 4;
                    pDst += 4;
                }
                // handle blk
                v2s a = *((v2s *)pSrcA);
                v2s b = *((v2s *)pSrcB);
                *((v2s *)pDst) = __ADD2(a, b);
                pSrcA += 2;
                pSrcB += 2;
                pDst += 2;
                // go to next line
                pSrcA += step_a;
                pSrcB += step_b;
                pDst += step_y;
            }
        } else {
            // n_rem == 0
            // n_blk == 0
            for (m = core_id; m < M; m += nPE) {
                for (n = 0; n < n_iter; n++) {
                    v2s a1 = *((v2s *)pSrcA);
                    v2s b1 = *((v2s *)pSrcB);
                    v2s a2 = *((v2s *)(pSrcA + 2));
                    v2s b2 = *((v2s *)(pSrcB + 2));
                    *((v2s *)pDst) = __ADD2(a1, b1);
                    *((v2s *)(pDst + 2)) = __ADD2(a2, b2);
                    pSrcA += 4;
                    pSrcB += 4;
                    pDst += 4;
                }
                // go to next line
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
