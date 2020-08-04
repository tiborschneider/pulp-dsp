/* =====================================================================
 * Project:      PULP DSP Library
 * Title:        plp_mat_copy_stride_i16p_xpulpv2.c
 * Description:  parallel 16-bit integer strided matrix copy for XPULPV2
 *
 * $Date:        17. July 2020
 * $Revision:    V0
 *
 * Target Processor: PULP cores
 * ===================================================================== */
/*
 * Copyright (C) 2020 ETH Zurich and Ubiversity of Bologna. All rights reserved.
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
  @ingroup MatCopyStride
 */

/**
  @addtogroup MatCopyStrideKernels
  @{
 */

/**
  @brief      Copy an MxN strided 16-bit integers matrix on XpulpV2 in parallel
  @param[in]  args  pointer to plp_mat_mat_copy_stride_instance_i16 struct initialized by
                    plp_mat_copy_stride_i16_parallel
  @return     none

  @par Exploiting SIMD instructions
  The 16 bit values are packed two each into 32 bit vectors and then the two dot products are
  performed on 32 bit vectors, with 32 bit accumulator.
*/

void plp_mat_copy_stride_i16p_xpulpv2(void *args) {

    int core_id = rt_core_id();

    plp_mat_copy_stride_instance_i16 *a = (plp_mat_copy_stride_instance_i16 *)args;

    const int16_t *__restrict__ pSrc = a->pSrc;
    uint32_t M = a->M;
    uint32_t N = a->N;
    uint32_t strideSrc = a->strideSrc;
    uint32_t strideDst = a->strideDst;
    uint32_t nPE = a->nPE;
    int16_t *__restrict__ pDst = a->pDst;

    unsigned int m;
    unsigned int n;

    pSrc += strideSrc * core_id;
    pDst += strideDst * core_id;

    unsigned int src_offset = (strideSrc * nPE) - N;
    unsigned int dst_offset = (strideDst * nPE) - N;

    unsigned int n_iter = N >> 2;
    unsigned int n_blk = N & 0b10; // Split the remaining into blk and rem
    unsigned int n_rem = N & 0b01;

    if (n_rem) {
        unsigned int src_offset = nPE * strideSrc - N + 1;
        unsigned int dst_offset = nPE * strideDst - N + 1;

        if (n_blk) {
            // n_rem == 1
            // n_blk == 1
            for (m = core_id; m < M; m += nPE) {
                // n_iter
                for (n = 0; n < n_iter; n++) {
                    *((int32_t *)pDst) = *((int32_t *)pSrc);
                    pDst += 2;
                    pSrc += 2;
                    *((int32_t *)pDst) = *((int32_t *)pSrc);
                    pDst += 2;
                    pSrc += 2;
                }
                // n_blk
                *((int32_t *)pDst) = *((int32_t *)pSrc);
                pDst += 2;
                pSrc += 2;
                // n_rem
                *pDst = *pSrc;
                // go to next line
                pSrc += src_offset;
                pDst += dst_offset;
            }
        } else {
            // n_rem == 1
            // n_blk == 0
            for (m = core_id; m < M; m += nPE) {
                // n_iter
                for (n = 0; n < n_iter; n++) {
                    *((int32_t *)pDst) = *((int32_t *)pSrc);
                    pDst += 2;
                    pSrc += 2;
                    *((int32_t *)pDst) = *((int32_t *)pSrc);
                    pDst += 2;
                    pSrc += 2;
                }
                // n_rem
                *pDst = *pSrc;
                // go to next line
                pSrc += src_offset;
                pDst += dst_offset;
            }
        }
    } else {
        unsigned int src_offset = nPE * strideSrc - N;
        unsigned int dst_offset = nPE * strideDst - N;

        if (n_blk) {
            // n_rem == 0
            // n_blk == 1
            for (m = core_id; m < M; m += nPE) {
                // n_iter
                for (n = 0; n < n_iter; n++) {
                    *((int32_t *)pDst) = *((int32_t *)pSrc);
                    pDst += 2;
                    pSrc += 2;
                    *((int32_t *)pDst) = *((int32_t *)pSrc);
                    pDst += 2;
                    pSrc += 2;
                }
                // n_blk
                *((int32_t *)pDst) = *((int32_t *)pSrc);
                pDst += 2;
                pSrc += 2;
                // go to next line
                pSrc += src_offset;
                pDst += dst_offset;
            }
        } else {
            // n_rem == 0
            // n_blk == 0
            for (m = core_id; m < M; m += nPE) {
                // n_iter
                for (n = 0; n < n_iter; n++) {
                    *((int32_t *)pDst) = *((int32_t *)pSrc);
                    pDst += 2;
                    pSrc += 2;
                    *((int32_t *)pDst) = *((int32_t *)pSrc);
                    pDst += 2;
                    pSrc += 2;
                }
                // go to next line
                pSrc += src_offset;
                pDst += dst_offset;
            }
        }
    }
}

/**
   @} end of MatCopyStrideKernels group
*/
