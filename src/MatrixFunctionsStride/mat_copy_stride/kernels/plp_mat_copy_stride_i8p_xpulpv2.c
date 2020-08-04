/* =====================================================================
 * Project:      PULP DSP Library
 * Title:        plp_mat_copy_stride_i8p_xpulpv2.c
 * Description:  parallel 8-bit integer strided matrix copy for XPULPV2
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
  @brief      Copy an MxN strided 8-bit integers matrix on XpulpV2 in parallel
  @param[in]  args  pointer to plp_mat_mat_copy_stride_instance_i8 struct initialized by
                    plp_mat_copy_stride_i8_parallel
  @return     none

  @par Exploiting SIMD instructions
  The 8 bit values are packed four each into 32 bit vectors and then the four dot products are
  performed on 32 bit vectors, with 32 bit accumulator.
*/

void plp_mat_copy_stride_i8p_xpulpv2(void *args) {

    int core_id = rt_core_id();

    plp_mat_copy_stride_instance_i8 *a = (plp_mat_copy_stride_instance_i8 *)args;

    const int8_t *__restrict__ pSrc = a->pSrc;
    uint32_t M = a->M;
    uint32_t N = a->N;
    uint32_t strideSrc = a->strideSrc;
    uint32_t strideDst = a->strideDst;
    uint32_t nPE = a->nPE;
    int8_t *__restrict__ pDst = a->pDst;

    unsigned int m;
    unsigned int n;

    unsigned int n_iter = N >> 3;
    unsigned int n_rem = N & 0b011;
    unsigned int n_last_full = N & 0b100;

    pSrc = pSrc + strideSrc * core_id;
    pDst = pDst + strideDst * core_id;

    unsigned int src_offset = (strideSrc * nPE) - N;
    unsigned int dst_offset = (strideDst * nPE) - N;

    /*
     * There exists many cases:
     * - n_iter = 0
     *   - n_rem = 0 (implies n_last_full = 1)
     *   - n_rem > 0
     *     - n_last_full = 1
     *     - n_last_full = 0
     * - n_iter > 0
     *   - n_rem = 0
     *     - n_last_full = 0
     *     - n_last_full = 1
     *   - n_rem > 0
     *     - n_last_full = 0
     *     - n_last_full = 1
     *
     */

    if (n_iter == 0) {
        if (n_rem == 0) {
            if (n_last_full == 0) {
                // n_iter == 0
                // n_last_full == 0
                // n_rem == 0
                // TODO this case should not happen! Here, we don't do anyhing
            } else { // n_last_full == 1
                // n_iter == 0
                // n_last_full == 1
                // n_rem == 0
                for (m = core_id; m < M; m += nPE) {
                    *((int32_t *)pDst) = *((int32_t *)pSrc);
                    pDst += 4;
                    pSrc += 4;

                    pSrc += src_offset;
                    pDst += dst_offset;
                }
            }
        } else { // n_rem > 1
            if (n_last_full == 0) {
                // n_iter == 0
                // n_last_full == 0
                // n_rem >= 1
                for (m = core_id; m < M; m += nPE) {
                    for (n = 0; n < n_rem; n++) {
                        *pDst++ = *pSrc++;
                    }
                    pSrc += src_offset;
                    pDst += dst_offset;
                }
            } else { // n_last_full == 1
                // n_iter == 0
                // n_last_full == 1
                // n_rem >= 1
                for (m = core_id; m < M; m += nPE) {
                    *((int32_t *)pDst) = *((int32_t *)pSrc);
                    pDst += 4;
                    pSrc += 4;
                    for (n = 0; n < n_rem; n++) {
                        *pDst++ = *pSrc++;
                    }
                    pSrc += src_offset;
                    pDst += dst_offset;
                }
            }
        }
    } else { // n_iter > 0
        if (n_rem == 0) {
            if (n_last_full == 0) {
                // n_iter >= 1
                // n_last_full == 0
                // n_rem == 0
                for (m = core_id; m < M; m += nPE) {
                    for (n = 0; n < n_iter; n++) {
                        *((int32_t *)pDst) = *((int32_t *)pSrc);
                        pDst += 4;
                        pSrc += 4;
                        *((int32_t *)pDst) = *((int32_t *)pSrc);
                        pDst += 4;
                        pSrc += 4;
                    }
                    pSrc += src_offset;
                    pDst += dst_offset;
                }
            } else { // n_last_full == 1
                // n_iter >= 1
                // n_last_full == 1
                // n_rem == 0
                for (m = core_id; m < M; m += nPE) {
                    for (n = 0; n < n_iter; n++) {
                        *((int32_t *)pDst) = *((int32_t *)pSrc);
                        pDst += 4;
                        pSrc += 4;
                        *((int32_t *)pDst) = *((int32_t *)pSrc);
                        pDst += 4;
                        pSrc += 4;
                    }
                    *((int32_t *)pDst) = *((int32_t *)pSrc);
                    pDst += 4;
                    pSrc += 4;

                    pSrc += src_offset;
                    pDst += dst_offset;
                }
            }
        } else { // n_rem > 1
            if (n_last_full == 0) {
                // n_iter >= 1
                // n_last_full == 0
                // n_rem >= 1
                for (m = core_id; m < M; m += nPE) {
                    for (n = 0; n < n_iter; n++) {
                        *((int32_t *)pDst) = *((int32_t *)pSrc);
                        pDst += 4;
                        pSrc += 4;
                        *((int32_t *)pDst) = *((int32_t *)pSrc);
                        pDst += 4;
                        pSrc += 4;
                    }
                    for (n = 0; n < n_rem; n++) {
                        *pDst++ = *pSrc++;
                    }
                    pSrc += src_offset;
                    pDst += dst_offset;
                }
            } else { // n_last_full == 1
                // n_iter >= 1
                // n_last_full == 1
                // n_rem >= 1
                for (m = core_id; m < M; m += nPE) {
                    for (n = 0; n < n_iter; n++) {
                        *((int32_t *)pDst) = *((int32_t *)pSrc);
                        pDst += 4;
                        pSrc += 4;
                        *((int32_t *)pDst) = *((int32_t *)pSrc);
                        pDst += 4;
                        pSrc += 4;
                    }
                    *((int32_t *)pDst) = *((int32_t *)pSrc);
                    pDst += 4;
                    pSrc += 4;
                    for (n = 0; n < n_rem; n++) {
                        *pDst++ = *pSrc++;
                    }
                    pSrc += src_offset;
                    pDst += dst_offset;
                }
            }
        }
    }
}

/**
   @} end of MatCopyStrideKernels group
*/
