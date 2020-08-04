/* =====================================================================
 * Project:      PULP DSP Library
 * Title:        plp_mat_copy_stride_i16s_xpulpv2.c
 * Description:  16-bit integer strided matrix copy for XPULPV2
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
  @brief      Copy an MxN strided 16-bit integers matrix on XpulpV2
  @param[in]  pSrc      Points to the input matrix of shape MxN
  @param[in]  M         Height of both matrices
  @param[in]  N         Width of both matrices
  @param[in]  strideSrc Stride of the input matrix (elements between each row)
  @param[in]  strideDst Stride of the output matrix (elements between each row)
  @param[out] pDst      Points to the output matrix of shape MxN
  @return     none
 */

void plp_mat_copy_stride_i16s_xpulpv2(const int16_t *__restrict__ pSrc,
                                      uint32_t M,
                                      uint32_t N,
                                      uint32_t strideSrc,
                                      uint32_t strideDst,
                                      int16_t *__restrict__ pDst) {

#ifdef PLP_MATH_LOOPUNROLL

    unsigned int m;
    unsigned int n;

    unsigned int n_iter = N >> 2;
    unsigned int n_blk = N & 0b10; // Split the remaining into blk and rem
    unsigned int n_rem = N & 0b01;

    if (n_rem) {
        unsigned int src_offset = strideSrc - N + 1;
        unsigned int dst_offset = strideDst - N + 1;

        if (n_blk) {
            // n_rem == 1
            // n_blk == 1
            for (m = 0; m < M; m++) {
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
            for (m = 0; m < M; m++) {
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
        unsigned int src_offset = strideSrc - N;
        unsigned int dst_offset = strideDst - N;

        if (n_blk) {
            // n_rem == 0
            // n_blk == 1
            for (m = 0; m < M; m++) {
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
            for (m = 0; m < M; m++) {
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

#else

    unsigned int m;
    unsigned int n;

    unsigned int n_iter = N >> 1;
    unsigned int n_rem = N & 0x1;

    if (n_rem) {
        // N is odd
        unsigned int src_offset = strideSrc - N + 1;
        unsigned int dst_offset = strideDst - N + 1;

        for (m = 0; m < M; m++) {
            for (n = 0; n < n_iter; n++) {
                *((int32_t *)pDst) = *((int32_t *)pSrc);
                pDst += 2;
                pSrc += 2;
            }
            *pDst = *pSrc;
            pSrc += src_offset;
            pDst += dst_offset;
        }
    } else {
        // N is even
        unsigned int src_offset = strideSrc - N;
        unsigned int dst_offset = strideDst - N;

        for (m = 0; m < M; m++) {
            for (n = 0; n < n_iter; n++) {
                *((int32_t *)pDst) = *((int32_t *)pSrc);
                pDst += 2;
                pSrc += 2;
            }
            pSrc += src_offset;
            pDst += dst_offset;
        }
    }

#endif
}
/**
   @} end of MatCopyStrideKernels group
*/
