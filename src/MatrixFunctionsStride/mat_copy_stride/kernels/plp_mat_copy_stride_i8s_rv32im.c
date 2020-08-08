/* =====================================================================
 * Project:      PULP DSP Library
 * Title:        plp_mat_copy_stride_i8s_rv32im.c
 * Description:  8-bit strided matrix copy kernel for RV32IM
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
  @brief      Copy an MxN strided 8-bit integers matrix on RV32IM
  @param[in]  pSrc      Points to the input matrix of shape MxN
  @param[in]  M         Height of both matrices
  @param[in]  N         Width of both matrices
  @param[in]  strideSrc Stride of the input matrix (elements between each row)
  @param[in]  strideDst Stride of the output matrix (elements between each row)
  @param[out] pDst      Points to the output matrix of shape MxN
  @return     none
 */

void plp_mat_copy_stride_i8s_rv32im(const int8_t *__restrict__ pSrc,
                                    uint32_t M,
                                    uint32_t N,
                                    uint32_t strideSrc,
                                    uint32_t strideDst,
                                    int8_t *__restrict__ pDst) {

#ifdef PLP_MATH_LOOPUNROLL

    unsigned int m;
    unsigned int n;

    const int8_t *__restrict__ pSrc1 = pSrc;
    const int8_t *__restrict__ pSrc2 = pSrc + strideSrc;
    int8_t *__restrict__ pDst1 = pDst;
    int8_t *__restrict__ pDst2 = pDst + strideDst;

    unsigned int offset_src = 2 * strideSrc - N;
    unsigned int offset_dst = 2 * strideDst - N;

    unsigned int m_iter = M >> 1;
    unsigned int m_rem = M & 0x1;

    unsigned int n_iter = N >> 2;
    unsigned int n_rem = N & 0x3;

    if (n_rem) {
        for (m = 0; m < m_iter; m++) {
            for (n = 0; n < n_iter; n++) {
                *((int32_t *)pDst1) = *((int32_t *)pSrc1);
                *((int32_t *)pDst2) = *((int32_t *)pSrc2);
                pSrc1 += 4;
                pSrc2 += 4;
                pDst1 += 4;
                pDst2 += 4;
            }
            for (n = 0; n < n_rem; n++) {
                *pDst1++ = *pSrc1++;
                *pDst2++ = *pSrc2++;
            }
            pSrc1 += offset_src;
            pSrc2 += offset_src;
            pDst1 += offset_dst;
            pDst2 += offset_dst;
        }
        if (m_rem) {
            for (n = 0; n < n_iter; n++) {
                *((int32_t *)pDst1) = *((int32_t *)pSrc1);
                pSrc1 += 4;
                pDst1 += 4;
            }
            for (n = 0; n < n_rem; n++) {
                *pDst1++ = *pSrc1++;
            }
        }
    } else {
        for (m = 0; m < m_iter; m++) {
            for (n = 0; n < n_iter; n++) {
                *((int32_t *)pDst1) = *((int32_t *)pSrc1);
                *((int32_t *)pDst2) = *((int32_t *)pSrc2);
                pSrc1 += 4;
                pSrc2 += 4;
                pDst1 += 4;
                pDst2 += 4;
            }
            pSrc1 += offset_src;
            pSrc2 += offset_src;
            pDst1 += offset_dst;
            pDst2 += offset_dst;
        }
        if (m_rem) {
            for (n = 0; n < n_iter; n++) {
                *((int32_t *)pDst1) = *((int32_t *)pSrc1);
                pSrc1 += 4;
                pDst1 += 4;
            }
        }
    }

#else

    // No Loop Unroll
    unsigned int m;
    unsigned int n;

    unsigned int offset_src = strideSrc - N;
    unsigned int offset_dst = strideDst - N;

    unsigned int n_iter = N >> 2;
    unsigned int n_rem = N & 0x3;

    if (n_rem) {
        for (m = 0; m < M; m++) {
            for (n = 0; n < n_iter; n++) {
                *((int32_t *)pDst) = *((int32_t *)pSrc);
                pSrc += 4;
                pDst += 4;
            }
            for (n = 0; n < n_rem; n++) {
                *pDst++ = *pSrc++;
            }
            pSrc += offset_src;
            pDst += offset_dst;
        }
    } else {
        for (m = 0; m < M; m++) {
            for (n = 0; n < n_iter; n++) {
                *((int32_t *)pDst) = *((int32_t *)pSrc);
                pSrc += 4;
                pDst += 4;
            }
            pSrc += offset_src;
            pDst += offset_dst;
        }
    }

#endif
}

/**
   @} end of MatCopyStrideKernels group
*/
