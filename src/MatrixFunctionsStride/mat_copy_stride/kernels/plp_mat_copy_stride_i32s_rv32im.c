/* =====================================================================
 * Project:      PULP DSP Library
 * Title:        plp_mat_copy_stride_i32s_rv32im.c
 * Description:  32-bit strided matrix copy kernel for RV32IM
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
  @brief      Copy an MxN strided 32-bit integers matrix on RV32IM
  @param[in]  pSrc      Points to the input matrix of shape MxN
  @param[in]  M         Height of both matrices
  @param[in]  N         Width of both matrices
  @param[in]  strideSrc Stride of the input matrix (elements between each row)
  @param[in]  strideDst Stride of the output matrix (elements between each row)
  @param[out] pDst      Points to the output matrix of shape MxN
  @return     none
 */

void plp_mat_copy_stride_i32s_rv32im(const int32_t *__restrict__ pSrc,
                                     uint32_t M,
                                     uint32_t N,
                                     uint32_t strideSrc,
                                     uint32_t strideDst,
                                     int32_t *__restrict__ pDst) {

#ifdef PLP_MATH_LOOPUNROLL

    unsigned int m;
    unsigned int n;

    unsigned int n_iter = N >> 1;
    unsigned int n_rem = N & 0x1;

    for (m = 0; m < M; m++) {
        for (n = 0; n < n_iter; n++) {
            *pDst++ = *pSrc++;
            *pDst++ = *pSrc++;
        }
        if (n_rem) {
            *pDst++ = *pSrc++;
        }
        pSrc += strideSrc - N;
        pDst += strideDst - N;
    }

#else // PLP_MATH_LOOPUNROLL

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            pDst[m * strideDst + n] = pSrc[m * strideSrc + n];
        }
    }

#endif
}

/**
   @} end of MatCopyStrideKernels group
*/
