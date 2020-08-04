/* =====================================================================
 * Project:      PULP DSP Library
 * Title:        plp_mat_copy_stride_i16s_rv32im.c
 * Description:  16-bit strided matrix copy kernel for RV32IM
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
  @defgroup MatCopyStrideKernels Strided Matrix Copy Kernels

  The source and destination matrix can have different strides.

  There are functions for integer 32- 16- and 8-bit data types, as well as for floating-point. The
  naming scheme of the functions follows the following pattern (for example
  `plp_mat_copy_stride_i32s_xpulpv2`):

      `plp_<function name>_<data type><precision><method>_<isa_extension>`

  name          | description
  ------------- | ---------------------------------------------------------
  function_name | `mat_copy_stride`
  data type     | {f, i, q} respectively for floats, integers, fixed points
  precision     | {32, 16, 8} bits
  method        | {`s`, `v`, `p`} meaning scalar, vectorized (i.e. SIMD) and parallel, respectively
  isa_extension | {`rv32im`, `xpulpv2`} respectively for ibex and riscy

  The `strideSrc` and `strideDst` argument tells how many elements are in between the start of each
  row of the matrix. In other words, it is the width of the original matrix. @ref groupMatrixStride
 */

/**
  @addtogroup MatCopyStrideKernels
  @{
 */

/**
  @brief      Copy an MxN strided 16-bit integers matrix on RV32IM
  @param[in]  pSrc      Points to the input matrix of shape MxN
  @param[in]  M         Height of both matrices
  @param[in]  N         Width of both matrices
  @param[in]  strideSrc Stride of the input matrix (elements between each row)
  @param[in]  strideDst Stride of the output matrix (elements between each row)
  @param[out] pDst      Points to the output matrix of shape MxN
  @return     none
 */

void plp_mat_copy_stride_i16s_rv32im(const int16_t *__restrict__ pSrc,
                                     uint32_t M,
                                     uint32_t N,
                                     uint32_t strideSrc,
                                     uint32_t strideDst,
                                     int16_t *__restrict__ pDst) {

#ifdef PLP_MATH_LOOPUNROLL

    unsigned int m;
    unsigned int n;

    const int16_t *__restrict__ pSrc1 = pSrc;
    const int16_t *__restrict__ pSrc2 = pSrc + strideSrc;
    int16_t *__restrict__ pDst1 = pDst;
    int16_t *__restrict__ pDst2 = pDst + strideDst;

    unsigned int n_iter = N >> 1;
    unsigned int n_rem = N & 0x1;

    unsigned int m_iter = M >> 1;
    unsigned int m_rem = M & 0x1;

    if (n_rem) {
        // N is odd
        unsigned int src_offset = 2 * strideSrc - N + 1;
        unsigned int dst_offset = 2 * strideDst - N + 1;

        for (m = 0; m < m_iter; m++) {
            for (n = 0; n < n_iter; n++) {
                *((int32_t *)pDst1) = *((int32_t *)pSrc1);
                *((int32_t *)pDst2) = *((int32_t *)pSrc2);
                pSrc1 += 2;
                pSrc2 += 2;
                pDst1 += 2;
                pDst2 += 2;
            }
            *pDst1 = *pSrc1;
            *pDst2 = *pSrc2;

            pSrc1 += src_offset;
            pSrc2 += src_offset;
            pDst1 += dst_offset;
            pDst2 += dst_offset;
        }
        if (m_rem) {
            for (n = 0; n < n_iter; n++) {
                *((int32_t *)pDst1) = *((int32_t *)pSrc1);
                *((int32_t *)pDst2) = *((int32_t *)pSrc2);
                pSrc1 += 2;
                pSrc2 += 2;
                pDst1 += 2;
                pDst2 += 2;
            }
            *pDst1 = *pSrc1;
            *pDst2 = *pSrc2;
        }
    } else {
        // N is even
        unsigned int src_offset = 2 * strideSrc - N;
        unsigned int dst_offset = 2 * strideDst - N;

        for (m = 0; m < m_iter; m++) {
            for (n = 0; n < n_iter; n++) {
                *((int32_t *)pDst1) = *((int32_t *)pSrc1);
                *((int32_t *)pDst2) = *((int32_t *)pSrc2);
                pSrc1 += 2;
                pSrc2 += 2;
                pDst1 += 2;
                pDst2 += 2;
            }
            pSrc1 += src_offset;
            pSrc2 += src_offset;
            pDst1 += dst_offset;
            pDst2 += dst_offset;
        }
        if (m_rem) {
            for (n = 0; n < n_iter; n++) {
                *((int32_t *)pDst1) = *((int32_t *)pSrc1);
                *((int32_t *)pDst2) = *((int32_t *)pSrc2);
                pSrc1 += 2;
                pSrc2 += 2;
                pDst1 += 2;
                pDst2 += 2;
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
