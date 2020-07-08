/* =====================================================================
 * Project:      PULP DSP Library
 * Title:        plp_mat_mult_i16vp_xpulpv2.c
 * Description:  parallel 16-bit integer matrix multiplication for XPULPV2
 *
 * $Date:        22. December 2019
 * $Revision:    V0
 *
 * Target Processor: PULP cores
 * ===================================================================== */
/*
 * Copyright (C) 2019 ETH Zurich and Ubiversity of Bologna. All rights reserved.
 *
 * Author: Tom Kuchler, ETH Zurich
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
  @ingroup MatMultTrans
 */

/**
  @addtogroup MatMultTransKernels
  @{
 */

/**
   @brief         Parallel matrix transposed matrix multiplication of a 16-bit integer matrices for XPULPV2 extension.
   @param[in]  args      pointer to plp_mat_mult_instance_i16 struct initialized by plp_mat_mult_i16_parallel
   @return        none

   @par Exploiting SIMD instructions
   The 16 bit values are packed two each into 32 bit vectors and then the two dot products are performed on 32 bit vectors, with 32 bit accumulator.
*/

void plp_mat_mult_trans_i16vp_xpulpv2(void* args) {

    int core_id = rt_core_id();

    plp_mat_mult_instance_i16* arguments = (plp_mat_mult_instance_i16*) args;
    const int16_t * __restrict__ pSrcA = arguments->pSrcA;
    const int16_t * __restrict__ pSrcB = arguments->pSrcB;
    uint32_t M = arguments->M;
    uint32_t N = arguments->N;
    uint32_t O = arguments->O;
    uint32_t nPE = arguments->nPE;
    int32_t * __restrict__ pDstC = arguments->pDstC;

#define BASIC_VERSION // if used don't forget to also use the undefine at end of file
#ifdef BASIC_VERSION

    uint32_t m; // loop counter for M
    uint32_t n; // loop counter for N
    uint32_t o; // loop counter for O
        
    for(m = core_id; m < M; m += nPE){
        for(o = 0; o < O; o++){
            int32_t sum = 0;
            for(n = 0; n < N; n++){
                sum = sum + pSrcA[m*N + n]*pSrcB[o*N + n];
            }
            pDstC[m*O + o] = sum;
        }
    }

    rt_team_barrier();

#else 

    // TODO hackathon

#endif
#undef BASIC_VERSION

}
/**
   @} end of MatMultTransKernels group
*/
