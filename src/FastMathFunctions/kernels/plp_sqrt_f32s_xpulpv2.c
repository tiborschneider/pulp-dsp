/* =====================================================================
 * Project:      PULP DSP Library
 * Title:        plp_sqrt_f32s_xpulpv2.c
 * Description:  32-Bit floating point square root kernel
 *
 * $Date:        02.07.2020
 *
 * Target Processor: PULP cores
 * ===================================================================== */
/*
 * Copyright (C) 2020 ETH Zurich and University of Bologna.
 *
 * Author: Moritz Scherer, ETH Zurich
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
 *
 * Notice: project inspired by ARM CMSIS DSP and parts of source code
 * ported and adopted for RISC-V PULP platform from ARM CMSIS DSP
 * released under Copyright (C) 2010-2019 ARM Limited or its affiliates
 * with Apache-2.0.
 */

#define numIters 15
#include "plp_math.h"

/**
   @ingroup sqrt
*/

/**
   @defgroup sqrtKernels Sqrt Kernels
*/

/**
   @addtogroup sqrtKernels
   @{
*/

/**
   @brief         Square root of a 32-bit floating point number for XPULPV2 extension.
   @param[in]     pSrc       points to the input vector
   @param[out]    pRes    Square root returned here
   @return        none
*/

void plp_sqrt_f32s_xpulpv2(const float *__restrict__ pSrc, float *__restrict__ pRes) {

  float intermediate = 1.f / (2.f * (*pSrc));
  float half = *pSrc / 2;
  
  if (half > 0) {
    for (int i = 0; i < numIters; i++) {
      intermediate = intermediate * (1.5f - (intermediate * intermediate * half));
    }
    
    *pRes = intermediate * (*pSrc);
  } else {
    *pRes = 0.f;
  }
}
