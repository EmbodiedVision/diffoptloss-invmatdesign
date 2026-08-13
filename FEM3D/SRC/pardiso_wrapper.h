/*
 * Copyright 2026 University of Augsburg, Intelligent Perception in Technical Systems Group
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Code author: Jens Kreber <jens.kreber@uni-a.de>
 */

#ifndef PardisoWrapper_h
#define PardisoWrapper_h

#include "mkl_types.h"

class PardisoState
{
private:
    MKL_INT maxfct = 1; // from doc: Maximum number of factors with identical sparsity structure that must be kept in memory at the same time. In most applications this value is equal to 1.
    MKL_INT mnum = 1;   // from doc: Indicates the actual matrix for the solution phase. With this scalar you can define which matrix to factorize. The value must be: 1 ≤mnum≤maxfct.
    MKL_INT mtype = 11; // matrix type: real nonsymmetric
    MKL_INT phase;      // will be set later, factorization etc.
    MKL_INT n;          // number of equations
    double *a;          // elements of A
    MKL_INT *ia;        // for CSR
    MKL_INT *ja;        // for CSR
    // perm not required, use dummy
    MKL_INT nrhs = 1;   // number of right-hand sides
    MKL_INT iparm[64];  // integer parameters. set on construction.
    MKL_INT msglvl = 0; // be verbose? no.
    // b provided on call
    void *pt[64];      // internal data pointers
    MKL_INT error = 0; // output value indicating error

    // dummy variables: will be passed to solver calls where the signature requires it, but without data.
    double emptyDouble;
    MKL_INT emptyInt;

public:
    PardisoState(MKL_INT n, MKL_INT *ia, MKL_INT *ja, double *a);
    int solve(double *b, double *x, bool release_memory = true);
    void release_mem();
    int solve_cached(double *new_b, double *new_x, bool release_memory = true);
};

int solve_pardiso(MKL_INT n, MKL_INT *ia, MKL_INT *ja, double *a, double *b, double *x);

#endif