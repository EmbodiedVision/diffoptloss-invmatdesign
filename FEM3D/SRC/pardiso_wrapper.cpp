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

#include <iostream>
#include <vector>
#include <math.h>

#include "pardiso_wrapper.h"
#include "mkl_pardiso.h"
#include "mkl_spblas.h"

using namespace std;

PardisoState::PardisoState(MKL_INT n, MKL_INT *ia, MKL_INT *ja, double *a) : n(n), ia(ia), ja(ja), a(a)
{
    for (int i = 0; i < 64; i++)
    {
        iparm[i] = 0; // set all to zero
        pt[i] = 0;    // set all to zero
    }

    // configure PARDISO parameters according to documentation
    iparm[0] = 1; // do NOT use default values
    iparm[1] = 2; // use METIS ordering
    iparm[2] = 0; // reserved
    iparm[3] = 0; // preconditioned CSG/CG: use default
    iparm[4] = 0; // no user permutation
    iparm[5] = 0; // write solution into x (instead of RHS b)
    // iparm[6] is an output value
    iparm[7] = 2;  // use max of 2 iterative refinement steps
    iparm[8] = 0;  // tolerance level for iterative refinement. default value.
    iparm[9] = 13; // pivoting perturbation. default value.
    iparm[10] = 1; // scaling vectors. default value (for unsymmetric matrices).
    iparm[11] = 0; // solve normal A X = B (no transposed A)
    iparm[12] = 1; // some matching algorithm. default value (for unsymmetric matrices).
    // iparm[13] is an output value
    // iparm[14] is an output value
    // iparm[15] is an output value
    // iparm[16] is an output value
    iparm[17] = -1; // < 0: enable reporting in this value (non-zero elements in factors)
    iparm[18] = -1; // < 0: enable reporting in this value (floating point ops)
    // iparm[19] is an output value
}

int PardisoState::solve(double *b, double *x, bool release_memory /*=true*/)
{
    phase = 11;
    PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &n, a, ia, ja, &emptyInt, &nrhs, iparm, &msglvl, &emptyDouble, &emptyDouble, &error);
    if (error != 0)
    {
        cout << "PARDISO: symbolic factorization failed!" << error << endl;
        exit(1);
    }

    phase = 22;
    PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &n, a, ia, ja, &emptyInt, &nrhs, iparm, &msglvl, &emptyDouble, &emptyDouble, &error);
    if (error != 0)
    {
        cout << "PARDISO: numerical factorization failed!" << error << endl;
        exit(1);
    }

    phase = 33;
    PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &n, a, ia, ja, &emptyInt, &nrhs, iparm, &msglvl, b, x, &error);
    if (error != 0)
    {
        cout << "PARDISO: solve failed!" << error << endl;
        exit(1);
    }

    // Calculate residual norm
    vector<double> y(n, 0);
    mkl_dcsrgemv("n", &n, a, ia, ja, x, &y[0]);

    double residual_norm = 0;
    for (int i = 0; i < n; i++)
    {
        residual_norm += (y[i] - b[i]) * (y[i] - b[i]);
    }
    residual_norm = sqrt(residual_norm);
    // cout << "Residual norm: " << residual_norm << endl;

    if (release_memory)
        release_mem();

    return 0;
}

void PardisoState::release_mem()
{
    phase = -1; /* Release internal memory. */
    PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &n, &emptyDouble, ia, ja, &emptyInt, &nrhs, iparm, &msglvl, &emptyDouble, &emptyDouble, &error);
}

int PardisoState::solve_cached(double *new_b, double *new_x, bool release_memory /*=true*/)
{
    phase = 33;
    PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &n, a, ia, ja, &emptyInt, &nrhs, iparm, &msglvl, new_b, new_x, &error);
    if (error != 0)
    {
        cout << "PARDISO: solve failed!" << error << endl;
        exit(1);
    }

    // Calculate residual norm
    vector<double> y(n, 0);
    mkl_dcsrgemv("n", &n, a, ia, ja, new_x, &y[0]);

    double residual_norm = 0;
    for (uint i = 0; i < n; i++)
    {
        residual_norm += (y[i] - new_b[i]) * (y[i] - new_b[i]);
    }
    residual_norm = sqrt(residual_norm);
    // cout << "cached solve residual norm: " << residual_norm << endl;

    if (release_memory)
        release_mem();

    return 0;
}

int solve_pardiso(MKL_INT n, MKL_INT *ia, MKL_INT *ja, double *a, double *b, double *x)
{
    PardisoState ps = PardisoState(n, ia, ja, a);
    return ps.solve(b, x);
}
