/*
 * Copyright 2026 University of Augsburg, Intelligent Perception in Technical Systems Group
 * Copyright 2022-2026 Christian Weißenfels
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
 * Main code author is Christian Weißenfels. Functionality for using the solver iteratively, combining the partial derivatives for gradient computation
 * and final transformation of the total derivatives added by Jens Kreber.
 */

// Includes
#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <stdlib.h>
#include <iomanip>
#include "AceGen.h"
#include "Material.h"
#include "Node.h"
#include "Element.h"
#include "Solution.h"
#include "Lagrange.h"
#include "pardiso_wrapper.h"
#include <cassert>

#include <mkl_spblas.h>
#include <optional>
#include <chrono>
#include <map>
#include <mkl.h>

#define PI 3.141592653589793238462643383279
#define one3 0.3333333333333

using namespace std;

extern string filename;
extern double Time;
extern int t;
extern unsigned int ndf, neq, numnp, numel;

extern vector<Material> material;
extern vector<Node> node;
extern vector<Element> element;
extern vector<Lagrange> lagrange;
extern AceGen acegen;

// Jens: added this function
void Solution::print_dense_K()
{
    vector<vector<double>> K(neq, vector<double>(neq, 0));

    unsigned int currentEntry = 0;
    for (unsigned int i_row = 0; i_row < ia.size() - 1; i_row++)
    {
        unsigned int NumberEntriesRow = ia[i_row + 1] - ia[i_row];
        for (unsigned int i_entry_this_row = 0; i_entry_this_row < NumberEntriesRow; i_entry_this_row++)
        {
            unsigned int Column = ja[currentEntry];
            K[i_row][Column - 1] = a[currentEntry];
            currentEntry++;
        }
    }
    cout << "K matrix with " << currentEntry << "non-zero entries:" << endl;
    for (uint i_row = 0; i_row < neq; i_row++)
    {
        for (uint i_col = 0; i_col < neq; i_col++)
        {
            cout << " " << K[i_row][i_col];
        }
        cout << endl;
    }
    cout << endl;
}

double Solution::prop(string type, int n, int dir, double t)
{

    double pp = 0;

    if (type == "Shockwave")
    {

        double bulk = material[0].matdata[0];
        double nu = material[0].matdata[1];
        double rho = material[0].matdata[2];
        double jp = material[0].matdata[3];
        double jm = material[0].matdata[4];

        double v = sqrt(bulk * 0.5 / rho * (1.0 + 1.0 / (jp * jm)));
        double l = 8 * nu * v / (3 * bulk) * jp * jm / (jp - jm);

        double xi = node[n].icoor[2] - v * t;
        double ch = cosh(0.5 * xi / l);

        pp = ((jp + jm) / 2.0 - 1.0) * xi + (jp - jm) * l * (log(ch) - log(2.0));
    }
    else if (type == "LinearProp")
    {

        // Proportional number

        int pn = -1;
        if (dir == 0)
        {
            pn = node[n].bprop[0] - 1;
        }
        else if (dir == 1)
        {
            pn = node[n].bprop[1] - 1;
        }
        else if (dir == 2)
        {
            pn = node[n].bprop[2] - 1;
        }

        pp = 0.0;
        if (pn != -1)
        {

            double t_0 = table[pn][0];
            double lambda_0 = table[pn][1];
            double t_max = table[pn][2];
            double lambda_max = table[pn][3];

            double clime = (lambda_max - lambda_0) / (t_max - t_0);

            pp = lambda_0 + clime * t;
        }
    }

    return pp;
}

// added bj Jens
double Solution::solve_iteratively(MKL_INT &nq, vector<double> *the_x, vector<double> *the_b, vector<double> *the_precond, bool calc_precond, int *it_count, bool zero_x, double abs_tol, double rel_tol)
{
    // iterative solve
    sparse_matrix_t A;
    // Jens Note: Row and column indices for PARDISO start with 1! This means there is a 1 at first position in ia.
    mkl_sparse_d_create_csr(&A, SPARSE_INDEX_BASE_ONE, nq, nq, ia.data(), ia.data() + 1, ja.data(), a.data()); // weird with the rows_end..

    // https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2025-1/fgmres-interface-description.html
    if (zero_x)
    {
        for (uint i = 0; i < the_x->size(); i++)
        {
            x[i] = 0; // set to zero
        }
    }

    // Initialize the FGMRES solver
    dfgmres_init(&nq, the_x->data(), the_b->data(), &slv_rci_request, slv_ipar, slv_dpar, nullptr); //     https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2025-1/dfgmres-init.html

    if (slv_rci_request != 0)
    {
        std::cerr << "Error initializing FGMRES" << std::endl;
        assert(false);
    }

    int ipar14 = slv_ipar[14];                                                    // can also compute myself
    int temp_size = (2 * ipar14 + 1) * nq + ipar14 * (ipar14 + 9) / 2 + 1 + 1000; // should remove eventually
    vector<double> tmp(temp_size);

    // -- preconditioner --
    MKL_INT ierr;
    if (calc_precond)
    {
        the_precond->resize(a.size());
        dcsrilu0(&nq, a.data(), ia.data(), ja.data(), the_precond->data(), slv_ipar, slv_dpar, &ierr);
        if (ierr == 0)
        {
            // cout << "preconditioner ok." << endl;
        }
        else
        {
            cout << "preconditioner error: " << ierr << endl;
            assert(false);
        }
    }

    slv_ipar[4] = slv_max_iter;
    assert(slv_ipar[7] == 1);  // check max number of iterations stopping criterion
    slv_ipar[8] = 1;           // also check residual stoppping test: dpar[4] <= dpar[3]
    slv_ipar[9] = 0;           // do NOT perform user-defined stopping test // lol like what would I even do XD
    slv_ipar[10] = 1;          // preconditioning?
    slv_ipar[11] = 1;          // DO check zero norm of currently generated vector: dpar[6] <= dpar[7]
    assert(slv_ipar[12] == 0); // write solution to x (instead of b), but can only access once AFTER computation!

    slv_dpar[0] = rel_tol; // relative tolerance, default 1e-6. I had set it to 1e-8. Compared to initial residual apparently.
    slv_dpar[1] = 0;       // absollute tolerance, default 0.
    // first, run with default relative target if abs == 0. Then, we take the final residual as absolute tolerance for downstream iterations.
    if (abs_tol != 0)
    {
        slv_dpar[1] = abs_tol;
        // cout << "abs tol high, using this abs from last" << endl;
    }
    else
    {
        // cout << "abs tol 0, using default rel!" << endl;
    }

    dfgmres_check(&nq, the_x->data(), the_b->data(), &slv_rci_request, slv_ipar, slv_dpar, tmp.data()); // https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2025-1/dfgmres-check.html
    if (slv_rci_request != 0)
    {
        std::cerr << "Error checking FGMRES" << std::endl;
        assert(false);
    }
    else
    {
        // cout << "Check ok." << endl;
    }

    while (true)
    {
        dfgmres(&nq, the_x->data(), the_b->data(), &slv_rci_request, slv_ipar, slv_dpar, tmp.data()); // https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2025-1/dfgmres.html

        if (slv_rci_request == 0)
            break; // Solution found

        if (slv_rci_request == 1)
        {
            // Perform sparse matrix-vector multiplication
            double *mult_with = &tmp[slv_ipar[21] - 1];
            double *mult_to = &tmp[slv_ipar[22] - 1];
            matrix_descr descr = {SPARSE_MATRIX_TYPE_GENERAL};
            mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1., A, descr, mult_with, 0., mult_to); // https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2025-1/mkl-sparse-mv.html
        }
        else if (slv_rci_request == 2)
        {
            assert(false); // stop criterion. auto-check
        }
        else if (slv_rci_request == 3)
        { // preconditioner
            double *mult_with = &tmp[slv_ipar[21] - 1];
            double intermediate[nq];
            double *mult_to = &tmp[slv_ipar[22] - 1];
            mkl_dcsrtrsv("L", "N", "U", &nq, the_precond->data(), ia.data(), ja.data(), mult_with, intermediate); // x is RHS!
            mkl_dcsrtrsv("U", "N", "N", &nq, the_precond->data(), ia.data(), ja.data(), intermediate, mult_to);
        }
        else if (slv_rci_request == 4)
        {
            assert(false);
        }
        else if (slv_rci_request == -1)
        {
            // cerr << "RCI: reached maximum number of iterations, but relative stopping not met! Still exiting, hope this is fine..." << endl;
            cerr << "°";
            // assert(false);
            slv_ipar[14] += 1; // allow one more iteration..
            break;
        }
        else if (slv_rci_request == -10)
        {
            cerr << "RCI: interrupted bc of divide by zero! Is your matrix degenerate? Or did you not stop the method after solution is found?" << endl;
            assert(false);
        }
        else if (slv_rci_request == -11)
        {
            cerr << "RCI: Infinite cycle! Did you set illegal parameters?" << endl;
            assert(false);
        }
        else if (slv_rci_request == -12)
        {
            cerr << "RCI: Illegal parameters! Did you alter them?" << endl;
            assert(false);
        }
        else
        {
            std::cerr << "Unexpected RCI request: " << slv_rci_request << std::endl;
            assert(false);
        }
    }

    double final_residual = slv_dpar[4];
    // cout << "Final residual: " << slv_dpar[4] << endl;

    dfgmres_get(&nq, the_x->data(), the_b->data(), &slv_rci_request, slv_ipar, slv_dpar, tmp.data(), &slv_itercount); // https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2025-1/dfgmres-get.html

    *it_count = slv_itercount;
    // cout << "Solved after " << slv_itercount << "iterations!" << endl;
    // cout << "sol " << (*the_x)[0] << " " << (*the_x)[1] << " " << (*the_x)[2]  << endl;

    mkl_sparse_destroy(A);

    return final_residual;
}

void Solution::Solve(unsigned int isw)
{
    if (isw == 3)
    {

        MKL_INT nq = b.size(); // MKL_INT kp=ja.size();

        // Direct solve
        if (!use_iterative_solver)
        {
            solve_pardiso(nq, ia.data(), ja.data(), a.data(), b.data(), x.data());
        }

        if (use_iterative_solver)
        {
            double abs_tol_ = abs_tol;
            if (abs_tol < 0)
            {
                assert(rel_tol > 0);
                abs_tol_ = warmstart ? forward_residual : 0; // use last residual from first step as guidance.
            }

            double last_residual = solve_iteratively(nq, &x, &b, &bilu0_forward, !warmstart, &last_forward_iterations, false, abs_tol_, rel_tol); // do NOT set x to zero
            last_backward_iterations = 0;
            if (!warmstart)
            {
                // cout << "setting residual to " << last_residual << endl;
                forward_residual = last_residual;
            }
        }
    }
    else
    {
        for (unsigned int i = 0; i < x.size(); i++)
        {
            x[i] = b[i] / a[i];
        }
    }
}

void Solution::NewtonIteration(unsigned int &it, unsigned int neq, double &aengy0)
{
    ofstream output(filename.c_str(), ios::app);

    double rnorm = 0.0;
    double aengy = 0.0;

    for (unsigned int i = 0; i < neq; i++)
    {
        rnorm += b[i] * b[i];
        aengy += b[i] * x[i];
    }
    rnorm = sqrt(rnorm);
    if (aengy0 == 0.0)
        aengy0 = aengy;

    // Convergence criterion

    output << "\tIteration: " << setw(2) << it << endl;
    output << "\tResidual norm =\t" << scientific << rnorm << endl;
    output << "\tEnergy convergence test" << endl;
    output << "\t  Current  =\t" << aengy << "\tMaximum   = \t" << aengy0 << endl;
    output << "\t  Relative =\t" << aengy / aengy0 << "\tTolerance = \t" << 1e-16 << endl;
    // cout << " newton Iteration: rel =" << aengy/aengy0;

    if (fabs(aengy / aengy0) < 1e-16)
        it = 50;
    if (time == 0)
        it = 50;
}
void Solution::NodalAssembly(unsigned int nI, unsigned int ndf, vector<double> Force, vector<vector<double>> Tangent)
{
    for (unsigned int i1 = 0; i1 < ndf; i1++)
    {

        int ii = node[nI].btype[i1];

        if (ii >= 0)
        {
            for (unsigned int i2 = 0; i2 < node[nI].ListNodes.size(); i2++)
            {
                unsigned int nJ = node[nI].ListNodes[i2];
                for (unsigned int i3 = 0; i3 < ndf; i3++)
                {
                    // Row and column number of corresponding degree of freedom of that node

                    int jj = node[nJ].btype[i3];

                    if (jj >= 0)
                    {

                        int n1;
                        if (ii - 1 > 0)
                        {
                            n1 = ia[ii - 1] - 1;
                        }
                        else
                        {
                            n1 = 0;
                        }
                        int n2 = ia[ii] - 1;
                        int inz = -1;

                        for (int k = n1; k < n2; k++)
                            if (ja[k] == jj)
                            {
                                inz = k;
                                k = n2 + 1;
                            }

                        if (inz == -1)
                        {
                            cout << "Wrong assignment in assemble matrix in CRS format" << endl;
                            exit(0);
                        }

                        a[inz] += Tangent[i1][i2 * ndf + i3];
                    }
                }
            }

            // Assemble force vector

            b[ii - 1] += Force[i1];
        }
    }
}
void Solution::NodalAssemblyLagrange(unsigned int isw, unsigned int nI, unsigned int ndf, vector<double> Force, vector<vector<double>> Tangent)
{
    // Displacement contribution

    if (isw == 0)
    {
        for (unsigned int i1 = 0; i1 < ndf; i1++)
        {

            // Assemble displacement to Lagrange multiplier contribution

            int ii = node[nI].btype[i1];

            if (ii >= 0)
            {
                for (unsigned int i2 = 0; i2 < node[nI].Lagrange2Node.size(); i2++)
                {

                    int lag = node[nI].Lagrange2Node[i2];
                    int lam = abs(lag);

                    // Row and column number of corresponding degree of freedom of that node

                    int jj = node[lam].ltype[i1];

                    if (jj > 0)
                    {

                        int n1;
                        if (ii - 1 > 0)
                        {
                            n1 = ia[ii - 1] - 1;
                        }
                        else
                        {
                            n1 = 0;
                        }
                        int n2 = ia[ii] - 1;
                        int inz = -1;

                        for (int k = n1; k < n2; k++)
                            if (ja[k] == jj)
                            {
                                inz = k;
                                k = n2 + 1;
                            }

                        if (inz == -1)
                        {
                            cout << "Wrong assignment in assemble matrix in CRS format for Lagrange multiplier method part UL" << endl;
                            exit(0);
                        }

                        a[inz] += Tangent[i1][i2 * ndf + i1];
                    }
                }

                // Assemble force vector

                b[ii - 1] += Force[i1];
            }
        }
    }

    // Lagrange multiplier contribution

    else if (isw == 1)
    {

        for (unsigned int i1 = 0; i1 < ndf; i1++)
        {

            // Assemble displacement to Lagrange multiplier contribution

            int ii = node[nI].ltype[i1];
            if (ii > 0)
            {
                for (unsigned int i2 = 0; i2 < node[nI].Nodes2Lagrange.size(); i2++)
                {
                    int nJ = node[nI].Nodes2Lagrange[i2];

                    // Row and column number of corresponding degree of freedom of that node

                    int jj = node[nJ].btype[i1];
                    if (jj >= 0)
                    {

                        int n1;
                        if (ii - 1 > 0)
                        {
                            n1 = ia[ii - 1] - 1;
                        }
                        else
                        {
                            n1 = 0;
                        }
                        int n2 = ia[ii] - 1;
                        int inz = -1;

                        for (int k = n1; k < n2; k++)
                            if (ja[k] == jj)
                            {
                                inz = k;
                                k = n2 + 1;
                            }

                        if (inz == -1)
                        {
                            cout << "Wrong assignment in assemble matrix in CRS format for Lagrange multiplier method part LU" << endl;
                            exit(0);
                        }
                        a[inz] += Tangent[i1][i2 * ndf + i1];
                    }
                }

                // Assemble force vector

                b[ii - 1] += Force[i1];
            }
        }
    }
}
void Solution::PlotTangentResidual()
{

    ofstream output(filename.c_str(), ios::app);

    unsigned int neq = b.size();
    vector<vector<double>> K(neq, vector<double>(neq, 0));

    unsigned int currentEntry = 0;
    for (unsigned int i1 = 0; i1 < ia.size() - 1; i1++)
    {
        unsigned int NumberEntriesRow = ia[i1 + 1] - ia[i1];

        for (unsigned int i2 = 0; i2 < NumberEntriesRow; i2++)
        {
            unsigned int Column = ja[currentEntry];

            K[i1][Column - 1] = a[currentEntry];

            currentEntry++;
        }
    }

    unsigned int NumberRow = (neq + 5) / 6;
    unsigned int aux1, aux2 = 0;

    output << endl;
    for (unsigned int i1 = 0; i1 < NumberRow; i1++)
    {
        output << "     Tangent matrix\t" << endl;
        aux1 = aux2;
        aux2 = min(neq, aux1 + 6);

        output << "     row/col";
        for (unsigned int i2 = aux1; i2 < aux2; i2++)
        {
            output << fixed << setw(4) << i2 + 1 << "          ";
        }
        output << endl;
        for (unsigned int i2 = 0; i2 < neq; i2++)
        {
            output << "    " << fixed << setw(4) << i2 + 1 << "   ";

            for (unsigned int i3 = aux1; i3 < aux2; i3++)
            {
                output << showpos << scientific << setprecision(3) << K[i2][i3] << "   ";
            }
            output << endl;
        }
        output << endl;
    }
    output << endl;

    output << endl;
    output << "     Residual vector\t" << endl;
    output << endl;
    output << "     row/col";
    for (unsigned int i1 = 0; i1 < 1; i1++)
    {
        output << fixed << setw(4) << i1 + 1 << "          ";
    }
    output << endl;
    for (unsigned int i1 = 0; i1 < neq; i1++)
    {
        output << "    " << fixed << setw(4) << i1 + 1 << "   ";
        output << showpos << scientific << setprecision(3) << b[i1] << "\t" << endl;
    }
    output << endl;

    output << endl;
    output << "     Solution vector\t" << endl;
    output << endl;
    output << "     row/col";
    for (unsigned int i1 = 0; i1 < 1; i1++)
    {
        output << fixed << setw(4) << i1 + 1 << "          ";
    }
    output << endl;
    for (unsigned int i1 = 0; i1 < neq; i1++)
    {
        output << "    " << fixed << setw(4) << i1 + 1 << "   ";
        output << showpos << scientific << setprecision(3) << x[i1] << "\t" << endl;
    }
    output << endl;
}
unsigned int Solution::NumberIntegrationPoints(unsigned int etype, int ll)
{

    unsigned int lint = 0;

    // Tetrahedral element

    if (etype == 1)
    {
        // 1 pt. quadrature O(h^2)

        if (ll == 1)
        {
            lint = 1;
        }

        // 4 pt. quadrature O(h^2) - nodes of linear element

        else if (ll == -1)
        {
            lint = 4;
        }

        // 4 pt. quadrature O(h^3)

        else if (ll == 2)
        {
            lint = 4;
        }

        // 5  pt. quadrature O(h^4) -- has negative weight

        else if (ll == 3)
        {
            lint = 5;
        }

        // 11 pt. quadrature O(h^4) -- has negative weight

        else if (ll == 4)
        {
            lint = 11;
        }

        // 11 pt. quadrature O(h^4) -- has no negative weight

        else if (ll == -4)
        {
            lint = 11;
        }

        // 16 pt. quadrature O(h^5)

        else
        {
            lint = 16;
        }
    }

    // Brick element

    else if (etype == 2)
    {

        // 1 pt. quadrature

        if (ll == 1)
        {
            lint = 1;
        }

        // 2 x 2 x 2 pt. quadrature

        else if (ll == 2)
        {
            lint = 8;
        }

        // Special 9 pt. quadrature

        else if (ll == -9)
        {
            lint = 9;
        }

        // Special 4 pt. quadrature

        else if (ll == -4)
        {
            lint = 4;
        }

        // ll x ll x ll pt. quadrature

        else if (ll == 5)
        {
            lint = 5 * 5 * 5;
        }
    }

    return lint;
}
void Solution::Tint3D(int ll, unsigned int lint, vector<vector<double>> &s)
{

    const double one6 = 0.16666666666666666666666666666666;

    unsigned int ind1[6] = {1, 2, 2, 1, 1, 2};
    unsigned int ind2[6] = {1, 1, 2, 2, 2, 1};
    unsigned int ind3[6] = {2, 1, 1, 2, 1, 2};
    unsigned int ind4[6] = {2, 2, 1, 1, 2, 1};

    double alp[2] = {0.399403576166799219, 0.100596423833200785};

    // 1 pt. quadrature O(h^2)

    if (ll == 1)
    {
        for (unsigned int i1 = 0; i1 < 4; i1++)
        {
            s[i1][0] = 0.25;
        }
        s[4][0] = one6;
    }

    // 4 pt. quadrature O(h^2) - nodes of linear element

    else if (ll == -1)
    {
        s[4][3] = 0.25 * one6;
        for (unsigned int i1 = 0; i1 < 4; i1++)
        {
            for (unsigned int i2 = 0; i2 < 4; i2++)
            {
                s[i1][i2] = 0.0;
            }
            s[i1][i1] = 1.0;
            s[4][i1] = s[4][3];
        }
    }

    // 4 pt. quadrature O(h^3)

    else if (ll == 2)
    {
        s[4][3] = 0.25 * one6;
        for (unsigned int i1 = 0; i1 < 4; i1++)
        {
            for (unsigned int i2 = 0; i2 < 4; i2++)
            {
                s[i1][i2] = 0.1381966011250105;
            }
            s[i1][i1] = 0.5854101966249658;
            s[4][i1] = s[4][3];
        }
    }

    // 5  pt. quadrature O(h^4) -- has negative weight

    else if (ll == 3)
    {
        for (unsigned int i1 = 0; i1 < 4; i1++)
        {
            s[i1][0] = one6;
            s[i1][1] = s[i1][0];
            s[i1][2] = s[i1][0];
            s[i1][3] = s[i1][0];
            s[i1][4] = 0.25;
        }
        for (unsigned int i1 = 0; i1 < 4; i1++)
        {
            s[i1][i1] = 0.5;
        }
        s[4][0] = 0.075;
        s[4][1] = s[4][0];
        s[4][2] = s[4][0];
        s[4][3] = s[4][0];
        s[4][4] = -0.80 * one6;
    }

    // 11 pt. quadrature O(h^4) -- has negative weight

    else if (ll == 4)
    {
        for (unsigned int i1 = 0; i1 < 4; i1++)
        {
            for (unsigned int i2 = 0; i2 < 4; i2++)
            {
                s[i2][i1] = 0.714285714285714285e-01;
            }
        }
        for (unsigned int i1 = 0; i1 < 4; i1++)
        {
            s[i1][i1] = 0.785714285714285714e+00;
            s[4][i1] = 0.762222222222222222e-02;
        }

        for (unsigned int i1 = 4; i1 < 10; i1++)
        {
            unsigned int j = ind1[i1 - 4];
            unsigned int k = ind2[i1 - 4];
            unsigned int l = ind3[i1 - 4];
            unsigned int m = ind4[i1 - 4];
            s[0][i1] = alp[j];
            s[1][i1] = alp[k];
            s[2][i1] = alp[l];
            s[3][i1] = alp[m];
            s[4][i1] = 0.248888888888888880e-01;
        }
        s[0][10] = 0.25e+00;
        s[1][10] = 0.25e+00;
        s[2][10] = 0.25e+00;
        s[3][10] = 0.25e+00;
        s[4][10] = -0.131555555555555550e-01;
    }

    // 11 pt. quadrature O(h^4) -- has no negative weight

    else if (ll == -4)
    {
        for (unsigned int i1 = 0; i1 < 4; i1++)
        {
            for (unsigned int i2 = 0; i2 < 10; i2++)
            {
                s[i1][i2] = 0.0;
            }
            s[i1][i1] = 1.00;
            s[i1][i1 + 4] = 0.50;
            s[i1][i1 + 7] = 0.50;
            s[i1][10] = 0.25;
        }
        s[1][4] = 0.50;
        s[2][5] = 0.50;
        s[0][6] = 0.50;
        for (unsigned int i1 = 0; i1 < 4; i1++)
        {
            s[4][i1] = 1.0 / 360.0;
        }
        for (unsigned int i1 = 4; i1 < 10; i1++)
        {
            s[4][i1] = 1.0 / 90.0;
        }
        s[4][10] = 4.0 / 45.0;
    }

    // 16 pt. quadrature O(h^5)

    else
    {
        s[4][3] = 0.8395632516687135e-02;
        for (unsigned int i1 = 0; i1 < 3; i1++)
        {
            for (unsigned int i2 = 0; i2 < 4; i2++)
            {
                s[i1][i2] = 0.7611903264425430e-01;
            }
            s[i1][i1] = 0.7716429020672371e+00;
            s[4][i1] = s[4][3];
        }
        for (unsigned int i1 = 4; i1 < 16; i1++)
        {
            s[4][i1] = 0.1109034477221540e-01;
        }

        s[0][4] = 0.1197005277978019e+00;
        s[1][4] = 0.7183164526766925e-01;
        s[2][4] = 0.4042339134672644e+00;
        s[0][5] = 0.4042339134672644e+00;
        s[1][5] = 0.1197005277978019e+00;
        s[2][5] = 0.7183164526766925e-01;
        s[0][6] = 0.4042339134672644e+00;
        s[1][6] = 0.4042339134672644e+00;
        s[2][6] = 0.1197005277978019e+00;
        s[0][7] = 0.7183164526766925e-01;
        s[1][7] = 0.4042339134672644e+00;
        s[2][7] = 0.4042339134672644e+00;

        s[0][8] = 0.1197005277978019e+00;
        s[1][8] = 0.4042339134672644e+00;
        s[2][8] = 0.7183164526766925e-01;
        s[0][9] = 0.4042339134672644e+00;
        s[1][9] = 0.1197005277978019e+00;
        s[2][9] = 0.4042339134672644e+00;
        s[0][10] = 0.7183164526766925e-01;
        s[1][10] = 0.4042339134672644e+00;
        s[2][10] = 0.1197005277978019e+00;
        s[0][11] = 0.4042339134672644e+00;
        s[1][11] = 0.7183164526766925e-01;
        s[2][11] = 0.4042339134672644e+00;

        s[0][12] = 0.1197005277978019e+00;
        s[1][12] = 0.4042339134672644e+00;
        s[2][12] = 0.4042339134672644e+00;
        s[0][13] = 0.7183164526766925e-01;
        s[1][13] = 0.1197005277978019e+00;
        s[2][13] = 0.4042339134672644e+00;
        s[0][14] = 0.4042339134672644e+00;
        s[1][14] = 0.7183164526766925e-01;
        s[2][14] = 0.1197005277978019e+00;
        s[0][15] = 0.4042339134672644e+00;
        s[1][15] = 0.4042339134672644e+00;
        s[2][15] = 0.7183164526766925e-01;
    }

    // Compute fourth points

    for (unsigned int i1 = 0; i1 < lint; i1++)
    {
        s[3][i1] = 1.0 - (s[0][i1] + s[1][i1] + s[2][i1]);
    }
}
void Solution::Int3D(int ll, vector<vector<double>> &s)
{

    double sqt13 = sqrt(one3);
    double five9 = 5.0 / 9.0;
    double thty29 = 32.0 / 9.0;
    double sqtp6 = sqrt(0.6);

    int ig[4] = {-1, 1, 1, -1};
    int jg[4] = {-1, -1, 1, 1};

    // 1 pt. quadrature

    if (ll == 1)
    {
        for (unsigned int i1 = 0; i1 < 3; i1++)
        {
            s[i1][0] = 0.0;
        }
        s[3][0] = 8.0;
    }

    // 2 x 2 x 2 pt. quadrature

    else if (ll == 2)
    {
        double g = sqt13;
        for (unsigned int i1 = 0; i1 < 4; i1++)
        {
            s[0][i1] = ig[i1] * g;
            s[0][i1 + 4] = s[0][i1];
            s[1][i1] = jg[i1] * g;
            s[1][i1 + 4] = s[1][i1];
            s[2][i1] = g;
            s[2][i1 + 4] = -g;
            s[3][i1] = 1.0;
            s[3][i1 + 4] = 1.0;
        }
    }

    // Special 9 pt. quadrature

    else if (ll == -9)
    {
        double g = sqtp6;
        for (unsigned int i1 = 0; i1 < 4; i1++)
        {
            s[0][i1] = ig[i1] * g;
            s[0][i1 + 4] = s[0][i1];
            s[1][i1] = jg[i1] * g;
            s[1][i1 + 4] = s[1][i1];
            s[2][i1] = g;
            s[2][i1 + 4] = -g;
            s[3][i1] = five9;
            s[3][i1 + 4] = five9;
        }
        s[0][8] = 0.0;
        s[1][8] = 0.0;
        s[2][8] = 0.0;
        s[3][8] = thty29;
    }

    // Special 4 pt. quadrature

    else if (ll == -4)
    {
        double g = sqt13;
        for (unsigned int i1 = 0; i1 < 4; i1++)
        {
            s[0][i1] = ig[i1] * g;
            s[1][i1] = s[0][i1];
            s[2][i1] = jg[i1] * g;
            s[3][i1] = 2.0;
        }
        s[1][2] = -g;
        s[1][3] = g;
    }

    // ll x ll x ll pt. quadrature

    else if (ll == 5)
    {
        vector<double> xg;
        xg.resize(ll);
        vector<double> Wg;
        Wg.resize(ll);
        Integration1D(ll, xg, Wg);
        unsigned int lint = 0;
        for (int i1 = 0; i1 < ll; i1++)
        {
            for (int i2 = 0; i2 < ll; i2++)
            {
                for (int i3 = 0; i3 < ll; i3++)
                {
                    lint = lint + 1;
                    s[0][lint] = xg[i1];
                    s[1][lint] = xg[i2];
                    s[2][lint] = xg[i3];
                    s[3][lint] = Wg[i1] * Wg[i2] * Wg[i3];
                }
            }
        }
    }
}
void Solution::Integration1D(int ll, vector<double> &xg, vector<double> &Wg)
{
    if (ll == 1)
    {
        xg[0] = 0;
        Wg[0] = 2;
    }
    else if (ll == 2)
    {
        xg[0] = -1 / sqrt(3);
        Wg[0] = 1;
        xg[1] = 1 / sqrt(3);
        Wg[1] = 1;
    }
    else if (ll == 3)
    {
        double aux = sqrt(0.6);
        xg[0] = -aux;
        Wg[0] = 5.0 / 9.0;
        xg[1] = 0.0;
        Wg[1] = 8.0 / 9.0;
        xg[2] = aux;
        Wg[2] = 5.0 / 9.0;
    }
    else if (ll == 4)
    {
        double aux = sqrt(4.8);
        double aux2 = 1.0 / 3.0 / aux;
        xg[0] = -sqrt((3.0 + aux) / 7.0);
        Wg[0] = 0.5 - aux2;
        xg[1] = -sqrt((3.0 - aux) / 7.0);
        Wg[1] = 0.5 + aux2;
        xg[2] = -xg[1];
        Wg[2] = Wg[1];
        xg[3] = -xg[0];
        Wg[3] = Wg[0];
    }
    else if (ll == 5)
    {
        double aux = sqrt(1120);

        double aux2 = (70 + aux) / 126;
        double aux3 = (70 - aux) / 126;
        double aux4 = 1.0 / (15.0 * (aux3 - aux2));

        xg[0] = -sqrt(aux2);
        Wg[0] = (5.0 * aux3 - 3.0) * aux4 / aux2;
        xg[1] = -sqrt(aux3);
        Wg[1] = (3.0 - 5.0 * aux2) * aux4 / aux3;
        xg[2] = 0;
        Wg[2] = 2.0 * (1.0 - Wg[0] - Wg[1]);
        xg[3] = -xg[1];
        Wg[3] = Wg[1];
        xg[4] = -xg[0];
        Wg[4] = Wg[0];
    }
    else
    {
        Gauss(ll, xg, Wg);
    }
}
void Solution::Gauss(int elint, vector<double> &xg, vector<double> &Wg)
{
    double fn, beta, cc, dpn, pn1;
    double xt = 0;

    fn = double(elint);
    beta = exp(2.0 * flgamma(1.0) - flgamma(2.0));

    cc = 2.0 * beta;

    for (int n = 2; n <= elint; n++)
    {
        cc = cc * 4.0 * pow(double(n - 1), 4) / (double(2 * n - 1) * double(2 * n - 3) * pow(double(2 * n - 2), 2));
    }

    for (int n = 0; n < elint; n++)
    {
        if (n == 0)
        {
            xt = 1.0 - 2.78 / (4.0 + fn * fn);
        }
        else if (n == 1)
        {
            xt = xt - (4.10 + 0.246 * (fn - 8.0) / fn) * (1.0 - xt);
        }
        else if (n == 2)
        {
            xt = xt - (1.67 + 0.3674 * (fn - 8.0) / fn) * (xg[0] - xt);
        }
        else if (n == elint - 2)
        {
            xt = xt + (xt - xg[n - 2]) / 0.766 / (1.0 + 0.639 * (fn - 4.0) / (1.0 + 0.710 * (fn - 4.0)));
        }
        else if (n == elint - 1)
        {
            xt = xt + (xt - xg[n - 2]) / 1.67 / (1.0 + 0.22 * (fn - 8.0) / fn);
        }
        else
        {
            xt = 3.0 * xg[n - 1] - 3.0 * xg[n - 2] + xg[n - 3];
        }

        // Wurzel berechnen

        unsigned iter = 0;
        bool notconv = true;
        double dp;
        double dpn_aux;
        do
        {
            iter = iter + 1;

            // Rekursionalgorithmus

            double p1 = 1.0;
            double p = xt;
            double dp1 = 0.0;
            dp = 1.0;
            for (int m = 2; m <= elint; m++)
            {
                double c = 4.0 * pow(double(m - 1), 4) / (double(2 * m - 1) * double(2 * m - 3) * pow(double(2 * m - 2), 2));
                double q = xt * p - c * p1;
                double dq = xt * dp - c * dp1 + p;
                p1 = p;
                p = q;
                dp1 = dp;
                dp = dq;
            }
            double pn = p;
            dpn_aux = dp;
            pn1 = p1;

            // Berechnung Wurzel

            double d = pn / dpn_aux;
            xt = xt - d;

            if (fabs(d) < 1e-16)
            {
                notconv = false;
            }
        } while (notconv && iter < 50);
        dpn = dpn_aux;

        xg[n] = xt;
        Wg[n] = cc / (dpn * pn1);
    }

    for (int n = 0; n < elint; n++)
    {
        xg[n] = -xg[n];
    }
}
double Solution::flgamma(double w)
{
    int m;
    double x, p, fk, y, z, zz;
    double pi = acos(-1.0);

    x = w;
    fk = -1.0;

    if (x < 0.5)
    {
        m = 1;
        p = pi / sin(x * pi);
        x = 1.0 - x;
    }
    else
    {
        m = 0;
        p = 0.0;
    }

    do
    {
        fk = fk + 1.0;
    } while ((x + fk - 6.0) <= 0.0);

    z = x + fk;
    zz = z * z;

    y = (z - 0.5) * log(z) - z + 0.9189385332047 + (((((-4146 / zz + 1820) / zz - 1287) / zz + 1716) / zz - 6006) / zz + 180180) / z / 2162160;

    if (fk > 0.0)
    {
        int size = int(fk);
        for (int i = 0; i < size; i++)
        {
            fk = fk - 1.0;
            y = y - log(x + fk);
        }
    }

    if (m != 0)
    {
        if (p <= 0.0)
        {
            y = 0;
        }
        else
        {
            y = log(p) - y;
        }
    }

    return y;
}
void Solution::Average(bool calc_grad, optional<double> override_treEps)
{
    auto start = std::chrono::steady_clock::now();
    double stress[6] = {0, 0, 0, 0, 0, 0};
    double strain[6] = {0, 0, 0, 0, 0, 0};
    double volume = 0.0;
    for (unsigned int e = 0; e < element.size(); e++)
    {
        // Load stress

        unsigned int m = element[e].mtype - 1;
        material[m].stress(m, e, element[e].lint, 0);

        for (unsigned int l = 0; l < element[e].lint; l++)
        {
            // Summation of volume weightes stresses

            stress[0] += element[e].Sigma[0][l] * element[e].vol[l];
            stress[1] += element[e].Sigma[1][l] * element[e].vol[l];
            stress[2] += element[e].Sigma[2][l] * element[e].vol[l];

            stress[3] += element[e].Sigma[3][l] * element[e].vol[l];
            stress[4] += element[e].Sigma[4][l] * element[e].vol[l];
            stress[5] += element[e].Sigma[5][l] * element[e].vol[l];

            if (material[m].mate == "LinearElasticity")
            {
                // Displacement gradient

                double H[3][3];

                H[0][0] = element[e].tenF[0][0][l] - 1.0;
                H[0][1] = element[e].tenF[0][1][l];
                H[0][2] = element[e].tenF[0][2][l];
                H[1][0] = element[e].tenF[1][0][l];
                H[1][1] = element[e].tenF[1][1][l] - 1.0;
                H[1][2] = element[e].tenF[1][2][l];
                H[2][0] = element[e].tenF[2][0][l];
                H[2][1] = element[e].tenF[2][1][l];
                H[2][2] = element[e].tenF[2][2][l] - 1.0;

                // Strain tensor

                double eps[6];
                eps[0] = H[0][0];
                eps[1] = H[1][1];
                eps[2] = H[2][2];
                eps[3] = H[0][1] + H[1][0];
                eps[4] = H[1][2] + H[2][1];
                eps[5] = H[0][2] + H[2][0];

                // Summation of volume weighted strains

                strain[0] += eps[0] * element[e].vol[l];
                strain[1] += eps[1] * element[e].vol[l];
                strain[2] += eps[2] * element[e].vol[l];
                strain[3] += eps[3] * element[e].vol[l];
                strain[4] += eps[4] * element[e].vol[l];
                strain[5] += eps[5] * element[e].vol[l];
            }

            // Summation of volume weights

            volume += element[e].vol[l];
        }
    }

    // Averaged strains

    double ivol = 1.0 / volume;
    AveStrain[0] = strain[0] * ivol;
    AveStrain[1] = strain[1] * ivol;
    AveStrain[2] = strain[2] * ivol;
    AveStrain[3] = strain[3] * ivol;
    AveStrain[4] = strain[4] * ivol;
    AveStrain[5] = strain[5] * ivol;

    // Averages stress

    AveStress[0] = stress[0] * ivol;
    AveStress[1] = stress[1] * ivol;
    AveStress[2] = stress[2] * ivol;
    AveStress[3] = stress[3] * ivol;
    AveStress[4] = stress[4] * ivol;
    AveStress[5] = stress[5] * ivol;

    // Output

    // cout << "Average Stress and strain  " << endl;
    // cout << AveStress[0] << "  " << AveStrain[0] << endl;
    // cout << AveStress[1] << "  " << AveStrain[1] << endl;
    // cout << AveStress[2] << "  " << AveStrain[2] << endl;
    // cout << AveStress[3] << "  " << AveStrain[3] << endl;
    // cout << AveStress[4] << "  " << AveStrain[4] << endl;
    // cout << AveStress[5] << "  " << AveStrain[5] << endl;

    // Averaged bulk modulus

    double treSig = AveStress[0] + AveStress[1] + AveStress[2];
    treEps = AveStrain[0] + AveStrain[1] + AveStrain[2];

    if (override_treEps)
    {
        treEps = override_treEps.value();
    }
    else
    {
    }

    AveK = one3 * treSig / treEps;

    // Contraint

    J = (AveK - DesignK) * (AveK - DesignK);

    // cout << "AveK " << AveK << " --> J " << J << endl;

    // if (calc_grad)
    //     cout << "AveK " << AveK << "  DesignK  " << DesignK << endl;
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    // cout << "Average: stress,strain took " << elapsed.count() << " seconds!" << endl;

    if (!calc_grad)
    {
        return;
    }

    /* GRADIENT COMPUTATION  */

    // Initialize arrays

    // from Material::init()
    // lam = matdata[0]*matdata[1]/((1+matdata[1])*(1-2*matdata[1])); // matdata[0] = E, matdata[1] = ν,  K = λ + 2/3 μ
    // mu  = matdata[0]/(2*(1+matdata[1]));
    // rho = matdata[2];

    // Cu:          neq x neq
    // Ctheta:      neq x numel*2
    // Ju:          neq
    // Jtheta:      numel*2
    // θ: for each element, then offset 0: lam. 1: mu

    start = std::chrono::steady_clock::now();
    if (!inited_backward)
    {
        Ju.resize(neq);
        Jtheta.resize(numel * 2);
        inited_backward = true;
    }

    for (unsigned int i1 = 0; i1 < neq; i1++)
    {
        Ju[i1] = 0;
    }
    for (unsigned int i1 = 0; i1 < numel * 2; i1++)
    {
        Jtheta[i1] = 0;
    }
    end = std::chrono::steady_clock::now();
    elapsed = end - start;

    // Derivative of J with respect to material parameters of each element
    start = std::chrono::steady_clock::now();
    double fac = 2 * (AveK - DesignK) * one3 / treEps * ivol;
    // cout << "Fac:" << fac << endl; // J
    for (unsigned int e = 0; e < element.size(); e++)
    {
        for (unsigned int l = 0; l < element[e].lint; l++)
        {

            // Normal strain

            double eps[3];
            eps[0] = element[e].tenF[0][0][l] - 1.0;
            eps[1] = element[e].tenF[1][1][l] - 1.0;
            eps[2] = element[e].tenF[2][2][l] - 1.0;

            Jtheta[e * 2] += fac * 3 * (eps[0] + eps[1] + eps[2]) * element[e].vol[l];
            Jtheta[e * 2 + 1] += fac * 2 * (eps[0] + eps[1] + eps[2]) * element[e].vol[l];
        }
    }
    end = std::chrono::steady_clock::now();
    elapsed = end - start;
    // cout << "Average: Jtheta took " << elapsed.count() << " seconds!" << endl;

    // Derivative of J with respect to displacements

    // Loop over all elements

    start = std::chrono::steady_clock::now();
    for (unsigned int e = 0; e < element.size(); e++)
    {
        // Loop over all Gauss points within each element

        for (unsigned int l = 0; l < element[e].lint; l++)
        {
            // Loop over all nodes of corresponding element

            for (unsigned int i1 = 0; i1 < element[e].connectivity.size(); i1++)
            {
                // Global node number & Location in vector Ju

                unsigned int nG = element[e].connectivity[i1];

                //                 cout << "e   " << e << "  i1  " << i1 << "  nG  " << nG << "  ii  "  << ii << endl;

                // Residual and tangent with respect to deformation gradient

                double shpI[3], d[2];

                // Load material properties

                unsigned int m = element[e].mtype - 1;
                d[0] = material[m].lam;
                d[1] = material[m].mu; // update lam, mu here.

                // Load shape functions

                shpI[0] = element[e].shp[0][i1][l];
                shpI[1] = element[e].shp[1][i1][l];
                shpI[2] = element[e].shp[2][i1][l];

                // Contribution to derivative of J with respect to displacements

                int j1;
                j1 = node[nG].btype[0] - 1;
                if (j1 >= 0)
                {
                    Ju[j1] += fac * (3 * d[0] + 2 * d[1]) * shpI[0] * element[e].vol[l];
                }
                j1 = node[nG].btype[1] - 1;
                if (j1 >= 0)
                {
                    Ju[j1] += fac * (3 * d[0] + 2 * d[1]) * shpI[1] * element[e].vol[l];
                }
                j1 = node[nG].btype[2] - 1;
                if (j1 >= 0)
                {
                    Ju[j1] += fac * (3 * d[0] + 2 * d[1]) * shpI[2] * element[e].vol[l];
                }
            }
        }
    }
    end = std::chrono::steady_clock::now();
    elapsed = end - start;
    // cout << "Averag: Ju took " << elapsed.count() << " seconds!" << endl;

    // Derivatieve of C with respect to displacments and material parameters

    // Loop over nodes

    // start = std::chrono::steady_clock::now();
    int j1, j2;

    // -- Ctheta Sparse --
    start = std::chrono::steady_clock::now();
    std::vector<std::map<int, double>> Ct_rows(neq);
    for (unsigned int n = 0; n < node.size(); n++)
    {
        // Loop over neighboring elements
        for (unsigned int i1 = 0; i1 < node[n].NeighborElements.size(); i1++)
        {
            // Corresponding element number
            unsigned int e = node[n].NeighborElements[i1];

            // Corresponing node number in element
            unsigned int nP = node[n].AssignNodeInList(n, e);

            // Loop over integration points
            for (unsigned int l = 0; l < element[e].lint; l++)
            {
                // Residual and tangent with respect to deformation gradient
                double shpI[3], F[3][3], d[3], R[3], T[3][3][3], TT[3][2];

                // Load material properties
                unsigned int m = element[e].mtype - 1;
                d[0] = material[m].lam;
                d[1] = material[m].mu;
                string MaterialModel = material[m].mate;

                // Load shape functions
                shpI[0] = element[e].shp[0][nP][l];
                shpI[1] = element[e].shp[1][nP][l];
                shpI[2] = element[e].shp[2][nP][l];

                // Load volume
                double volume = element[e].vol[l];

                // Deformation gradient
                F[0][0] = element[e].tenF[0][0][l];
                F[0][1] = element[e].tenF[0][1][l];
                F[0][2] = element[e].tenF[0][2][l];
                F[1][0] = element[e].tenF[1][0][l];
                F[1][1] = element[e].tenF[1][1][l];
                F[1][2] = element[e].tenF[1][2][l];
                F[2][0] = element[e].tenF[2][0][l];
                F[2][1] = element[e].tenF[2][1][l];
                F[2][2] = element[e].tenF[2][2][l];

                // Residual and tangent with respect to theta
                acegen.LinearElasticity3dTheta(shpI, F, d, volume, TT); // TT = dR/dθ

                // Assign location in tangent
                unsigned int jj = e * 2;

                // Compute derivative of C with respect to theta
                j1 = node[n].btype[0] - 1;
                if (j1 >= 0)
                {
                    Ct_rows[j1][jj] += TT[0][0];
                    Ct_rows[j1][jj + 1] += TT[0][1];
                }

                j1 = node[n].btype[1] - 1;
                if (j1 >= 0)
                {
                    Ct_rows[j1][jj] += TT[1][0];
                    Ct_rows[j1][jj + 1] += TT[1][1];
                }

                j1 = node[n].btype[2] - 1;
                if (j1 >= 0)
                {
                    Ct_rows[j1][jj] += TT[2][0];
                    Ct_rows[j1][jj + 1] += TT[2][1];
                }
            }
        }
    }

    // 1-based indexing!!
    vector<MKL_INT> Ct_ia(neq + 1);
    Ct_ia[0] = 1;
    vector<MKL_INT> Ct_ja(0);
    std::vector<double> Ct_a;

    for (unsigned int i = 0; i < neq; i++)
    {
        auto &row_map = Ct_rows[i];
        for (const auto &[j, val] : row_map)
        {
            Ct_a.push_back(val);
            Ct_ja.push_back(j + 1);
        }
        Ct_ia[i + 1] = Ct_a.size() + 1;
    }
    end = std::chrono::steady_clock::now();
    elapsed = end - start;
    // cout << "Averag: Ctheta SPARSE took " << elapsed.count() << " seconds!" << endl;

    start = std::chrono::steady_clock::now();

    if (!use_iterative_solver)
    {
        // solve DIRECT
        solve_pardiso(neq, ia.data(), ja.data(), a.data(), Ju.data(), adjoint_sol_noadj.data());
    }
    else
    {
        // solve ITERATIVELY
        MKL_INT nq = neq;

        double abs_tol_ = abs_tol;
        if (abs_tol < 0)
        {
            assert(rel_tol > 0);
            abs_tol_ = warmstart ? backward_residual : 0; // use last residual from first step as guidance.
        }
        double last_residual = solve_iteratively(nq, &adjoint_sol_noadj, &Ju, &bilu0_backward, !warmstart, &last_backward_iterations, false, abs_tol_, rel_tol); // do NOT set sol to zero!
        if (!warmstart)
        {
            backward_residual = last_residual;
        }
    }

    end = std::chrono::steady_clock::now();
    elapsed = end - start;
    // cout << "Average: transposed solve took " << elapsed.count() << " seconds!" << endl;

    // J theta SPARSE!
    start = std::chrono::steady_clock::now();
    // Create MKL sparse matrix descriptor
    sparse_matrix_t Ct_descr;
    mkl_sparse_d_create_csr(&Ct_descr, SPARSE_INDEX_BASE_ONE, neq, numel * 2, Ct_ia.data(), Ct_ia.data() + 1, Ct_ja.data(), Ct_a.data());

    vector<double> dJ_dtheta_fromu_fromsparse(numel * 2);
    matrix_descr descr = {SPARSE_MATRIX_TYPE_GENERAL};
    mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1., Ct_descr, descr, adjoint_sol_noadj.data(), 0., dJ_dtheta_fromu_fromsparse.data()); // https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2025-1/mkl-sparse-mv.html

    vector<double> dJ_dtheta_total_fromsparse(numel * 2);
    for (uint col = 0; col < numel * 2; col++)
    {
        dJ_dtheta_total_fromsparse[col] = -dJ_dtheta_fromu_fromsparse[col] + Jtheta[col];
    }
    end = std::chrono::steady_clock::now();
    elapsed = end - start;
    // cout << "Average: Jtheta from SPARSE total took " << elapsed.count() << " seconds!" << endl;

    // Conversion: https://www.chemeurope.com/en/encyclopedia/Elastic_modulus.html

    // My mapping: f(E,ν) = (λ,μ)
    dJ_dEnu_total.resize(numel * 2);
    for (uint iel = 0; iel < numel; iel++)
    {
        // convert (λ,μ) to (E,ν):
        unsigned int m = element[iel].mtype - 1;
        Material &mat = material[m];
        double E = mat.mu * (3 * mat.lam + 2 * mat.mu) / (mat.lam + mat.mu);
        E = mat.matdata[0];
        double nu = mat.lam / (2 * (mat.lam + mat.mu));
        nu = mat.matdata[1];

        double d_lam_by_d_E = nu / ((1 + nu) * (1 - 2 * nu));
        double denom = (-2 * nu * nu - nu + 1);
        double d_lam_by_d_nu = E * (2 * nu * nu + 1) / (denom * denom);
        double d_mu_by_d_E = 1 / (2 * (1 + nu));
        double d_mu_by_d_nu = -E / (2 * (1 + nu) * (1 + nu));

        double this_dJ_dE = dJ_dtheta_total_fromsparse[2 * iel] * d_lam_by_d_E + dJ_dtheta_total_fromsparse[2 * iel + 1] * d_mu_by_d_E;
        double this_dJ_dnu = dJ_dtheta_total_fromsparse[2 * iel] * d_lam_by_d_nu + dJ_dtheta_total_fromsparse[2 * iel + 1] * d_mu_by_d_nu;
        dJ_dEnu_total[2 * iel] = this_dJ_dE;
        dJ_dEnu_total[2 * iel + 1] = this_dJ_dnu;
    }

    // lam = E * nu ( (1+nu)  (1 - 2nu))
    // mu = E / (2 (1+nu))
    // d lam / d E = nu / ((1+nu) (1 - 2nu))
    // d lam / d nu = E * (2 nu^2 + 1) / (- 2 nu^2 - nu + 1)^2
    // d mu / d E = 1 / (2 (1+nu))
    // d mu / d nu = - E / (2 (1 + nu)^2 )
}

void Solution::ResizeEquationNumber()
{

    // FEM nodes

    int i1 = 0;
    int i2 = 0;
    neq = 0;
    for (unsigned int n = 0; n < node.size(); n++)
    {
        for (unsigned int i = 0; i < ndf; i++)
        {
            if (node[n].btype[i] == 0)
            {
                i1++;
                node[n].btype[i] = i1;
                if ((unsigned int)node[n].btype[i] > neq)
                    neq = (unsigned int)node[n].btype[i];
            }
            else if (node[n].btype[i] == 1)
            {
                i2--;
                node[n].btype[i] = i2;
            }
            if (node[n].ltype[i] == 1)
            {
                i1++;
                node[n].ltype[i] = i1;
                if ((unsigned int)node[n].ltype[i] > neq)
                    neq = (unsigned int)node[n].ltype[i];
            }
            else if (node[n].ltype[i] == 1)
            {
                i2--;
                node[n].ltype[i] = i2;
            }
        }
    }
}
