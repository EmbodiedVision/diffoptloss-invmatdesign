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
 * Main code author is Christian Weißenfels, functionality for exporting data and using the solver iteratively added by Jens Kreber.
 */

// Includes
#ifndef Solution_h
#define Solution_h

// Includes
#include <iostream>
#include <string>
#include <optional>

#include "pardiso_wrapper.h"

using namespace std;

class Solution
{

public:
  Solution() : searchdomain("Box"), debug(0), shiftMP(0), mass(1), contact(0), factorSD(1.5) {}

  string stype, ptype, difftype, searchtype, homogenized, searchdomain, diffkind, stabtype, theory, formulation, TestFunction, TrialFunction, TestFunctionType, HomType, HomMethod;
  int debug, shiftMP, mass, time, contact;
  double HomBoundary[6], HomFixBoundary[3], AveStress[6], AveStrain[6], factorSD, TestParameter, TrialParameter, SearchEnlargeFactor;
  vector<double> b;
  vector<double> x;
  vector<double> a;
  vector<long long int> ia;
  vector<long long int> ja; // Jens: changed to long long to match MKL indexing sizes.
  vector<double> tparam;
  vector<vector<double>> table;
  vector<vector<double>> ContactMethod;
  vector<vector<double>> ContactPair;
  vector<double> sload;
  vector<vector<int>> ListNodes2Lagrange;

  double AveK, DesignK, J, treEps;
  vector<vector<double>> Cu;
  vector<vector<double>> Ctheta;
  vector<double> Ju;
  vector<double> Jtheta;
  vector<double> dJ_dtheta_total;
  vector<double> dJ_dEnu_total;
  PardisoState *pardisoState = nullptr; // Jens

  // Jens: iterative solver state:
  MKL_INT slv_rci_request;
  int slv_max_iter = 150;
  MKL_INT slv_ipar[128];
  double slv_dpar[128];
  MKL_INT slv_itercount;
  vector<double> bilu0_forward;
  vector<double> bilu0_backward;
  vector<double> adjoint_sol_noadj;
  double forward_residual = -1, backward_residual = -1;

  // Jens: initialization stuff
  bool inited_forward = false, inited_backward = false;
  bool warmstart = false;
  bool use_iterative_solver = false;
  bool do_export_npz = false;
  bool do_also_export_system = false;
  bool do_calc_grad = false;
  float abs_tol = -1;
  float rel_tol = 1e-8;
  int last_forward_iterations = 0;
  int last_backward_iterations = 0;

  vector<double> get_tparam() { return tparam; }
  void add_tparam(double c1) { tparam.push_back(c1); }
  void add_table(vector<double> dat) { table.push_back(dat); }

  void print_dense_K(); // Jens: added
  void DirectSolver();
  void inverse(int n, vector<vector<double>> &a, vector<vector<double>> &ainv);
  void eigen(vector<vector<double>> &v, vector<double> &lam, int &rot);
  double prop(string type, int n, int dir, double t);
  void Solve(unsigned int isw);
  double solve_iteratively(MKL_INT &nq, vector<double> *the_x, vector<double> *the_b, vector<double> *the_precond, bool calc_precond, int *it_count, bool zero_x = true, double abs_tol = 0, double rel_tol = 1e-8);
  void NewtonIteration(unsigned int &it, unsigned int neq, double &aengy0);
  void NodalAssembly(unsigned int nI, unsigned int ndf, vector<double> Force, vector<vector<double>> Tangent);
  void NodalAssemblyLagrange(unsigned int isw, unsigned int nI, unsigned int ndf, vector<double> Force, vector<vector<double>> Tangent);
  void PlotTangentResidual();
  unsigned int NumberIntegrationPoints(unsigned int etype, int itype);
  void Tint3D(int ll, unsigned int lint, vector<vector<double>> &s);
  void Int3D(int ll, vector<vector<double>> &s);
  void Integration1D(int ll, vector<double> &xg, vector<double> &Wg);
  void Gauss(int elint, vector<double> &xg, vector<double> &Wg);
  double flgamma(double w);
  void Average(bool calc_grad = false, optional<double> override_treEps = nullopt);
  void ResizeEquationNumber();

private:
};

#endif
