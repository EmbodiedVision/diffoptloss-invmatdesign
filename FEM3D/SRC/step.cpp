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

#include "Element.h"
#include "Node.h"
#include "Material.h"
#include "Lagrange.h"
#include "Solution.h"
#include <string>
#include <fstream>
#include <assert.h>
#include <stdlib.h>

// Jens:
#include "../cnpy/cnpy.h"
#include <limits.h>
#include <optional>
#include <iostream>
#include <chrono>

extern bool projection;
extern int t, ndf, tsteps, ngp;
extern double dt, Time;
extern unsigned int neq, numnp, nummp, nprint;
extern string filename;
extern vector<Node> node;
extern vector<Element> element;
extern vector<Material> material;
extern vector<Lagrange> lagrange;
extern Solution solution;
extern bool save_np_sol, save_np_grad;

void step(bool calc_grad, optional<double> override_treEps)
{

	// Solution of displacement increment with sparse solver (isw=3) or explicit with lumped mass matrix

	int isw = 6;
	if ((solution.time == 0 && solution.mass == 0) || solution.time != 0)
		isw = 3;

	// Define Column Indicator and Row Pointer for solution within the CRS format

	auto start = std::chrono::steady_clock::now();
	if (isw == 3 && !solution.inited_forward)
	{
		auto start = std::chrono::steady_clock::now();
		solution.ia.resize(neq + 1);
		solution.ja.clear();
		for (unsigned int n = 0; n < node.size(); n++)
		{
			node[n].InitializeCRSFormat(neq);
		}
		auto end = std::chrono::steady_clock::now();
		std::chrono::duration<double> elapsed = end - start;
		// Jens Note: Row and column indices for PARDISO start with 1! This means there is a 1 at first position in ia.
		// Adjust row pointer
		solution.ia[0] = 1;
		for (unsigned int i = 1; i < neq + 1; i++)
			solution.ia[i] += solution.ia[i - 1];
	}
	double aengy = 0;
	unsigned int kp = solution.ja.size();

	if (!solution.inited_forward)
	{
		solution.b.resize(neq);
		solution.a.resize(kp);
		solution.x.resize(neq);
		solution.adjoint_sol_noadj.resize(neq);
		solution.inited_forward = true;
	}

	// Initialize nodal data
	for (unsigned int n = 0; n < node.size(); n++)
	{
		node[n].initialize(n, t, solution.time, isw);
	} // I think this is required every time. Maybe not if removing update and so on...

	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed = end - start;
	// cout << "step() initialization took " << elapsed.count() << " seconds!" << endl;

	ofstream output(filename.c_str(), ios::app);
	string npz_filename = filename;
	npz_filename.replace(filename.find(".out"), 4, ".npz");

	for (unsigned int it = 0; it < 25; it++)
	{

		start = std::chrono::steady_clock::now();
		// Initialize solution arrays

		for (unsigned int i1 = 0; i1 < solution.b.size(); i1++)
		{
			solution.b[i1] = 0;
		}
		if (!solution.warmstart)
		{
			// cout << "-- COLD starting!" << endl;
			for (unsigned int i1 = 0; i1 < solution.x.size(); i1++)
			{
				solution.x[i1] = 0;
				solution.adjoint_sol_noadj[i1] = 0;
			}
		}

		for (unsigned int i1 = 0; i1 < solution.a.size(); i1++)
		{
			solution.a[i1] = 0;
		}

		// Kinematics at integration points

		for (unsigned int e = 0; e < element.size(); e++)
		{
			element[e].Kinematics();
		}

		// Stress at material point
		for (unsigned int e = 0; e < element.size(); e++)
		{
			unsigned int m = element[e].mtype - 1;
			int istrt = 1;
			if (it == 0)
				istrt = 0;
			material[m].stress(m, e, element[e].lint, istrt);
		}

		// Add force boundary

		for (unsigned int n = 0; n < node.size(); n++)
			node[n].BoundaryForce(n);
		end = std::chrono::steady_clock::now();
		elapsed = end - start;
		// cout << "Step initial took " << elapsed.count() << " seconds!" << endl;

		// Nodal residual and tangent

		start = std::chrono::steady_clock::now();
		for (unsigned int n = 0; n < node.size(); n++)
		{
			node[n].SolidStaticAceGen(n);
		}
		end = std::chrono::steady_clock::now();
		elapsed = end - start;
		// cout << "SolidStaticAceGen took " << elapsed.count() << " seconds!" << endl;

		// Residual vector and tangent, if Lagrange multiplier method

		if (solution.HomMethod == "Lagrange")
		{
			for (unsigned int n = 0; n < node.size(); n++)
			{
				node[n].TangentAndResidualLagrange(n);
			}
		}
		// Solution
		start = std::chrono::steady_clock::now();
		solution.Solve(isw);
		end = std::chrono::steady_clock::now();
		elapsed = end - start;

		// Jens --- Numpy array export ---

		vector<double> positions(node.size() * 3, -1);
		if (it == 0 && solution.do_export_npz)
		{
			cnpy::npz_save(npz_filename, "x", solution.x, "w");
			// save assignment node.dof -> eq
			vector<int> node_dof_to_eq(node.size() * 3, -1);
			for (uint n = 0; n < node.size(); n++)
			{
				for (uint d = 0; d < 3; d++)
				{
					node_dof_to_eq[n * 3 + d] = node[n].btype[d];
				}
			}
			cnpy::npz_save(npz_filename, "btype", &node_dof_to_eq[0], {node.size(), 3}, "a");

			for (uint n = 0; n < node.size(); n++)
			{
				for (uint d = 0; d < 3; d++)
				{
					positions[n * 3 + d] = node[n].acoor[d];
				}
			}
			cnpy::npz_save(npz_filename, "initial_coords", &positions[0], {node.size(), 3}, "a");

			if (solution.do_also_export_system)
			{
				// Stores in CSR format for PARDISO, which starts indexing by 1, not 0!!
				cnpy::npz_save(npz_filename, "b", solution.b, "a");
				cnpy::npz_save(npz_filename, "a", solution.a, "a");
				// create 32 bit arrays to save a bit of space (lol)
				vector<int> ia32(solution.ia.size());
				vector<int> ja32(solution.ja.size());
				for (uint i = 0; i < solution.ia.size(); i++)
				{
					assert(solution.ia[i] <= INT_MAX);
					ia32[i] = solution.ia[i];
				}
				for (uint i = 0; i < solution.ja.size(); i++)
				{
					assert(solution.ja[i] <= INT_MAX);
					ja32[i] = solution.ja[i];
				}
				cnpy::npz_save(npz_filename, "ia", ia32, "a");
				cnpy::npz_save(npz_filename, "ja", ja32, "a");
			}
		}
		// Jens --- End Numpy array export ---

		// Residual norm and energy

		solution.NewtonIteration(it, neq, aengy);

		// Jens --- Numpy array export ---
		if (it == 0 && solution.do_export_npz)
		{
			// save known displacement values
			vector<double> boundary_displacements;
			int current_counter = 0;
			for (uint n = 0; n < node.size(); n++)
			{
				for (uint d = 0; d < 3; d++)
				{
					if (node[n].btype[d] < 0)
					{
						current_counter--;
						assert(current_counter == node[n].btype[d]);
						boundary_displacements.push_back(node[n].bknown[d]);
					}
				}
			}
			cnpy::npz_save(npz_filename, "known_displacements", boundary_displacements, "a");
		}
		// Jens --- End Numpy array export ---

		for (unsigned int n = 0; n < node.size(); n++)
		{
			node[n].Update(n, dt, solution.time, isw);
		}

		// Jens --- Numpy array export ---
		if (it == 0 && solution.do_export_npz)
		{
			// Save updated coordinates
			for (uint n = 0; n < node.size(); n++)
			{
				for (uint d = 0; d < 3; d++)
				{
					positions[n * 3 + d] = node[n].acoor[d];
				}
			}
			cnpy::npz_save(npz_filename, "new_coords", &positions[0], {node.size(), 3}, "a");
		}
		// Jens --- End Numpy array export ---

		// Update Lagrange multipliers
		if (solution.HomMethod == "Lagrange")
		{
			for (unsigned int n = 0; n < lagrange.size(); n++)
			{
				lagrange[n].Update(n);
			}
		}

		// Jens: only need 1 actual iteration and then one to compute all the data..
		assert(it == 0 || it == 1 || it == 50);
		for (unsigned int e = 0; e < element.size(); e++)
		{
			element[e].Kinematics();
		}
		break;
	}

	// Store previous nodal coordinates

	for (unsigned int n = 0; n < node.size(); n++)
	{
		node[n].pcoor[0] = node[n].acoor[0];
		node[n].pdisp[0] = node[n].disp[0];
		node[n].pcoor[1] = node[n].acoor[1];
		node[n].pdisp[1] = node[n].disp[1];
		node[n].pcoor[2] = node[n].acoor[2];
		node[n].pdisp[2] = node[n].disp[2];
	}

	// Projection of stress onto the nodes

	if (projection)
	{
		for (unsigned int n = 0; n < node.size(); n++)
			node[n].project(n);
	}

	start = std::chrono::steady_clock::now();
	solution.Average(calc_grad, override_treEps);
	end = std::chrono::steady_clock::now();
	elapsed = end - start;

	// Jens --- Numpy array export ---
	if (solution.do_export_npz)
	{
		cnpy::npz_save(npz_filename, "AveStrain", solution.AveStrain, {6}, "a");
		cnpy::npz_save(npz_filename, "AveStress", solution.AveStress, {6}, "a");
		cnpy::npz_save(npz_filename, "treEps", &solution.treEps, {1}, "a");
		cnpy::npz_save(npz_filename, "AveK", &solution.AveK, {1}, "a");
		cnpy::npz_save(npz_filename, "J", &solution.J, {1}, "a");
	}
	// Jens --- End Numpy array export ---
}
