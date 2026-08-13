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
 * Main code author is Christian Weißenfels, functionality for communicating with control process added by Jens Kreber.
 */

// #include "Common.h"
#include "Node.h"
#include "Element.h"
#include "Material.h"
#include "Solution.h"
#include "AceGen.h"
#include "Lagrange.h"
#include <fstream>
#include <string>
#include <algorithm>
#include <cassert>
#include <string.h>
#include <chrono>
#include <getopt.h>

using namespace std;

// Initialization of simulation adta

bool interactive = false;
bool projection, contact, vtk_output, print;
unsigned int t, start, numnp, numel, ndf, ndm, ndd, nprint, tsteps, lsteps, neq;
double Time = 0;
double dt, sload;
double G[3];
string filename;

vector<Node> node;
vector<Element> element;
vector<Material> material;
vector<Lagrange> lagrange;
Solution solution;
AceGen acegen;
std::ifstream *interactive_stream_tofem;
std::ofstream *interactive_stream_topy;

// Initialization of functions
void init_system();
void init_system();
void reset_system();
void init_algorithm();
void step(bool calc_grad = false, optional<double> override_treEps = nullopt);
void plot(int isw);

void setup_stream()
{
	char buf[4];
	buf[3] = 0;
	interactive_stream_tofem->read(buf, 3);
	assert(strcmp(buf, "hi.") == 0);
	interactive_stream_topy->write(buf, 3);
	interactive_stream_topy->flush();
}

int recv_mat()
{
	uint32_t size;
	interactive_stream_tofem->read(reinterpret_cast<char *>(&size), 4);
	if (size == 0)
		return 0;

	char control[3]; // calc_grad, warmstart, use_old_sol
	interactive_stream_tofem->read(control, 3);
	if (control[0] > 0)
		solution.do_calc_grad = true;
	else
		solution.do_calc_grad = false;

	if (control[1] == 0)
		solution.warmstart = false;
	if (control[2] == 0)
	{
		// reset solution
		std::fill(solution.x.begin(), solution.x.end(), 0);
		std::fill(solution.adjoint_sol_noadj.begin(), solution.adjoint_sol_noadj.end(), 0);
	}

	float tolerances[2];
	interactive_stream_tofem->read(reinterpret_cast<char *>(&tolerances), 8);
	solution.abs_tol = tolerances[0];
	solution.rel_tol = tolerances[1];

	// cout << "Material size is " << material.size() << endl;
	assert(material.size() * 3 * sizeof(double) == size);
	uint nel = element.size();
	for (uint iel = 0; iel < nel; iel++)
	{
		unsigned int m = element[iel].mtype - 1;
		assert(iel == m);
		Material &mat = material[m];
		interactive_stream_tofem->read(reinterpret_cast<char *>(mat.matdata.data()), 3 * sizeof(double));
		mat.init("LinearElasticity"); // simply re-calculates lam and mu from E and ν
	}
	return 1;
}

void send_grad()
{
	interactive_stream_topy->write(reinterpret_cast<char *>(&solution.last_forward_iterations), 4);
	interactive_stream_topy->write(reinterpret_cast<char *>(&solution.last_backward_iterations), 4);

	interactive_stream_topy->write(reinterpret_cast<char *>(&solution.AveK), sizeof(solution.AveK));
	interactive_stream_topy->write(reinterpret_cast<char *>(&solution.J), sizeof(solution.J));

	uint32_t size = solution.dJ_dEnu_total.size() * sizeof(double);
	interactive_stream_topy->write(reinterpret_cast<char *>(&size), sizeof(uint32_t));
	uint nel = element.size();
	interactive_stream_topy->write(reinterpret_cast<char *>(solution.dJ_dEnu_total.data()), size);
	interactive_stream_topy->flush();

	// Linesearch mode!
	double stepsize;
	interactive_stream_tofem->read(reinterpret_cast<char *>(&stepsize), 8);
	while (stepsize >= 0)
	{
		for (uint iel = 0; iel < nel; iel++)
		{
			unsigned int m = element[iel].mtype - 1;
			assert(iel == m);
			Material &mat = material[m];
			mat.matdata[0] -= stepsize * solution.dJ_dEnu_total[2 * iel];
			mat.matdata[1] -= stepsize * solution.dJ_dEnu_total[2 * iel + 1];
			mat.init("LinearElasticity"); // simply re-calculates lam and mu from E and ν
		}
		// do forward pass! calc aveK!
		step(false);
		interactive_stream_topy->write(reinterpret_cast<char *>(&solution.AveK), sizeof(solution.AveK));
		interactive_stream_tofem->read(reinterpret_cast<char *>(&stepsize), 8);
	}
}

int main(int argc, char **argv)
{

	// Name of inputfile
	if (argc < 2)
	{
		cerr << "Usage: " << argv[0] << " config.ini [--flags=...]" << endl;
		exit(1);
	}

	filename = argv[1];
	string fifo_path;

	if (argc > 2)
	{
		const struct option long_options[] = {
			{"solver", required_argument, 0, 's'},
			{"interactive_fifo", required_argument, 0, 'i'},
			{"export_npz", no_argument, 0, 'n'},
			{"also_export_system", no_argument, 0, 'a'},
			{"calc_grad", no_argument, 0, 'g'},
			{"abs_tol", required_argument, 0, 't'},
			{"rel_tol", required_argument, 0, 'r'},
			{0, 0, 0, 0}};

		int c;
		int option_index = 0;
		while ((c = getopt_long(argc - 1, argv + 1, "", long_options, &option_index)) != -1)
		{
			switch (c)
			{
			case 's': // solver: "direct" or "iterative"
				if (strcmp(optarg, "direct") == 0)
					solution.use_iterative_solver = false;
				else if (strcmp(optarg, "iterative") == 0)
					solution.use_iterative_solver = true;
				else
				{
					cerr << "solver arg must be direct or iterative!" << endl;
					exit(1);
				}
				break;
			case 'i': // interactive
				interactive = true;
				fifo_path = optarg;
				break;
			case 'n': // export data to npz
				solution.do_export_npz = true;
				break;
			case 'a': // export whole system
				solution.do_also_export_system = true;
				break;
			case 'g':
				solution.do_calc_grad = true;
				break;
			case 't':
				solution.abs_tol = stof(optarg);
				break;
			case 'r':
				solution.rel_tol = stof(optarg);
				break;
			default:
				cerr << "Unknown switch case!" << endl;
				exit(2);
			}
		}
	}

	if (interactive)
	{
		// cout << "trying to open fifo path '" << fifo_path << "'.." << endl;
		interactive_stream_tofem = new std::ifstream(fifo_path + "_tofem", std::ios::binary);
		if (!interactive_stream_tofem)
		{
			std::cerr << "Error opening tofem FIFO: " << strerror(errno) << endl;
		}
		interactive_stream_topy = new std::ofstream(fifo_path + "_topy", std::ios::binary);
		if (!interactive_stream_topy)
		{
			std::cerr << "Error opening topy FIFO: " << strerror(errno) << endl;
		}
		setup_stream();
	}

	// Initialization of the system
	auto start = std::chrono::steady_clock::now();
	init_system();
	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed = end - start;
	// cout << "init_system() took " << elapsed.count() << " seconds!" << endl;

	// Initialization of the algorithm
	start = std::chrono::steady_clock::now();
	init_algorithm();
	end = std::chrono::steady_clock::now();
	elapsed = end - start;
	// cout << "init_algorithm() took " << elapsed.count() << " seconds!" << endl;

	// Output of initial values

	plot(1);

	// Time loop

	for (t = 0; t < tsteps; t++)
	{

		// Actual time

		Time += dt;

		// Computation step

		plot(0);
		uint N_MAT_STEPS = 1;
		uint i_matstep = 0;

		while (interactive || i_matstep < N_MAT_STEPS)
		{
			reset_system();
			// cout << "runing mat step " << i_matstep << endl;

			if (interactive)
			{
				int cont = recv_mat();
				if (!cont)
					break;
			}

			// normal:
			start = std::chrono::steady_clock::now();
			step(solution.do_calc_grad);
			end = std::chrono::steady_clock::now();
			elapsed = end - start;
			// cout << "step() took " << elapsed.count() << " seconds!" << endl;

			if (interactive)
			{
				send_grad();
			}

			solution.warmstart = true;
			i_matstep += 1;
		}

		// Output of data at selected time steps

		if ((t + 1) % nprint == 0)
			plot(2);
	}

	// Output of final values

	if (interactive)
	{
		// cout << "C: --- exiting ---" << endl;
	}
	else
	{
		// cout << "Ok tschüss! :)" << endl;
	}

	plot(3);
}
