/*
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
 */

#include "Node.h"
#include "Element.h"
#include "Material.h"
#include "Solution.h"
#include <assert.h>
#include <string>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <iterator>
#include <stdlib.h>
#include <math.h>

using namespace std;

extern vector<Node> node;
extern vector<Element> element;
extern vector<Material> material;
extern Solution solution;
extern bool projection, vtk_output, print;
extern unsigned int numnp, numel, ndf, ndm, ndd, nprint, tsteps, lsteps;
extern double dt, G[3];
extern string filename;

void BoundaryConditions(unsigned int ndf, unsigned int typ, vector<vector<int>> Information, vector<vector<double>> Values, vector<vector<vector<double>>> Coordinates);
void BoundaryHomogenization();

void init_system()
{

	ifstream input(filename.c_str(), ios::in);

	// Default values and flags

	projection = false;
	vtk_output = false;
	print = true;
	ndd = 0;
	solution.formulation = "Standard";
	solution.HomType = "Zero";

	vector<vector<int>> BoundaryType;
	vector<vector<double>> BoundaryValues;
	vector<vector<vector<double>>> BoundaryCoor;
	vector<vector<int>> DisplacementType;
	vector<vector<double>> DisplacementValues;
	vector<vector<vector<double>>> DisplacementCoor;
	vector<vector<int>> ForceType;
	vector<vector<double>> ForceValues;
	vector<vector<vector<double>>> ForceCoor;
	vector<vector<int>> ProportionalType;
	vector<vector<double>> ProportionalValues;
	vector<vector<vector<double>>> ProportionalCoor;
	vector<vector<int>> ContactTypeN;
	vector<vector<double>> ContactValuesN;
	vector<vector<vector<double>>> ContactCoorN;
	vector<vector<int>> ContactTypeT;
	vector<vector<double>> ContactValuesT;
	vector<vector<vector<double>>> ContactCoorT;

	filename.replace(filename.find(".ini"), 4, ".out");
	ofstream output(filename.c_str(), ios::out);

	while (input)
	{
		string line;
		getline(input, line);

		if (line == "NumberOfNodes")
		{
			input >> numnp;
			node.resize(numnp);
		}
		else if (line == "NumberOfElements")
		{
			input >> numel;
			element.resize(numel);
		}
		else if (line == "DimensionNodalDegree")
		{
			input >> ndm;
			input >> ndf;
		}
		else if (line == "Materials")
		{
			input >> ndd;
			material.resize(ndd);
		}
		else if (line == "StepsPrint")
		{
			input >> nprint;
		}
		else if (line == "TimeIncrement")
		{
			input >> dt;
		}
		else if (line == "TimeSteps")
		{
			input >> tsteps;
		}
		else if (line == "SurfaceLoad")
		{
			string Stype;
			input >> Stype;
			if (Stype == "Constant")
			{
				solution.sload.resize(1);
				input >> solution.sload[0];
			}
			if (Stype == "Linear")
			{
				solution.sload.resize(2);
				input >> solution.sload[0];
				input >> solution.sload[1];
			}
		}
		else if (line == "Loadtype")
		{

			// Store load type

			input >> solution.ptype;

			// Number of different proportional loads

			int ploads = 0;
			input >> ploads;

			// Reads current line (wich is now empty)

			getline(input, line);
			for (int i = 0; i < ploads; i++)
			{
				void eigenvalue(int n, double *a, int it_max, double *v, double *d, int &it_num, int &rot_num);
				getline(input, line);

				istringstream iss(line);
				vector<double> values;
				copy(istream_iterator<double>(iss), istream_iterator<double>(), back_insert_iterator<vector<double>>(values));
				if (values.size() > 0)
					solution.table.push_back(values);
			}
		}
		else if (line == "Node")
		{
			for (uint n = 0; n < numnp; n++)
			{
				input >> node[n].icoor[0];
				input >> node[n].icoor[1];
				input >> node[n].icoor[2];

				node[n].acoor[0] = node[n].icoor[0];
				node[n].acoor[1] = node[n].icoor[1];
				node[n].acoor[2] = node[n].icoor[2];
				node[n].pcoor[0] = node[n].icoor[0];
				node[n].acoor[1] = node[n].icoor[1];
				node[n].acoor[2] = node[n].icoor[2];

				// Initialize displacements and velocity

				node[n].disp[0] = 0.0;
				node[n].disp[1] = 0.0;
				node[n].disp[2] = 0.0;
				node[n].velo[0] = 0.0;
				node[n].velo[1] = 0.0;
				node[n].velo[2] = 0.0;

				// Initialize fields

				node[n].btype.resize(ndf);
				for (uint i = 0; i < ndf; i++)
					node[n].btype[i] = 0.0;
				node[n].ltype.resize(ndf);
				for (uint i = 0; i < ndf; i++)
					node[n].ltype[i] = 0.0;
				node[n].bprop.resize(ndf);
				for (uint i = 0; i < ndf; i++)
					node[n].bprop[i] = 0.0;
				node[n].bknown.resize(ndf);
				for (uint i = 0; i < ndf; i++)
					node[n].bknown[i] = 0.0;
				node[n].bdisp.resize(ndf);
				for (uint i = 0; i < ndf; i++)
					node[n].bdisp[i] = 0.0;
				node[n].bforce.resize(ndf);
				for (uint i = 0; i < ndf; i++)
				{
					node[n].bforce[i] = 0.0;
					node[n].istf = 0;
				}
				node[n].ptype.resize(ndf);
				for (uint i = 0; i < ndf; i++)
					node[n].ptype[i] = 0.0;
				node[n].penaltyN.resize(ndf);
				for (uint i = 0; i < ndf; i++)
					node[n].penaltyN[i] = 0.0;
				node[n].penaltyT.resize(ndf);
				for (uint i = 0; i < ndf; i++)
					node[n].penaltyT[i] = 0.0;
			}
		}
		else if (line == "NumberNodesElementIntegrationTypeMaterialNumber")
		{
			for (uint e = 0; e < numel; e++)
			{
				unsigned int nodes_el;
				input >> nodes_el;
				element[e].nen = nodes_el;
				int ityp_el;
				input >> ityp_el;
				element[e].itype = ityp_el;
				unsigned int mtyp_el;
				input >> mtyp_el;
				element[e].mtype = mtyp_el;
			}
		}
		else if (line == "Element")
		{
			for (uint e = 0; e < numel; e++)
			{

				// Read initial connectivity list

				unsigned int nodes_el = element[e].nen;
				element[e].connectivity.resize(nodes_el);
				for (unsigned int n = 0; n < nodes_el; n++)
				{
					input >> element[e].connectivity[n];
					element[e].connectivity[n]--;
				}
			}
		}
		else if (line == "TimeSolution")
		{
			string aux;
			input >> aux;
			if (aux == "Central")
			{
				solution.time = 0;
				string aux2;
				input >> aux2;
				if (aux2 == "Consistent")
				{
					solution.mass = 0;
				}
				else if (aux2 == "RowSum")
				{
					solution.mass = 1;
				}
				else if (aux2 == "Dual")
				{
					solution.mass = 2;
				}
				else if (aux2 == "Quadratic")
				{
					solution.mass = 3;
				}
				double c1;
				c1 = dt * dt;
				c1 = 1.0 / c1;
				solution.add_tparam(c1);
			}
			else if (aux == "Newmark")
			{
				solution.time = 1;
				string aux2;
				double beta;
				double gamma;
				input >> aux2;
				input >> beta;
				input >> gamma;
				double c1;
				if (aux2 == "Consistent")
				{
					solution.mass = 0;
				}
				else if (aux2 == "RowSum")
				{
					solution.mass = 1;
				}
				c1 = dt * dt * beta;
				c1 = 1.0 / c1;
				solution.add_tparam(c1);
				c1 = dt * beta;
				c1 = 1.0 / c1;
				solution.add_tparam(c1);
				c1 = 2.0 * beta;
				c1 = (1.0 - 2.0 * beta) / c1;
				solution.add_tparam(c1);
				c1 = dt * beta;
				c1 = gamma / c1;
				solution.add_tparam(c1);
				c1 = beta;
				c1 = 1.0 - gamma / c1;
				solution.add_tparam(c1);
				c1 = 2.0 * beta;
				c1 = (1.0 - gamma / c1) * dt;
				solution.add_tparam(c1);
			}
			else if (aux == "Static")
			{
				solution.time = 2;
			}
		}
		else if (line == "DiffType")
		{
			string aux;
			input >> aux;
			solution.diffkind = aux;
		}
		else if (line == "Theory")
		{
			string aux;
			input >> aux;
			solution.theory = aux;
		}
		else if (line == "Gravity")
		{
			input >> G[0];
			input >> G[1];
			input >> G[2];
		}
		else if (line == "Materiallist")
		{
			for (uint i = 0; i < ndd; i++)
			{
				input >> material[i].mate;

				getline(input, line);
				getline(input, line);

				istringstream iss(line);
				vector<double> values;
				copy(istream_iterator<double>(iss), istream_iterator<double>(), back_insert_iterator<vector<double>>(values));
				for (unsigned int j = 0; j < values.size(); j++)
					material[i].matdata.push_back(values[j]);
				material[i].init(material[i].mate);
			}
		}
		else if (line == "TestFunction")
		{
			input >> solution.TestFunction;
		}
		else if (line == "TrialFunction")
		{
			input >> solution.TrialFunction;
		}
		else if (line == "Formulation")
		{
			input >> solution.formulation;
		}
		else if (line == "Projection")
		{
			projection = true;
		}
		else if (line == "Vtk")
		{
			vtk_output = true;
		}
		else if (line == "NoPrint")
		{
			print = false;
		}
		else if (line == "Boundary" || line == "Displacement" || line == "Force" || line == "Proportional" || line == "ContactNormal" || line == "ContactTangential")
		{

			// Initialize data

			unsigned int typ = 0;
			vector<int> Ivalues;
			vector<double> Dvalues;
			if (line == "Boundary")
			{
				typ = 1;
			}
			else if (line == "Proportional")
			{
				typ = 2;
			}
			else if (line == "Displacement")
			{
				typ = 3;
			}
			else if (line == "Force")
			{
				typ = 4;
			}
			else if (line == "ContactNormal")
			{
				typ = 5;
			}
			else if (line == "ContactTangential")
			{
				typ = 6;
			}

			// Input data

			string Geometry, Treatment, aux;
			unsigned int number;
			vector<vector<double>> Icoordinates;
			input >> Geometry;
			input >> Treatment;
			getline(input, line);
			input >> aux;
			input >> number;
			getline(input, line);

			if (Geometry == "Plane" || Geometry == "Node")
			{
				input >> aux;
				getline(input, line);
				istringstream iss(line);
				vector<double> coordinates;
				copy(istream_iterator<double>(iss), istream_iterator<double>(), back_insert_iterator<vector<double>>(coordinates));
				Icoordinates.push_back(coordinates);
			}
			else if (Geometry == "Surface")
			{
				for (unsigned int i1 = 0; i1 < number; i1++)
				{
					input >> aux;
					getline(input, line);
					istringstream iss(line);
					vector<double> coordinates;
					copy(istream_iterator<double>(iss), istream_iterator<double>(), back_insert_iterator<vector<double>>(coordinates));
					Icoordinates.push_back(coordinates);
				}
			}

			input >> aux;
			getline(input, line);
			istringstream iss1(line);
			copy(istream_iterator<double>(iss1), istream_iterator<double>(), back_insert_iterator<vector<double>>(Dvalues));

			// Typevalue 0: Node, 1:

			vector<int> Typevalues(3);
			if (Geometry == "Plane")
			{
				Typevalues[0] = 1;
			}
			else if (Geometry == "Surface")
			{
				Typevalues[0] = 2;
			}
			else if (Geometry == "Node")
			{
				Typevalues[0] = 3;
			}
			else
			{
				cout << "Input wrong" << endl;
			}
			Typevalues[1] = number;
			if (Treatment == "Add")
			{
				Typevalues[2] = 1;
			}
			else
			{
				Typevalues[2] = 0;
			}

			// Store data in corresponding array

			if (typ == 1)
			{
				BoundaryType.push_back(Typevalues);
				BoundaryValues.push_back(Dvalues);
				BoundaryCoor.push_back(Icoordinates);
			}
			if (typ == 2)
			{
				ProportionalType.push_back(Typevalues);
				ProportionalValues.push_back(Dvalues);
				ProportionalCoor.push_back(Icoordinates);
			}
			if (typ == 3)
			{
				DisplacementType.push_back(Typevalues);
				DisplacementValues.push_back(Dvalues);
				DisplacementCoor.push_back(Icoordinates);
			}
			if (typ == 4)
			{
				ForceType.push_back(Typevalues);
				ForceValues.push_back(Dvalues);
				ForceCoor.push_back(Icoordinates);
			}
			if (typ == 5)
			{
				ContactTypeN.push_back(Typevalues);
				ContactValuesN.push_back(Dvalues);
				ContactCoorN.push_back(Icoordinates);
			}
			if (typ == 6)
			{
				ContactTypeT.push_back(Typevalues);
				ContactValuesT.push_back(Dvalues);
				ContactCoorT.push_back(Icoordinates);
			}

			void BoundaryConditions(unsigned int ndf, unsigned int typ, vector<vector<int>> Information, vector<vector<double>> Values, vector<vector<vector<double>>> Coordinates);
		}
		else if (line == "Homogenization")
		{
			input >> solution.HomType;
			input >> solution.HomMethod;
			getline(input, line);
			getline(input, line);

			istringstream iss(line);
			vector<double> values;
			copy(istream_iterator<double>(iss), istream_iterator<double>(), back_insert_iterator<vector<double>>(values));
			for (unsigned int j = 0; j < values.size(); j++)
			{
				solution.HomBoundary[j] = values[j];
			}

			string aux;
			input >> aux;

			getline(input, line);
			getline(input, line);

			istringstream iss1(line);
			vector<unsigned int> Ivalues;
			copy(istream_iterator<unsigned int>(iss1), istream_iterator<unsigned int>(), back_insert_iterator<vector<unsigned int>>(Ivalues));
			for (unsigned int j = 0; j < Ivalues.size(); j++)
			{
				solution.HomFixBoundary[j] = Ivalues[j];
			}
		}
		else if (line == "InverseDesign")
		{
			input >> solution.DesignK;
		}
		else
		{
			// cout << "Wrong input :" << line << endl;
		}
	}

	input.close();

	// Specify boundary information

	BoundaryConditions(ndf, 1, BoundaryType, BoundaryValues, BoundaryCoor);
	BoundaryConditions(ndf, 2, ProportionalType, ProportionalValues, ProportionalCoor);
	BoundaryConditions(ndf, 3, DisplacementType, DisplacementValues, DisplacementCoor);
	BoundaryConditions(ndf, 4, ForceType, ForceValues, ForceCoor);

	// Initialize displacement and velocity

	if (solution.ptype == "Shockwave")
	{

		double bulk = material[0].matdata[0];
		double nu = material[0].matdata[1];
		double rho = material[0].matdata[2];
		double jp = material[0].matdata[3];
		double jm = material[0].matdata[4];

		double v = sqrt(bulk * 0.5 / rho * (1.0 + 1.0 / (jp * jm)));
		double l = 8 * nu * v / (3 * bulk) * jp * jm / (jp - jm);

		double t = 0.0;

		for (uint n = 0; n < numnp; n++)
		{
			double xi = node[n].icoor[2] - v * t;
			double ch = cosh(0.5 * xi / l);

			node[n].disp[0] = 0.0;
			node[n].disp[1] = 0.0;
			node[n].disp[2] = ((jp + jm) / 2 - 1) * xi + (jp - jm) * l * (log(ch) - log(2));

			node[n].velo[0] = 0.0;
			node[n].velo[1] = 0.0;
			node[n].velo[2] = 0.5 * v * (2 - (jp + jm) - (jp - jm) * tanh(0.5 * xi / l));
		}
	}
	else if (solution.ptype == "LinearProp")
	{

		// Initial displacements and velocities at boundary nodes

		int psize = (int)solution.table.size();
		for (int p = 0; p < psize; p++)
		{

			double t0 = solution.table[p][0];
			double l0 = solution.table[p][1];
			double tmax = solution.table[p][2];
			double lmax = solution.table[p][3];

			double slope = (lmax - l0) / (tmax - t0);

			for (uint n = 0; n < numnp; n++)
			{
				if (node[n].bprop[0] == p + 1)
				{
					node[n].disp[0] = l0 * node[n].bdisp[0];
					node[n].velo[0] = slope * node[n].bdisp[0];
				}
				if (node[n].bprop[1] == p + 1)
				{
					node[n].disp[1] = l0 * node[n].bdisp[1];
					node[n].velo[1] = slope * node[n].bdisp[1];
				}
				if (node[n].bprop[2] == p + 1)
				{
					node[n].disp[2] = l0 * node[n].bdisp[2];
					node[n].velo[2] = slope * node[n].bdisp[2];
				}
			}
		}
	}
	else if (solution.ptype == "ConstVelocity")
	{

		double velo0[3];
		velo0[0] = solution.table[0][0];
		velo0[1] = solution.table[0][1];
		velo0[2] = solution.table[0][2];
		for (uint n = 0; n < numnp; n++)
		{
			node[n].velo[0] = velo0[0];
			node[n].velo[1] = velo0[1];
			node[n].velo[2] = velo0[2];
		}
	}

	output << "\nNumber of Nodes\n"
		   << numnp << endl;
	output << "\nNumber of Elements\n"
		   << numel << endl;
	output << "\nNumber of Materials\n"
		   << ndd << endl;
	output << "\nTime increment\n"
		   << dt << endl;
	output << "\nNumber of time steps\n"
		   << tsteps << endl;
	output << "\nIncrement of steps to plot data\n"
		   << nprint << endl;

	if (print)
	{
		output << "\nINITIAL NODAL COORDINATES\n"
			   << endl;
		for (uint i = 0; i < numnp; i++)
		{
			output << "   Node " << noshowpos << setw(6) << i + 1 << ":   " << showpos << setprecision(6) << scientific << node[i].icoor[0] << "  " << node[i].icoor[1] << "  " << node[i].icoor[2] << endl;
		}
		output << "\nEQUATION NUMBER OF NODES\n"
			   << endl;
		for (uint n = 0; n < numnp; n++)
		{
			output << "   Node " << noshowpos << setw(6) << n + 1 << ":   ";
			for (uint i = 0; i < ndf; i++)
				output << showpos << setw(6) << node[n].btype[i] << "  ";
			for (uint i = 0; i < ndf; i++)
				output << showpos << setw(6) << node[n].ltype[i] << "  ";
			output << endl;
		}
		output << "\nBOUNDARY DISPLACEMENTS\n"
			   << endl;
		for (uint n = 0; n < numnp; n++)
		{
			output << "   Node " << noshowpos << setw(6) << n + 1 << ":   ";
			for (uint i = 0; i < ndf; i++)
				output << showpos << setprecision(6) << scientific << node[n].bdisp[i] << "  ";
			output << endl;
		}
		if (node[0].bforce.size() > 0)
		{
			output << "\nBOUNDARY FORCES\n"
				   << endl;
			for (uint n = 0; n < numnp; n++)
			{
				output << "   Node " << noshowpos << setw(6) << n + 1 << ":   ";
				for (uint i = 0; i < ndf; i++)
					output << showpos << setprecision(6) << scientific << node[n].bforce[i] << "  ";
				output << endl;
			}
		}
		if (node[0].bprop.size() > 0)
		{
			output << "\nNUMBER OF PROPORTIONAL LOADING\n"
				   << endl;
			for (uint n = 0; n < numnp; n++)
			{
				output << "   Node " << noshowpos << setw(6) << n + 1 << ":   ";
				for (uint i = 0; i < ndf; i++)
					output << showpos << setw(6) << node[n].bprop[i] << "  ";
				output << endl;
			}
		}
		output << "\nInitialDisplacement" << endl;
		for (uint n = 0; n < numnp; n++)
		{
			output << "   Node " << noshowpos << setw(6) << n + 1 << ":   " << showpos << setprecision(6) << scientific << node[n].disp[0] << "  " << node[n].disp[1] << "  " << node[n].disp[2] << endl;
		}
		output << "\nInitialVelocity" << endl;
		for (uint n = 0; n < numnp; n++)
		{
			output << "   Node " << noshowpos << setw(6) << n + 1 << ":   " << showpos << setprecision(6) << scientific << node[n].velo[0] << " " << node[n].velo[1] << " " << node[n].velo[2] << endl;
		}

		output << "\nMaterial type of element \n"
			   << endl;
		for (uint i = 0; i < numel; i++)
		{
			output << "   Element " << noshowpos << setw(6) << i + 1 << ":   " << noshowpos << setw(6) << element[i].mtype << endl;
		}
		output << "\nNodal numbers of each Element\n"
			   << endl;
		for (uint p = 0; p < numel; p++)
		{
			output << "   Element " << noshowpos << setw(6) << p + 1 << ":   ";
			for (unsigned int j = 0; j < element[p].connectivity.size(); j++)
			{
				output << noshowpos << setw(6) << element[p].connectivity[j] << "  ";
			}
			output << endl;
		}
	}

	output << "\nLIST OF MATERIAL MODELS" << endl;
	for (uint i = 0; i < ndd; i++)
	{
		output << "\n Material:   " << i + 1 << "\n"
			   << endl;
		output << "  " << material[i].mate << endl;
		vector<double> data = material[i].get_data();
		if (material[i].mate == "CompFlow")
		{
			output << "\n\tBulk modulus:                  " << data[0] << endl;
			output << "\tNewtonian viscosity:           " << data[1] << endl;
			output << "\tGas density:                   " << data[2] << endl;
			output << "\tMaximal relative density J+:   " << data[3] << endl;
			output << "\tMinimal relative density J-:   " << data[4] << endl;
		}
		else if (material[i].mate == "NeoHooke" || material[i].mate == "LinearElasticity")
		{
			output << "\n\tYoungs modulus:        " << data[0] << endl;
			output << "\tPoisson ratio:           " << data[1] << endl;
			output << "\tDensity:                 " << data[2] << endl;
		}
	}

	output << "\nSHAPE FUNCTION\n"
		   << endl;
	output << "  " << solution.stype << endl;

	output << "\nLOADING TYPE\n"
		   << endl;
	output << "  " << solution.ptype << endl;
	if (solution.ptype == "LinearProp")
	{
		int ploads = (int)solution.table.size();
		for (int p = 0; p < ploads; p++)
		{
			// Restore table

			output << "\n\t Loading type of boundary value x: x(t) = l(t)*x_max" << endl;
			output << "\n\t with:        l(t) = l0 + (lmax - l0)/(tmax - t0)*t" << endl;

			output << "\n\tTime t = 0:                    " << solution.table[p][0] << endl;
			output << "\tprop value at time t = 0:      " << solution.table[p][1] << endl;
			output << "\tTime t = t_max:                " << solution.table[p][2] << endl;
			output << "\tprop value at time t = t_max:  " << solution.table[p][3] << endl;
		}
	}
}
void BoundaryConditions(unsigned int ndf, unsigned int typ, vector<vector<int>> Information, vector<vector<double>> Values, vector<vector<vector<double>>> Coordinates)
{
	// Define conditons at corresponding node

	vector<double> NodalValue;
	NodalValue.resize(ndf);

	for (unsigned int i1 = 0; i1 < Information.size(); i1++)
	{
		unsigned int Geometry = Information[i1][0];
		unsigned int Number = Information[i1][1];
		unsigned int AddOrReplace = Information[i1][2];

		// Define maximal and minimal values

		double xmin[3], xmax[3];
		xmin[0] = Coordinates[i1][0][0];
		xmin[1] = Coordinates[i1][0][1];
		xmin[2] = Coordinates[i1][0][2];
		xmax[0] = xmin[0];
		xmax[1] = xmin[1];
		xmax[2] = xmin[2];

		if (Geometry == 1)
		{
			for (unsigned int i2 = 0; i2 < node.size(); i2++)
			{
				// Initialization

				if (typ == 1)
				{
					for (unsigned int i3 = 0; i3 < ndf; i3++)
					{
						NodalValue[i3] = node[i2].btype[i3];
					}
				}
				else if (typ == 2)
				{
					for (unsigned int i3 = 0; i3 < ndf; i3++)
					{
						NodalValue[i3] = node[i2].bprop[i3];
					}
				}
				else if (typ == 3)
				{
					for (unsigned int i3 = 0; i3 < ndf; i3++)
					{
						NodalValue[i3] = node[i2].bdisp[i3];
					}
				}
				else if (typ == 4)
				{
					for (unsigned int i3 = 0; i3 < ndf; i3++)
					{
						NodalValue[i3] = node[i2].bforce[i3];
					}
				}

				// Initial arrays of boundary conditions

				if (Number == 1 && fabs(node[i2].icoor[0] - xmax[0]) < 1e-6)
				{
					for (unsigned int i3 = 0; i3 < ndf; i3++)
					{
						if (AddOrReplace == 1 && NodalValue[i3] == 0)
						{
							NodalValue[i3] = Values[i1][i3];
						}
						else if (AddOrReplace == 0)
						{
							NodalValue[i3] = Values[i1][i3];
						}
					}
				}
				else if (Number == 2 && fabs(node[i2].icoor[1] - xmax[1]) < 1e-6)
				{
					for (unsigned int i3 = 0; i3 < ndf; i3++)
					{
						if (AddOrReplace == 1 && NodalValue[i3] == 0)
						{
							NodalValue[i3] = Values[i1][i3];
						}
						else if (AddOrReplace == 0)
						{
							NodalValue[i3] = Values[i1][i3];
						}
					}
				}
				if (Number == 3 && fabs(node[i2].icoor[2] - xmax[2]) < 1e-6)
				{
					for (unsigned int i3 = 0; i3 < ndf; i3++)
					{
						if (AddOrReplace == 1 && NodalValue[i3] == 0)
						{
							NodalValue[i3] = Values[i1][i3];
						}
						else if (AddOrReplace == 0)
						{
							NodalValue[i3] = Values[i1][i3];
						}
					}
				}

				// Boundary condition

				if (typ == 1)
				{
					for (unsigned int i3 = 0; i3 < ndf; i3++)
					{
						node[i2].btype[i3] = static_cast<int>(NodalValue[i3]);
					}
				}

				// Proportial loading

				else if (typ == 2)
				{
					for (unsigned int i3 = 0; i3 < ndf; i3++)
					{
						node[i2].bprop[i3] = static_cast<int>(NodalValue[i3]);
					}
				}

				// Dirichlet boundary

				else if (typ == 3)
				{
					for (unsigned int i3 = 0; i3 < ndf; i3++)
					{
						node[i2].bdisp[i3] = NodalValue[i3];
					}
				}

				// Neumann boundary

				else if (typ == 4)
				{
					for (unsigned int i3 = 0; i3 < ndf; i3++)
					{
						node[i2].bforce[i3] = NodalValue[i3];
					}
				}
			}
		}
	}
}

void reset_system()
{
	// some stuff from init_system, just what is required after step()
	{
		for (uint n = 0; n < numnp; n++)
		{
			node[n].acoor[0] = node[n].icoor[0];
			node[n].acoor[1] = node[n].icoor[1];
			node[n].acoor[2] = node[n].icoor[2];
			node[n].pcoor[0] = node[n].icoor[0];
			node[n].acoor[1] = node[n].icoor[1];
			node[n].acoor[2] = node[n].icoor[2];

			// Initialize displacements and velocity

			node[n].disp[0] = 0.0;
			node[n].disp[1] = 0.0;
			node[n].disp[2] = 0.0;
			node[n].velo[0] = 0.0;
			node[n].velo[1] = 0.0;
			node[n].velo[2] = 0.0;
		}
	}

	assert(solution.ptype == "LinearProp");
	if (solution.ptype == "LinearProp")
	{
		// Initial displacements and velocities at boundary nodes

		int psize = (int)solution.table.size();
		for (int p = 0; p < psize; p++)
		{
			double t0 = solution.table[p][0];
			double l0 = solution.table[p][1];
			double tmax = solution.table[p][2];
			double lmax = solution.table[p][3];

			double slope = (lmax - l0) / (tmax - t0);

			for (uint n = 0; n < numnp; n++)
			{
				if (node[n].bprop[0] == p + 1)
				{
					node[n].disp[0] = l0 * node[n].bdisp[0];
					node[n].velo[0] = slope * node[n].bdisp[0];
				}
				if (node[n].bprop[1] == p + 1)
				{
					node[n].disp[1] = l0 * node[n].bdisp[1];
					node[n].velo[1] = slope * node[n].bdisp[1];
				}
				if (node[n].bprop[2] == p + 1)
				{
					node[n].disp[2] = l0 * node[n].bdisp[2];
					node[n].velo[2] = slope * node[n].bdisp[2];
				}
			}
		}
	}
}
