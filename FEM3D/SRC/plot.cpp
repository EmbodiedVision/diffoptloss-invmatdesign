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

#include "Element.h"
#include "Node.h"
#include "Material.h"
#include "Solution.h"
#include <assert.h>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <ctime>

using namespace std;

extern bool vtk_output;
extern unsigned int t, start, numnp, numel, ndf;
extern double dt, Time;
extern string filename;
extern vector<Node> node;
extern vector<Element> element;
extern vector<Material> material;
extern Solution solution;

extern vector<int> NodalInfluence;

void plot(int isw)
{

	// Output mode for material point data

	string filename_ini = filename;
	string filename_vtk = filename;

	// Text to output file

	ofstream output(filename.c_str(), ios::app);
	if (isw == 1)
	{
		output << "\n\nInitialization\n";
		start = clock();
	}
	double duration = (clock() - start) / (double)CLOCKS_PER_SEC;
	if (isw == 3)
		output << "\n\nComputation successfully completed\n";
	if (isw == 0)
		output << "\n   Time  " << setw(12) << Time << "   Time step   " << setw(12) << t + 1 << "   CPU Time   " << setw(12) << duration << "\n"
			   << endl;

	// Text to Paraview file

	if (vtk_output && isw != 0)
	{

		ostringstream strs;
		strs << setw(12) << scientific << Time;
		string stime = strs.str();

		// Modify filename

		filename_vtk.erase(filename.find("."), 4);
		filename_vtk = filename_vtk + stime + ".vtu";
		ofstream vtk(filename_vtk.c_str(), ios::out);

		// Header

		vtk << "<VTKFile type=\"UnstructuredGrid\"     version=\"0.1\"     byte_order=\"LittleEndian\">" << endl;
		vtk << "<UnstructuredGrid>" << endl;
		vtk << "<Piece     NumberOfPoints=\"" << numnp << "\"     NumberOfCells=\"" << numel << "\">" << endl;

		// Actual position vector

		vtk << "<Points>" << endl;
		vtk << "<DataArray Name=\"Position\"     type=\"Float64\"     NumberOfComponents=\"" << 3 << "\"     format=\"ascii\">" << endl;
		for (unsigned int n = 0; n < node.size(); n++)
		{
			vtk << " " << setprecision(4) << scientific << uppercase << node[n].acoor[0] << "  " << node[n].acoor[1] << "  " << node[n].acoor[2] << endl;
		}
		vtk << "</DataArray>" << endl;
		vtk << "</Points>" << endl;

		// Point data

		vtk << "<PointData>" << endl;

		// Displacement

		vtk << "<DataArray Name=\"Displacement\"     type=\"Float64\"     NumberOfComponents=\"" << 3 << "\"\tformat=\"ascii\">" << endl;
		for (unsigned int n = 0; n < node.size(); n++)
		{
			vtk << " " << setprecision(4) << scientific << uppercase << node[n].disp[0] << "  " << node[n].disp[1] << "  " << node[n].disp[2] << "  " << endl;
		}
		vtk << "</DataArray>" << endl;

		// Velocity

		vtk << "<DataArray Name=\"Velocity\"\t type=\"Float64\"\t NumberOfComponents=\"" << 3 << "\"\tformat=\"ascii\">" << endl;
		for (unsigned int n = 0; n < node.size(); n++)
		{
			vtk << " " << setprecision(4) << scientific << uppercase << node[n].velo[0] << "  " << node[n].velo[1] << "  " << node[n].velo[2] << "  " << endl;
		}
		vtk << "</DataArray>" << endl;

		// Stress

		vtk << "<DataArray Name=\"Stress\"\t type=\"Float64\"\t NumberOfComponents=\"" << 10 << "\"\tformat=\"ascii\">" << endl;
		for (unsigned int n = 0; n < node.size(); n++)
		{
			for (int i = 0; i < 10; i++)
			{
				vtk << " " << setprecision(4) << scientific << uppercase << node[n].stress[i] << "   ";
			}
			vtk << endl;
		}
		vtk << "</DataArray>" << endl;

		// End of file

		vtk << "</PointData>" << endl;

		// Plot cell data

		vtk << "<CellData Scalars=\"scalars\">" << endl;
		vtk << "<DataArray type=\"Float32\" Name=\"MaterialNumber\" format=\"ascii\">" << endl;

		for (unsigned int e = 0; e < element.size(); e++)
		{
			unsigned int m = element[e].mtype;
			vtk << "\t" << m << endl;
		}

		vtk << "</DataArray>" << endl;
		vtk << "</CellData>" << endl;

		vtk << "<Cells>" << endl;
		vtk << "<DataArray type=\"Int32\"\t Name=\"connectivity\"\t format=\"ascii\">" << endl;

		for (unsigned int e = 0; e < element.size(); e++)
		{
			for (unsigned int i = 0; i < element[e].connectivity.size(); i++)
			{
				vtk << "\t" << element[e].connectivity[i];
			}
			vtk << endl;
		}

		vtk << "</DataArray>" << endl;
		vtk << "<DataArray type=\"Int32\"\t Name=\"offsets\"\t format=\"ascii\">" << endl;

		int ii = 0;
		for (unsigned int e = 0; e < element.size(); e++)
		{
			int nel = element[e].connectivity.size();
			ii = ii + nel;
			vtk << "\t" << ii << endl;
		}

		vtk << "</DataArray>" << endl;
		vtk << "<DataArray type=\"Int32\"\t Name=\"types\"\t format=\"ascii\">" << endl;

		int number = 10;
		for (unsigned int e = 0; e < element.size(); e++)
		{
			int nel = element[e].connectivity.size();
			if (ndf == 3 && nel == 10)
				number = 24;
			else if (ndf == 3 && nel == 8)
				number = 12;
			else if (ndf == 3 && nel == 27)
				number = 25;
			else if (ndf == 2 && nel == 3)
				number = 5;
			else if (ndf == 2 && nel == 6)
				number = 22;
			else if (ndf == 2 && nel == 4)
				number = 9;
			else if (ndf == 2 && nel == 9)
				number = 23;

			vtk << "\t" << number << endl;
		}

		vtk << "</DataArray>" << endl;
		vtk << "</Cells>" << endl;
		vtk << "</Piece>" << endl;
		vtk << "</UnstructuredGrid>" << endl;
		vtk << "</VTKFile>" << endl;
	}

	// Write header for paraview movie output

	if (isw == 1 && vtk_output)
	{
		// Start file

		filename_ini.replace(filename_ini.find("out"), 3, "pvd");
		ofstream vtk_movie(filename_ini.c_str(), ios::out);

		// Header

		vtk_movie << "<?xml version=\"1.0\"?>" << endl;
		vtk_movie << "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">" << endl;
		vtk_movie << "<Collection>" << endl;

		vtk_movie << "<DataSet timestep=\"" << setw(6) << t << "\" group= \"0\" part=\"0\" file=\"" << filename_vtk << "\"/>" << endl;
	}

	if (isw == 2 && vtk_output)
	{

		// Open file

		filename_ini.replace(filename_ini.find("out"), 3, "pvd");
		ofstream vtk_movie(filename_ini.c_str(), ios::app);

		vtk_movie << "<DataSet timestep=\"" << setw(6) << t + 1 << "\" group= \"0\" part=\"0\" file=\"" << filename_vtk << "\"/>" << endl;
	}

	if (isw == 3 && vtk_output)
	{

		// Close

		filename_ini.replace(filename_ini.find("out"), 3, "pvd");
		ofstream vtk_movie(filename_ini.c_str(), ios::app);

		vtk_movie << "</Collection>" << endl;
		vtk_movie << "</VTKFile>" << endl;
	}
}
