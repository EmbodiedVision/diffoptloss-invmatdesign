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
#include "Lagrange.h"
#include "Solution.h"
#include <string>
#include <assert.h>
#include <fstream>
#include <stdlib.h>

using namespace std;

extern bool projection;
extern double dt;
extern unsigned int ndf, numnp, numel;
extern string filename;

extern vector<Node> node;
extern vector<Element> element;
extern vector<Material> material;
extern vector<Lagrange> lagrange;
extern Solution solution;

void plot(string filename);

void init_algorithm()
{

    ofstream output(filename.c_str(), ios::app);

    // Initialization of arrays on element level

    for (unsigned int e = 0; e < element.size(); e++)
    {

        if (element[e].nen == 4 || element[e].nen == 10)
        {
            element[e].lint = solution.NumberIntegrationPoints(1, element[e].itype);
        }
        else
        {
            element[e].lint = solution.NumberIntegrationPoints(2, element[e].itype);
        }
        element[e].InitArrays();
    }

    // Computation of shape functions and volume

    for (unsigned int e = 0; e < element.size(); e++)
    {
        // Load coordinates integration points and weightings

        unsigned int nen = element[e].connectivity.size();
        vector<vector<double>> xn;
        xn.resize(3);
        for (unsigned int i1 = 0; i1 < 3; i1++)
        {
            xn[i1].resize(nen);
        }
        for (unsigned int i1 = 0; i1 < nen; i1++)
        {
            unsigned int n = element[e].connectivity[i1];
            xn[0][i1] = node[n].icoor[0];
            xn[1][i1] = node[n].icoor[1];
            xn[2][i1] = node[n].icoor[2];
        }

        // Load coordinates integration points and weightings

        if (element[e].nen == 4 || element[e].nen == 10)
        {
            vector<vector<double>> txi;
            txi.resize(5);
            for (unsigned int i1 = 0; i1 < 5; i1++)
            {
                txi[i1].resize(element[e].lint);
            }
            solution.Tint3D(element[e].itype, element[e].lint, txi);
            for (unsigned int l = 0; l < element[e].lint; l++)
            {
                element[e].TetShapeFunctions(l, element[e].nen, txi, xn);
            }
        }
        else
        {
            vector<vector<double>> xi;
            xi.resize(4);
            for (unsigned int i1 = 0; i1 < 4; i1++)
            {
                xi[i1].resize(element[e].lint);
            }
            solution.Int3D(element[e].itype, xi);
            for (unsigned int l = 0; l < element[e].lint; l++)
            {
                element[e].HexShapeFunctions(l, element[e].nen, xi, xn);
            }
        }
    }

    // Determine adjacend elements to each node

    for (unsigned int e = 0; e < element.size(); e++)
    {
        element[e].NodalNeighborElements(e);
    }

    // Define nodes in the vicinity of each node

    for (unsigned int n = 0; n < node.size(); n++)
    {
        node[n].NodalDomain(n);
    }

    // Normal vector and area contribution of each node

    for (unsigned int n = 0; n < node.size(); n++)
    {
        node[n].InitialNormalVectorAndArea(n);
    }

    // Maximal area contribution

    double maxArea = 0.0;
    for (unsigned int n = 0; n < node.size(); n++)
    {
        node[n].MaxArea(maxArea);
    }

    // Check node on surface

    double tol = 1e-6;
    for (unsigned int n = 0; n < node.size(); n++)
    {
        node[n].NodeOnSurface(maxArea, tol);
    }

    // Displacement driven homogenization: Displacement on surface nodes and nodes linked to Lagrange multiplier

    if (solution.HomType != "Zero")
    {
        for (unsigned int n = 0; n < node.size(); n++)
        {
            node[n].DirichletBoundaryHomogenization();
        }

        if (solution.HomMethod == "Lagrange")
        {
            for (unsigned int n = 0; n < node.size(); n++)
            {
                node[n].LagrangeDomain(n);
            }
        }
    }

    // Define Lagrange nodes in the vicinity of each node

    if (solution.HomMethod == "Lagrange")
    {
        for (unsigned int n = 0; n < node.size(); n++)
        {
            node[n].ListLagrange2Node(n);
        }
    }

    // Resize equation number

    solution.ResizeEquationNumber();
}