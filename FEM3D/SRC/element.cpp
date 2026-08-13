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
#include <vector>
#include <fstream>
#include <string>
#include <assert.h>
#include <stdlib.h>

using namespace std;

extern vector<Node> node;
extern vector<Element> element;
extern vector<Material> material;
extern Solution solution;

extern int ndf, t;
extern double Time;

extern string filename;

void Element::InitArrays()
{

    // Initialization of deformation gradient and increment of deformation gradient

    tenF.resize(3);
    tendF.resize(3);
    for (unsigned int i1 = 0; i1 < 3; i1++)
    {
        tenF[i1].resize(3);
        tendF[i1].resize(3);
        for (unsigned int i2 = 0; i2 < 3; i2++)
        {
            tenF[i1][i2].resize(lint);
            tendF[i1][i2].resize(lint);
        }
    }

    // Initialization of determinant of deformation gradient and determinant of increment of deformation gradient

    detF.resize(lint);
    detdF.resize(lint);

    // Initialization of Cauchy stress tensor

    Sigma.resize(6);
    for (unsigned int i1 = 0; i1 < 6; i1++)
    {
        Sigma[i1].resize(lint);
    }

    // Initialization of shape functions

    shp.resize(4);
    for (unsigned int i1 = 0; i1 < 4; i1++)
    {
        shp[i1].resize(nen);
        for (unsigned int i2 = 0; i2 < nen; i2++)
        {
            shp[i1][i2].resize(lint);
        }
    }

    // Initialization of volume

    vol.resize(lint);
}
void Element::NodalNeighborElements(unsigned int e)
{

    // Nodes within the element

    for (unsigned int i1 = 0; i1 < connectivity.size(); i1++)
    {
        unsigned int n = connectivity[i1];
        node[n].NeighborElements.push_back(e);
    }
}
void Element::Kinematics()
{

    // Define initial deformation gradient
    if (solution.diffkind == "Lagrange")
    {

        // Deformation gradient

        for (unsigned int l = 0; l < lint; l++)
        {
            for (unsigned int i1 = 0; i1 < 3; i1++)
            {
                for (unsigned int i2 = 0; i2 < 3; i2++)
                {
                    tenF[i1][i2][l] = 0.0;
                }
            }
            for (unsigned int i1 = 0; i1 < nen; i1++)
            {

                unsigned int n = connectivity[i1];

                // Displacement gradient with respet to initial configuration

                tenF[0][0][l] += shp[0][i1][l] * node[n].disp[0];
                tenF[1][0][l] += shp[0][i1][l] * node[n].disp[1];
                tenF[2][0][l] += shp[0][i1][l] * node[n].disp[2];

                tenF[0][1][l] += shp[1][i1][l] * node[n].disp[0];
                tenF[1][1][l] += shp[1][i1][l] * node[n].disp[1];
                tenF[2][1][l] += shp[1][i1][l] * node[n].disp[2];

                tenF[0][2][l] += shp[2][i1][l] * node[n].disp[0];
                tenF[1][2][l] += shp[2][i1][l] * node[n].disp[1];
                tenF[2][2][l] += shp[2][i1][l] * node[n].disp[2];
            }

            tenF[0][0][l] += 1.0;
            tenF[1][1][l] += 1.0;
            tenF[2][2][l] += 1.0;

            // Determinant of deformation gradient

            detF[l] = tenF[0][0][l] * tenF[1][1][l] * tenF[2][2][l] + tenF[0][1][l] * tenF[1][2][l] * tenF[2][0][l] + tenF[0][2][l] * tenF[1][0][l] * tenF[2][1][l] - tenF[0][0][l] * tenF[1][2][l] * tenF[2][1][l] - tenF[0][1][l] * tenF[1][0][l] * tenF[2][2][l] - tenF[0][2][l] * tenF[1][1][l] * tenF[2][0][l];
        }
    }
    else if (solution.diffkind == "UpdatedLagrange")
    {

        // Increment deformation gradient

        for (unsigned int l = 0; l < lint; l++)
        {
            for (unsigned int i1 = 0; i1 < 3; i1++)
            {
                for (unsigned int i2 = 0; i2 < 3; i2++)
                {
                    tendF[i1][i2][l] = 0.0;
                }
            }
            for (unsigned int i1 = 0; i1 < nen; i1++)
            {

                unsigned int n = connectivity[i1];

                // Displacement gradient inverse

                tendF[0][0][l] += shp[0][i1][l] * (node[n].disp[0] - node[n].pdisp[0]);
                tendF[1][0][l] += shp[0][i1][l] * (node[n].disp[1] - node[n].pdisp[1]);
                tendF[2][0][l] += shp[0][i1][l] * (node[n].disp[2] - node[n].pdisp[2]);

                tendF[0][1][l] += shp[1][i1][l] * (node[n].disp[0] - node[n].pdisp[0]);
                tendF[1][1][l] += shp[1][i1][l] * (node[n].disp[1] - node[n].pdisp[1]);
                tendF[2][1][l] += shp[1][i1][l] * (node[n].disp[2] - node[n].pdisp[2]);

                tendF[0][2][l] += shp[2][i1][l] * (node[n].disp[0] - node[n].pdisp[0]);
                tendF[1][2][l] += shp[2][i1][l] * (node[n].disp[1] - node[n].pdisp[1]);
                tendF[2][2][l] += shp[2][i1][l] * (node[n].disp[2] - node[n].pdisp[2]);
            }

            tendF[0][0][l] += 1.0;
            tendF[1][1][l] += 1.0;
            tendF[2][2][l] += 1.0;

            // Deformation gradient inverse

            double Finv[3][3], F[3][3];
            for (unsigned int i1 = 0; i1 < 3; i1++)
            {
                for (unsigned int i2 = 0; i2 < 3; i2++)
                {
                    Finv[i1][i2] = 0.0;
                }
            }

            Finv[0][0] += 1.0;
            Finv[1][1] += 1.0;
            Finv[2][2] += 1.0;

            // Deformation gradient

            inverse3(Finv, F);
            for (unsigned int i = 0; i < 3; i++)
            {
                for (unsigned int j = 0; j < 3; j++)
                {
                    tenF[i][j][l] = F[i][j];
                }
            }

            // Determinant of increment of deformation gradient

            detdF[l] = tendF[0][0][l] * tendF[1][1][l] * tendF[2][2][l] + tendF[0][1][l] * tendF[1][2][l] * tendF[2][0][l] + tendF[0][2][l] * tendF[1][0][l] * tendF[2][1][l] - tendF[0][0][l] * tendF[1][2][l] * tendF[2][1][l] - tendF[0][1][l] * tendF[1][0][l] * tendF[2][2][l] - tendF[0][2][l] * tendF[1][1][l] * tendF[2][0][l];

            // Determinant of deformation gradient

            detF[l] = tenF[0][0][l] * tenF[1][1][l] * tenF[2][2][l] + tenF[0][1][l] * tenF[1][2][l] * tenF[2][0][l] + tenF[0][2][l] * tenF[1][0][l] * tenF[2][1][l] - tenF[0][0][l] * tenF[1][2][l] * tenF[2][1][l] - tenF[0][1][l] * tenF[1][0][l] * tenF[2][2][l] - tenF[0][2][l] * tenF[1][1][l] * tenF[2][0][l];
        }
    }
}
vector<double> Element::push_vector(vector<double> v, double fi[3][3])
{
    vector<double> t(3);

    t[0] = v[0] * fi[0][0] + v[1] * fi[1][0] + v[2] * fi[2][0];
    t[1] = v[0] * fi[0][1] + v[1] * fi[1][1] + v[2] * fi[2][1];
    t[2] = v[0] * fi[0][2] + v[1] * fi[1][2] + v[2] * fi[2][2];

    return t;
}
void Element::inverse3(double f[][3], double fi[][3])
{
    double detf = f[0][0] * f[1][1] * f[2][2] + f[0][1] * f[1][2] * f[2][0] + f[0][2] * f[1][0] * f[2][1] - f[0][0] * f[1][2] * f[2][1] - f[0][1] * f[1][0] * f[2][2] - f[0][2] * f[1][1] * f[2][0];
    double idetf = 1.0 / detf;

    fi[0][0] = (f[1][1] * f[2][2] - f[2][1] * f[1][2]) * idetf;
    fi[0][1] = -(f[0][1] * f[2][2] - f[2][1] * f[0][2]) * idetf;
    fi[0][2] = (f[0][1] * f[1][2] - f[1][1] * f[0][2]) * idetf;
    fi[1][0] = -(f[1][0] * f[2][2] - f[1][2] * f[2][0]) * idetf;
    fi[1][1] = (f[0][0] * f[2][2] - f[2][0] * f[0][2]) * idetf;
    fi[1][2] = -(f[0][0] * f[1][2] - f[1][0] * f[0][2]) * idetf;
    fi[2][0] = (f[1][0] * f[2][1] - f[2][0] * f[1][1]) * idetf;
    fi[2][1] = -(f[0][0] * f[2][1] - f[2][0] * f[0][1]) * idetf;
    fi[2][2] = (f[0][0] * f[1][1] - f[1][0] * f[0][1]) * idetf;
}
void Element::update_hist()
{
}
void Element::UpdateShapeFunction(unsigned int ngp, unsigned int nen)
{
}
void Element::UpdateVolume(unsigned int ngp)
{
}
void Element::NeighborElementsToNodes(unsigned int e)
{

    for (unsigned int i1 = 0; i1 < connectivity.size(); i1++)
    {
        unsigned int n = connectivity[i1];
        node[n].NeighborElements.push_back(e);
    }
}
void Element::Average(int p, double AverageStrain[6], double AverageStress[6], double &Volume)
{
}
void Element::TetShapeFunctions(unsigned int l, unsigned int nen, vector<vector<double>> xgl, vector<vector<double>> xn)
{

    // Compute shape functions and their natural coord. derivatives

    if (nen == 4)
    {
        // Compute shape functions and their natural coord. derivatives

        shp[0][0][l] = -1.0;
        shp[0][1][l] = 1.0;
        shp[0][2][l] = 0.0;
        shp[0][3][l] = 0.0;

        shp[1][0][l] = -1.0;
        shp[1][1][l] = 0.0;
        shp[1][2][l] = 1.0;
        shp[1][3][l] = 0.0;

        shp[2][0][l] = -1.0;
        shp[2][1][l] = 0.0;
        shp[2][2][l] = 0.0;
        shp[2][3][l] = 1.0;

        shp[3][0][l] = 1.0 - xgl[0][l] - xgl[1][l] - xgl[2][l];
        shp[3][1][l] = xgl[0][l];
        shp[3][2][l] = xgl[1][l];
        shp[3][3][l] = xgl[2][l];
    }
    else if (nen == 10)
    {

        double lambda = 1.0 - xgl[0][l] - xgl[1][l] - xgl[2][l];

        shp[3][0][l] = lambda * (2.0 * lambda - 1.0);
        shp[3][1][l] = xgl[0][l] * (2.0 * xgl[0][l] - 1.0);
        shp[3][2][l] = xgl[1][l] * (2.0 * xgl[1][l] - 1.0);
        shp[3][3][l] = xgl[2][l] * (2.0 * xgl[2][l] - 1.0);
        shp[3][4][l] = 4.0 * xgl[0][l] * lambda;
        shp[3][5][l] = 4.0 * xgl[0][l] * xgl[1][l];
        shp[3][6][l] = 4.0 * xgl[1][l] * lambda;
        shp[3][7][l] = 4.0 * xgl[2][l] * lambda;
        shp[3][8][l] = 4.0 * xgl[0][l] * xgl[2][l];
        shp[3][9][l] = 4.0 * xgl[1][l] * xgl[2][l];

        shp[0][0][l] = -4.0 * lambda + 1.0;
        shp[0][1][l] = 4.0 * xgl[0][l] - 1.0;
        shp[0][2][l] = 0.0;
        shp[0][3][l] = 0.0;
        shp[0][4][l] = 4.0 * lambda - 4.0 * xgl[0][l];
        shp[0][5][l] = 4.0 * xgl[1][l];
        shp[0][6][l] = -4.0 * xgl[1][l];
        shp[0][7][l] = -4.0 * xgl[2][l];
        shp[0][8][l] = 4.0 * xgl[2][l];
        shp[0][9][l] = 0.0;

        shp[1][0][l] = -4.0 * lambda + 1.0;
        shp[1][1][l] = 0.0;
        shp[1][2][l] = 4.0 * xgl[1][l] - 1.0;
        shp[1][3][l] = 0.0;
        shp[1][4][l] = -4.0 * xgl[0][l];
        shp[1][5][l] = 4.0 * xgl[0][l];
        shp[1][6][l] = 4.0 * lambda - 4.0 * xgl[1][l];
        shp[1][7][l] = -4.0 * xgl[2][l];
        shp[1][8][l] = 0.0;
        shp[1][9][l] = 4.0 * xgl[2][l];

        shp[2][0][l] = -4.0 * lambda - 1.0;
        shp[2][1][l] = 0.0;
        shp[2][2][l] = 0.0;
        shp[2][3][l] = 4.0 * xgl[2][l] - 1.0;
        shp[2][4][l] = -4.0 * xgl[0][l];
        shp[2][5][l] = 0.0;
        shp[2][6][l] = -4.0 * xgl[1][l];
        shp[2][7][l] = 4.0 * xgl[2][l] - 4.0 * xgl[2][l];
        shp[2][8][l] = 4.0 * xgl[0][l];
        shp[2][9][l] = 4.0 * xgl[1][l];
    }

    // Compute jacobian transformation

    double xs[3][3];
    for (unsigned int i1 = 0; i1 < 3; i1++)
    {
        for (unsigned int i2 = 0; i2 < 3; i2++)
        {
            xs[i2][i1] = 0.0;
            for (unsigned int i3 = 0; i3 < nen; i3++)
            {
                xs[i2][i1] += xn[i2][i3] * shp[i1][i3][l];
            }
        }
    }

    // Compute adjoint to jacobian

    double ad[3][3];
    ad[0][0] = xs[1][1] * xs[2][2] - xs[1][2] * xs[2][1];
    ad[0][1] = xs[2][1] * xs[0][2] - xs[2][2] * xs[0][1];
    ad[0][2] = xs[0][1] * xs[1][2] - xs[0][2] * xs[1][1];

    ad[1][0] = xs[1][2] * xs[2][0] - xs[1][0] * xs[2][2];
    ad[1][1] = xs[2][2] * xs[0][0] - xs[2][0] * xs[0][2];
    ad[1][2] = xs[0][2] * xs[1][0] - xs[0][0] * xs[1][2];

    ad[2][0] = xs[1][0] * xs[2][1] - xs[1][1] * xs[2][0];
    ad[2][1] = xs[2][0] * xs[0][1] - xs[2][1] * xs[0][0];
    ad[2][2] = xs[0][0] * xs[1][1] - xs[0][1] * xs[1][0];

    // Compute determinant of jacobian

    double xsj = xs[0][0] * ad[0][0] + xs[0][1] * ad[1][0] + xs[0][2] * ad[2][0];
    double rxsj = 1.0 / xsj;

    // Compute jacobian inverse

    for (unsigned int i1 = 0; i1 < 3; i1++)
    {
        for (unsigned int i2 = 0; i2 < 3; i2++)
        {
            xs[i1][i2] = ad[i1][i2] * rxsj;
        }
    }

    // Compute derivatives with repect to global coords.

    for (unsigned int i1 = 0; i1 < nen; i1++)
    {
        double c1 = shp[0][i1][l] * xs[0][0] + shp[1][i1][l] * xs[1][0] + shp[2][i1][l] * xs[2][0];
        double c2 = shp[0][i1][l] * xs[0][1] + shp[1][i1][l] * xs[1][1] + shp[2][i1][l] * xs[2][1];
        double c3 = shp[0][i1][l] * xs[0][2] + shp[1][i1][l] * xs[1][2] + shp[2][i1][l] * xs[2][2];

        shp[0][i1][l] = c1;
        shp[1][i1][l] = c2;
        shp[2][i1][l] = c3;
    }

    // Volume

    vol[l] = xsj * xgl[4][l];
}
void Element::HexShapeFunctions(unsigned int l, unsigned int nen, vector<vector<double>> xgl, vector<vector<double>> xn)
{

    // Ordering by vertex-edge-face-interior

    unsigned int ir[27] = {1, 2, 2, 1, 1, 2, 2, 1, 3, 2, 3, 1, 3, 2, 3, 1, 1, 2, 2, 1, 3, 3, 1, 2, 3, 3, 3};
    unsigned int is[27] = {1, 1, 2, 2, 1, 1, 2, 2, 1, 3, 2, 3, 1, 3, 2, 3, 1, 1, 2, 2, 3, 3, 3, 3, 1, 2, 3};
    unsigned int it[27] = {1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 1, 2, 3, 3, 3, 3, 3};

    // Compute shape functions and their natural coord. derivatives

    if (nen == 8)
    {
        // Compute shape functions and their natural coord. derivatives

        double ap1 = 1.0 + xgl[0][l];
        double am1 = 1.0 - xgl[0][l];
        double ap2 = 1.0 + xgl[1][l];
        double am2 = 1.0 - xgl[1][l];
        double ap3 = 1.0 + xgl[2][l];
        double am3 = 1.0 - xgl[2][l];

        // Compute for ( - , - ) values

        double c1 = 0.125 * am1 * am2;
        double c2 = 0.125 * am2 * am3;
        double c3 = 0.125 * am1 * am3;
        shp[0][0][l] = -c2;
        shp[0][1][l] = c2;
        shp[1][0][l] = -c3;
        shp[1][3][l] = c3;
        shp[2][0][l] = -c1;
        shp[2][4][l] = c1;
        shp[3][0][l] = c1 * am3;
        shp[3][4][l] = c1 * ap3;

        // Compute for ( + , + ) values

        c1 = 0.125 * ap1 * ap2;
        c2 = 0.125 * ap2 * ap3;
        c3 = 0.125 * ap1 * ap3;
        shp[0][7][l] = -c2;
        shp[0][6][l] = c2;
        shp[1][5][l] = -c3;
        shp[1][6][l] = c3;
        shp[2][2][l] = -c1;
        shp[2][6][l] = c1;
        shp[3][2][l] = c1 * am3;
        shp[3][6][l] = c1 * ap3;

        // Compute for ( - , + ) values

        c1 = 0.125 * am1 * ap2;
        c2 = 0.125 * am2 * ap3;
        c3 = 0.125 * am1 * ap3;
        shp[0][4][l] = -c2;
        shp[0][5][l] = c2;
        shp[1][4][l] = -c3;
        shp[1][7][l] = c3;
        shp[2][3][l] = -c1;
        shp[2][7][l] = c1;
        shp[3][3][l] = c1 * am3;
        shp[3][7][l] = c1 * ap3;

        // Compute for ( + , - ) values

        c1 = 0.125 * ap1 * am2;
        c2 = 0.125 * ap2 * am3;
        c3 = 0.125 * ap1 * am3;
        shp[0][3][l] = -c2;
        shp[0][2][l] = c2;
        shp[1][1][l] = -c3;
        shp[1][2][l] = c3;
        shp[2][1][l] = -c1;
        shp[2][5][l] = c1;
        shp[3][1][l] = c1 * am3;
        shp[3][5][l] = c1 * ap3;
    }
    else if (nen == 27)
    {

        // Set 1-d shape functions for each local direction

        double nr[3], ns[3], nt[3], dr[3], ds[3], dt[3];

        nr[0] = 0.5 * xgl[0][l] * (xgl[0][l] - 1.0);
        nr[1] = 0.5 * xgl[0][l] * (xgl[0][l] + 1.0);
        nr[2] = 1.0 - xgl[0][l] * xgl[0][l];

        ns[0] = 0.5 * xgl[1][l] * (xgl[1][l] - 1.0);
        ns[1] = 0.5 * xgl[1][l] * (xgl[1][l] + 1.0);
        ns[2] = 1.0 - xgl[1][l] * xgl[1][l];

        nt[0] = 0.5 * xgl[2][l] * (xgl[2][l] - 1.0);
        nt[1] = 0.5 * xgl[2][l] * (xgl[2][l] + 1.0);
        nt[2] = 1.0 - xgl[2][l] * xgl[2][l];

        dr[0] = xgl[0][l] - 0.50;
        dr[1] = xgl[0][l] + 0.50;
        dr[2] = -xgl[0][l] - xgl[0][l];

        ds[0] = xgl[1][l] - 0.50;
        ds[1] = xgl[1][l] + 0.50;
        ds[2] = -xgl[1][l] - xgl[1][l];

        dt[0] = xgl[2][l] - 0.50;
        dt[1] = xgl[2][l] + 0.50;
        dt[2] = -xgl[2][l] - xgl[2][l];

        // Set local 3-d shape functions

        for (unsigned int i1 = 0; i1 < nen; i1++)
        {
            shp[0][i1][l] = dr[ir[i1]] * ns[is[i1]] * nt[it[i1]];
            shp[1][i1][l] = nr[ir[i1]] * ds[is[i1]] * nt[it[i1]];
            shp[2][i1][l] = nr[ir[i1]] * ns[is[i1]] * dt[it[i1]];
            shp[3][i1][l] = nr[ir[i1]] * ns[is[i1]] * nt[it[i1]];
        }
    }

    // Compute jacobian transformation

    double xs[3][3];
    for (unsigned int i1 = 0; i1 < 3; i1++)
    {
        for (unsigned int i2 = 0; i2 < 3; i2++)
        {
            xs[i2][i1] = 0.0;
            for (unsigned int i3 = 0; i3 < nen; i3++)
            {
                xs[i2][i1] += xn[i2][i3] * shp[i1][i3][l];
            }
        }
    }

    // Compute adjoint to jacobian

    double ad[3][3];
    ad[0][0] = xs[1][1] * xs[2][2] - xs[1][2] * xs[2][1];
    ad[0][1] = xs[2][1] * xs[0][2] - xs[2][2] * xs[0][1];
    ad[0][2] = xs[0][1] * xs[1][2] - xs[0][2] * xs[1][1];

    ad[1][0] = xs[1][2] * xs[2][0] - xs[1][0] * xs[2][2];
    ad[1][1] = xs[2][2] * xs[0][0] - xs[2][0] * xs[0][2];
    ad[1][2] = xs[0][2] * xs[1][0] - xs[0][0] * xs[1][2];

    ad[2][0] = xs[1][0] * xs[2][1] - xs[1][1] * xs[2][0];
    ad[2][1] = xs[2][0] * xs[0][1] - xs[2][1] * xs[0][0];
    ad[2][2] = xs[0][0] * xs[1][1] - xs[0][1] * xs[1][0];

    // Compute determinant of jacobian

    double xsj = xs[0][0] * ad[0][0] + xs[0][1] * ad[1][0] + xs[0][2] * ad[2][0];
    double rxsj = 1.0 / xsj;

    // Compute jacobian inverse

    for (unsigned int i1 = 0; i1 < 3; i1++)
    {
        for (unsigned int i2 = 0; i2 < 3; i2++)
        {
            xs[i1][i2] = ad[i1][i2] * rxsj;
        }
    }

    // Compute derivatives with repect to global coords.

    for (unsigned int i1 = 0; i1 < nen; i1++)
    {
        double c1 = shp[0][i1][l] * xs[0][0] + shp[1][i1][l] * xs[1][0] + shp[2][i1][l] * xs[2][0];
        double c2 = shp[0][i1][l] * xs[0][1] + shp[1][i1][l] * xs[1][1] + shp[2][i1][l] * xs[2][1];
        double c3 = shp[0][i1][l] * xs[0][2] + shp[1][i1][l] * xs[1][2] + shp[2][i1][l] * xs[2][2];

        shp[0][i1][l] = c1;
        shp[1][i1][l] = c2;
        shp[2][i1][l] = c3;
    }

    // Volume

    vol[l] = xsj * xgl[3][l];
}
