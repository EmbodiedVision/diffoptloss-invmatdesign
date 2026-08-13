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
#include "AceGen.h"
#include "Material.h"
#include "Solution.h"
#include "Lagrange.h"
#include <fstream>
#include <string>
#include <assert.h>
#include <stdlib.h>
#include <math.h>

extern unsigned int ndf, numnp;
extern double Time, sload;
extern string filename;
extern vector<Node> node;
extern vector<Element> element;
extern vector<Material> material;
extern vector<Lagrange> lagrange;
extern Solution solution;
extern AceGen acegen;

extern string filename;

void Node::initialize(int n, int t, int type, int isw)
{

    // Parameters of time integration scheme

    vector<double> tparam = solution.get_tparam();
    double prop;

    // Central difference time integration

    if (type == 0)
    {

        // Increment of displacements on the boundary of the new time step

        if (btype[0] < 0)
        {
            prop = solution.prop(solution.ptype, n, 0, Time);
            bknown[0] = prop * bdisp[0] - disp[0];
        }
        if (btype[1] < 0)
        {
            prop = solution.prop(solution.ptype, n, 1, Time);
            bknown[1] = prop * bdisp[1] - disp[1];
        }
        if (btype[2] < 0)
        {
            prop = solution.prop(solution.ptype, n, 2, Time);
            bknown[2] = prop * bdisp[2] - disp[2];
        }
    }

    // Newmark time integration

    else if (type >= 1)
    {

        // Increment of displacements on the boundary

        if (btype[0] < 0)
        {
            prop = solution.prop(solution.ptype, n, 0, Time);
            bknown[0] = prop * bdisp[0] - disp[0];
        }
        if (btype[1] < 0)
        {
            prop = solution.prop(solution.ptype, n, 1, Time);
            bknown[1] = prop * bdisp[1] - disp[1];
        }
        if (btype[2] < 0)
        {
            prop = solution.prop(solution.ptype, n, 2, Time);
            bknown[2] = prop * bdisp[2] - disp[2];
        }

        // Displacements on the boundary

        if (ltype[0] > 0)
        {
            prop = solution.prop(solution.ptype, n, 0, Time);
            bknown[0] = disp[0] - prop * bdisp[0];
        }
        if (ltype[1] > 0)
        {
            prop = solution.prop(solution.ptype, n, 1, Time);
            bknown[1] = disp[1] - prop * bdisp[1];
        }
        if (ltype[2] > 0)
        {
            prop = solution.prop(solution.ptype, n, 1, Time);
            bknown[2] = disp[1] - prop * bdisp[2];
        }

        // Initialize increment of displacements

        du[0] = 0.0;
        du[1] = 0.0;
        du[2] = 0.0;

        // Initial velocity and acceleration

        if (type == 1)
        {
            double aux[3];
            aux[0] = tparam[4] * velo[0] + tparam[5] * acce[0];
            aux[1] = tparam[4] * velo[1] + tparam[5] * acce[1];
            aux[2] = tparam[4] * velo[2] + tparam[5] * acce[2];

            double c1;
            acce[0] = tparam[1] * velo[0] + tparam[2] * acce[0];
            c1 = -1.0;
            acce[0] = c1 * acce[0];
            acce[1] = tparam[1] * velo[1] + tparam[2] * acce[1];
            c1 = -1.0;
            acce[1] = c1 * acce[1];
            acce[2] = tparam[1] * velo[2] + tparam[2] * acce[2];
            c1 = -1.0;
            acce[2] = c1 * acce[2];

            velo[0] = aux[0];
            velo[1] = aux[1];
            velo[2] = aux[2];
        }
        else if (type == 2)
        {
            velo[0] = 0.0;
            velo[1] = 0.0;
            velo[2] = 0.0;
        }
    }

    // Initial nodal boundary force

    for (unsigned int i = 0; i < ndf; i++)
    {
        if (bforce[i] != 0)
            istf = 1;
    }
}
void Node::initial_boundary(string type, int n, double dt)
{

    // Displacements increments on the boundary at t=t0: dub_0

    double Time1 = Time + dt;

    double u1[3], uminus1[3];
    double c1 = dt * dt;
    c1 = 1.0 / c1;
    double prop1;

    if (btype[0] < 0)
    {
        prop1 = solution.prop(solution.ptype, n, 0, Time1);
        u1[0] = prop1 * bdisp[0];
        uminus1[0] = u1[0] - 2.0 * dt * velo[0];
        du[0] = disp[0] - uminus1[0];
        acce[0] = c1 * (u1[0] - 2.0 * disp[0] + uminus1[0]);
    }
    if (btype[1] < 0)
    {
        prop1 = solution.prop(solution.ptype, n, 1, Time1);
        u1[1] = prop1 * bdisp[1];
        uminus1[1] = u1[1] - 2.0 * dt * velo[1];
        du[1] = disp[1] - uminus1[1];
        acce[1] = c1 * (u1[1] - 2.0 * disp[1] + uminus1[1]);
    }
    if (btype[2] < 0)
    {
        prop1 = solution.prop(solution.ptype, n, 2, Time1);
        u1[2] = prop1 * bdisp[2];
        uminus1[2] = u1[2] - 2.0 * dt * velo[2];
        du[2] = disp[2] - uminus1[2];
        acce[2] = c1 * (u1[2] - 2.0 * disp[2] + uminus1[2]);
    }
}
void Node::initial(double dt, vector<double> x)
{

    double uminus1[3];
    double c1 = 0.5 * dt * dt;

    // Acceleration and displcement increment at time a_0

    if (btype[0] > 0)
    {
        acce[0] = dt * dt * x[btype[0] - 1];
        uminus1[0] = disp[0] - dt * velo[0] + c1 * acce[0];
        du[0] = disp[0] - uminus1[0];
    }
    if (btype[1] > 0)
    {
        acce[1] = dt * dt * x[btype[1] - 1];
        uminus1[1] = disp[1] - dt * velo[1] + c1 * acce[1];
        du[1] = disp[1] - uminus1[1];
    }
    if (btype[2] > 0)
    {
        acce[2] = dt * dt * x[btype[2] - 1];
        uminus1[2] = disp[2] - dt * velo[2] + c1 * acce[2];
        du[2] = disp[2] - uminus1[2];
    }
}
void Node::InitializeCRSFormat(unsigned int neq)
{

    // Counter for non zero entries

    ofstream output(filename.c_str(), ios::app);

    // Displacement contributions

    for (unsigned int i1 = 0; i1 < ndf; i1++)
    {

        // Displacement rows

        if (btype[i1] > 0)
        {
            // Connectivity of equation number for pure displacements

            vector<int> eqn_list(neq, 0);
            for (unsigned int i2 = 0; i2 < ListNodes.size(); i2++)
            {
                // Node connected to current node

                int nn = ListNodes[i2];

                // Enter non zero entries into column indicator

                for (unsigned int i4 = 0; i4 < ndf; i4++)
                {
                    if (node[nn].btype[i4] > 0)
                    {
                        eqn_list[node[nn].btype[i4] - 1] = 1;
                    }
                }
            }

            // Connectivity of equation number for Lagrange multiplier nodes

            // Jens: No Lagrange nodes for my case.
            for (unsigned int i2 = 0; i2 < Lagrange2Node.size(); i2++)
            {
                unsigned int lam = Lagrange2Node[i2];

                if (node[lam].ltype[i1] > 0)
                {
                    eqn_list[node[lam].ltype[i1] - 1] = 1;
                }
            }

            // Add number of nodes into the row pointer

            unsigned int count = 0;
            for (unsigned int i2 = 0; i2 < neq; i2++)
            {
                if (eqn_list[i2] == 1)
                {
                    solution.ja.push_back(i2 + 1);
                    count++;
                }
            }
            solution.ia[btype[i1]] = count;
        }

        if (ltype[i1] > 0)
        {

            // Connectivity of equation number for pure displacements
            vector<int> eqn_list(neq, 0);
            for (unsigned int i2 = 0; i2 < Nodes2Lagrange.size(); i2++)
            {

                // Node connected to Lagrange multiplier

                int nn = Nodes2Lagrange[i2];

                // Enter non zero entries into column indicator

                if (node[nn].btype[i1] > 0)
                {
                    eqn_list[node[nn].btype[i1] - 1] = 1;
                }
            }

            // Add number of nodes into the row pointer

            unsigned int count = 0;
            for (unsigned int i2 = 0; i2 < neq; i2++)
            {
                if (eqn_list[i2] == 1)
                {
                    solution.ja.push_back(i2 + 1);
                    count++;
                }
            }
            solution.ia[ltype[i1]] = count;
        }
    }
}
void Node::project(int nI)
{
    // Initialize data

    double volume = 0;
    for (unsigned int i1 = 0; i1 < 10; i1++)
    {
        stress[i1] = 0;
    }

    ofstream output(filename.c_str(), ios::app);

    // Loop over all material points in the influence domain

    for (unsigned int i1 = 0; i1 < NeighborElements.size(); i1++)
    {

        unsigned int e = NeighborElements[i1];

        // Load stress

        unsigned int m = element[e].mtype - 1;
        material[m].stress(m, e, element[e].lint, 0);

        for (unsigned int l = 0; l < element[e].lint; l++)
        {
            // von Mises stress at point

            double mises = element[e].Sigma[0][l] * element[e].Sigma[0][l] + element[e].Sigma[1][l] * element[e].Sigma[1][l] + element[e].Sigma[2][l] * element[e].Sigma[2][l] - element[e].Sigma[0][l] * element[e].Sigma[1][l] - element[e].Sigma[1][l] * element[e].Sigma[2][l] - element[e].Sigma[0][l] * element[e].Sigma[2][l] + 3 * (element[e].Sigma[3][l] * element[e].Sigma[3][l] + element[e].Sigma[4][l] * element[e].Sigma[4][l] + element[e].Sigma[5][l] * element[e].Sigma[5][l]);
            if (mises > 1e-6)
            {
                mises = sqrt(mises);
            }
            else
            {
                mises = 0.0;
            }

            // Project stresses

            stress[0] += element[e].shp[3][i1][l] * element[e].Sigma[0][l] * element[e].vol[l];
            stress[1] += element[e].shp[3][i1][l] * element[e].Sigma[1][l] * element[e].vol[l];
            stress[2] += element[e].shp[3][i1][l] * element[e].Sigma[2][l] * element[e].vol[l];

            stress[3] += element[e].shp[3][i1][l] * element[e].Sigma[3][l] * element[e].vol[l];
            stress[4] += element[e].shp[3][i1][l] * element[e].Sigma[4][l] * element[e].vol[l];
            stress[5] += element[e].shp[3][i1][l] * element[e].Sigma[5][l] * element[e].vol[l];

            stress[9] += element[e].shp[3][i1][l] * mises * element[e].vol[l];

            // Project volume

            volume += element[e].shp[3][i1][l] * element[e].vol[l];
        }
    }

    for (int i1 = 0; i1 < 10; i1++)
    {
        if (volume == 0)
            stress[i1] = 0.0;
        else
            stress[i1] = stress[i1] / volume;
    }
}
void Node::ContactRigid(unsigned int nI)
{

    ofstream output(filename.c_str(), ios::app);

    unsigned int nrs = solution.ContactMethod.size();
    unsigned int nn = ListNodes.size();
    vector<vector<double>> Tangent(2, vector<double>(nn * ndf, 0));

    for (unsigned int i = 0; i < nrs; i++)
    {

        // Coordinates of rigid line

        double x1[2], x2[2];
        x1[0] = solution.ContactPair[i][0];
        x1[1] = solution.ContactPair[i][1];
        x2[0] = solution.ContactPair[i][2];
        x2[1] = solution.ContactPair[i][3];

        // Update of rigid line

        double velocity[2];
        velocity[0] = solution.ContactMethod[i][2];
        velocity[1] = solution.ContactMethod[i][3];
        x1[0] += velocity[0] * Time;
        x1[1] += velocity[1] * Time;
        x2[0] += velocity[0] * Time;
        x2[1] += velocity[1] * Time;

        // Nodal Coordinate

        double xs[2];
        xs[0] = acoor[0];
        xs[1] = acoor[1];

        // Base vectors of rigid surface

        double aRigid[2];
        aRigid[0] = x2[0] - x1[0];
        aRigid[1] = x2[1] - x1[1];
        double nRigid[2];
        nRigid[0] = x2[1] - x1[1];
        nRigid[1] = x1[0] - x2[0];
        double NormRigid = sqrt(nRigid[0] * nRigid[0] + nRigid[1] * nRigid[1]);
        nRigid[0] = nRigid[0] / NormRigid;
        nRigid[1] = nRigid[1] / NormRigid;

        // Projection of node on rigid surface

        double tau = ((xs[1] - x1[1]) - aRigid[1] / aRigid[0] * (xs[0] - x1[0])) / (aRigid[1] / aRigid[0] * nRigid[0] - nRigid[1]);
        double xc[2];
        xc[0] = xs[0] + tau * nRigid[0];
        xc[1] = xs[1] + tau * nRigid[1];

        // Penetration

        double gn = (xs[0] - xc[0]) * nRigid[0] + (xs[1] - xc[1]) * nRigid[1];

        // Area

        double area = sqrt(normal[0] * normal[0] + normal[1] * normal[1]);

        // Active set

        int istgn = -1;
        if (gn < 0.0)
            istgn = 1;
        else if (gn == 0.0)
            istgn = 0;

        // Force vector

        if (istgn == 1)
        {

            // Initialization

            vector<double> Force(ndf, 0);
            unsigned int nn = ListNodes.size();
            vector<vector<double>> Tangent(2, vector<double>(nn * ndf, 0));

            // Force at point

            double cn = solution.ContactMethod[i][1];
            Force[0] = -cn * gn * nRigid[0] * area;
            Force[1] = -cn * gn * nRigid[1] * area;

            // Assign location in tangent

            int jj = AssignInTangent(nI);

            // Material part

            Tangent[0][jj + 0] += cn * nRigid[0] * nRigid[0] * area;
            Tangent[0][jj + 1] += cn * nRigid[0] * nRigid[1] * area;

            Tangent[1][jj + 0] += cn * nRigid[1] * nRigid[0] * area;
            Tangent[1][jj + 1] += cn * nRigid[1] * nRigid[1] * area;

            // Assemble

            solution.NodalAssembly(nI, ndf, Force, Tangent);
        }
    }
}
void Node::ForceVector(int nI, unsigned int typeLoad, vector<vector<double>> &Tangent)
{
}
void Node::initialForce(int n, double areaX, double areaY)
{

    if (fabs(bforce[0]) > 1e-6)
    {
        bforce[0] = areaX * sload;
    }
    if (fabs(bforce[1]) > 1e-6)
    {
        bforce[1] = areaY * sload;
    }
}
void Node::BoundaryForce(unsigned int nI)
{
}
void Node::ContactForce(unsigned int nI)
{
}
void Node::SolidStaticAceGen(unsigned int nI)
{

    // Initialize arrays

    vector<double> Force(ndf, 0);
    unsigned int nn = ListNodes.size();
    vector<vector<double>> Tangent(3, vector<double>(nn * ndf, 0));
    vector<double> Boundary(nn * ndf, 0);
    bool modify = false;

    // Loop over neighboring elements

    for (unsigned int i1 = 0; i1 < NeighborElements.size(); i1++)
    {
        // Corresponding element number

        unsigned int e = NeighborElements[i1];

        // Corresponing node number in element

        unsigned int nP = AssignNodeInList(nI, e);

        // Loop over integration points

        for (unsigned int l = 0; l < element[e].lint; l++)
        {

            // Residual and tangent with respect to deformation gradient

            double shpI[3], F[3][3], d[3], R[3], T[3][3][3];

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

            if (solution.diffkind == "Lagrange")
            {
                F[0][0] = element[e].tenF[0][0][l];
                F[0][1] = element[e].tenF[0][1][l];
                F[0][2] = element[e].tenF[0][2][l];
                F[1][0] = element[e].tenF[1][0][l];
                F[1][1] = element[e].tenF[1][1][l];
                F[1][2] = element[e].tenF[1][2][l];
                F[2][0] = element[e].tenF[2][0][l];
                F[2][1] = element[e].tenF[2][1][l];
                F[2][2] = element[e].tenF[2][2][l];
            }
            else if (solution.diffkind == "UpdatedLagrange")
            {
                F[0][0] = element[e].tendF[0][0][l];
                F[0][1] = element[e].tendF[0][1][l];
                F[0][2] = element[e].tendF[0][2][l];
                F[1][0] = element[e].tendF[1][0][l];
                F[1][1] = element[e].tendF[1][1][l];
                F[1][2] = element[e].tendF[1][2][l];
                F[2][0] = element[e].tendF[2][0][l];
                F[2][1] = element[e].tendF[2][1][l];
                F[2][2] = element[e].tendF[2][2][l];
            }

            // Residual and tangent

            if (MaterialModel == "NeoHooke")
            {
                acegen.NeoHooke3d(shpI, F, d, volume, R, T);
            }
            else if (MaterialModel == "LinearElasticity")
            {
                acegen.LinearElasticity3d(shpI, F, d, volume, R, T);
            }

            // Assemble tangent

            for (unsigned int i3 = 0; i3 < element[e].connectivity.size(); i3++)
            {

                // Load corresponding node

                unsigned int nJ = element[e].connectivity[i3];

                // Shape function for trail function

                double shpJ[3];
                shpJ[0] = element[e].shp[0][i3][l];
                shpJ[1] = element[e].shp[1][i3][l];
                shpJ[2] = element[e].shp[2][i3][l];

                // Assign location in tangent

                int jj = AssignInTangent(nJ);

                // Request modify residual

                for (unsigned int i4 = 0; i4 < ndf; i4++)
                {
                    if (node[nJ].btype[i4] < 0)
                    {
                        modify = true;
                        Boundary[jj + i4] = node[nJ].bknown[i4];
                    }
                }

                // Material part

                Tangent[0][jj + 0] += T[0][0][0] * shpJ[0] + T[0][0][1] * shpJ[1] + T[0][0][2] * shpJ[2];
                Tangent[0][jj + 1] += T[0][1][0] * shpJ[0] + T[0][1][1] * shpJ[1] + T[0][1][2] * shpJ[2];
                Tangent[0][jj + 2] += T[0][2][0] * shpJ[0] + T[0][2][1] * shpJ[1] + T[0][2][2] * shpJ[2];

                Tangent[1][jj + 0] += T[1][0][0] * shpJ[0] + T[1][0][1] * shpJ[1] + T[1][0][2] * shpJ[2];
                Tangent[1][jj + 1] += T[1][1][0] * shpJ[0] + T[1][1][1] * shpJ[1] + T[1][1][2] * shpJ[2];
                Tangent[1][jj + 2] += T[1][2][0] * shpJ[0] + T[1][2][1] * shpJ[1] + T[1][2][2] * shpJ[2];

                Tangent[2][jj + 0] += T[2][0][0] * shpJ[0] + T[2][0][1] * shpJ[1] + T[2][0][2] * shpJ[2];
                Tangent[2][jj + 1] += T[2][1][0] * shpJ[0] + T[2][1][1] * shpJ[1] + T[2][1][2] * shpJ[2];
                Tangent[2][jj + 2] += T[2][2][0] * shpJ[0] + T[2][2][1] * shpJ[1] + T[2][2][2] * shpJ[2];
            }

            // Assemble force vector

            Force[0] += -R[0];
            Force[1] += -R[1];
            Force[2] += -R[2];
        }
    }

    // Modify residual

    if (modify)
    {
        for (unsigned int i1 = 0; i1 < Boundary.size(); i1++)
        {
            for (unsigned int i2 = 0; i2 < ndf; i2++)
            {
                Force[i2] -= Tangent[i2][i1] * Boundary[i1];
            }
        }
    }

    // Assembly

    solution.NodalAssembly(nI, ndf, Force, Tangent);

    // Store force

    Forc[0] = Force[0];
    Forc[1] = Force[1];
    Forc[2] = Force[2];
}
void Node::Update(unsigned int n, double dt, int type, int isw)
{

    // Check if solution was correct

    if (btype[0] > 0)
    {
        if (isnan(solution.x[btype[0] - 1]) == true)
        {
            if (Time > 0)
            {
                cout << "No solution of primary variable of node  " << n << "  " << icoor[0] << "  at time  " << Time << endl;
                exit(0);
            }
        }
    }
    if (btype[1] > 0)
    {
        if (isnan(solution.x[btype[1] - 1]) == true)
        {
            if (Time > 0)
            {
                cout << "No solution of primary variable of node  " << n << "  " << icoor[1] << "  at time  " << Time << endl;
                exit(0);
            }
        }
    }
    if (btype[2] > 0)
    {
        if (isnan(solution.x[btype[2] - 1]) == true)
        {
            if (Time > 0)
            {
                cout << "No solution of primary variable of node  " << n << "  " << icoor[2] << "  at time  " << Time << endl;
                exit(0);
            }
        }
    }

    // Parameters of time integration scheme

    vector<double> tparam = solution.get_tparam();

    // Central difference time integration

    if (type == 0)
    {

        // Old displacement increment

        double du_old[3];
        du_old[0] = du[0];
        du_old[1] = du[1];
        du_old[2] = du[2];

        // Current increment of displacements (N.B. in case of lumped mass add displacement increment of previous time step)

        if (btype[0] > 0)
        {
            du[0] = solution.x[btype[0] - 1];
            if (isw == 6)
                du[0] += du_old[0];
        }
        else
        {
            du[0] = bknown[0];
        }
        if (btype[1] > 0)
        {
            du[1] = solution.x[btype[1] - 1];
            if (isw == 6)
                du[1] += du_old[1];
        }
        else
        {
            du[1] = bknown[1];
        }
        if (btype[2] > 0)
        {
            du[1] = solution.x[btype[2] - 1];
            if (isw == 6)
                du[2] += du_old[2];
        }
        else
        {
            du[2] = bknown[2];
        }

        // Actual acceleration a_n

        double c1 = 1.0 / dt;
        double c2 = c1 * c1;

        acce[0] = c2 * (du[0] - du_old[0]);
        acce[1] = c2 * (du[1] - du_old[1]);
        acce[2] = c2 * (du[2] - du_old[2]);

        // Displacements, velocity, position vector next time step u_n+1

        disp[0] += du[0];
        disp[1] += du[1];
        disp[2] += du[2];
        velo[0] += dt * acce[0];
        velo[1] += dt * acce[1];
        velo[2] += dt * acce[2];
        acoor[0] += du[0];
        acoor[1] += du[1];
        acoor[2] += du[2];
    }

    // Newmark time integrations

    else if (type >= 1)
    {

        // Old increment of displacments

        double du_inc[3] = {0, 0, 0};

        // Current increment of displacements

        if (btype[0] > 0)
        {
            du_inc[0] = solution.x[btype[0] - 1];
        }
        else
        {
            du_inc[0] = bknown[0];
            bknown[0] = 0.0;
        }
        if (btype[1] > 0)
        {
            du_inc[1] = solution.x[btype[1] - 1];
        }
        else
        {
            du_inc[1] = bknown[1];
            bknown[1] = 0.0;
        }
        if (btype[2] > 0)
        {
            du_inc[2] = solution.x[btype[2] - 1];
        }
        else
        {
            du_inc[2] = bknown[2];
            bknown[2] = 0.0;
        }

        // Current displacements and discplacement increments

        disp[0] += du_inc[0];
        disp[1] += du_inc[1];
        disp[2] += du_inc[2];
        du[0] += du_inc[0];
        du[1] += du_inc[1];
        du[2] += du_inc[2];

        // Current velocity and acceleration

        if (type == 1)
        {
            velo[0] += solution.tparam[3] * du_inc[0];
            velo[1] += solution.tparam[3] * du_inc[1];
            velo[2] += solution.tparam[3] * du_inc[2];
        }
        if (type == 1)
        {
            acce[0] += solution.tparam[0] * du_inc[0];
            acce[1] += solution.tparam[0] * du_inc[1];
            acce[2] += solution.tparam[0] * du_inc[2];
        }

        if (type == 2)
        {
            double c1 = 1.0 / dt;
            velo[0] += c1 * du_inc[0];
            velo[1] += c1 * du_inc[1];
            velo[2] += c1 * du_inc[2];
        }

        // Current position vector

        acoor[0] += du_inc[0];
        acoor[1] += du_inc[1];
        acoor[2] += du_inc[2];

        // Initial nodal boundary force

        for (unsigned int i = 0; i < ndf; i++)
        {
            if (bforce[i] != 0)
                istf = 1;
        }

        // Old increment of Lagrange multiplier

        double lagr_inc[3] = {0, 0, 0};

        // Current increment of Lagrange multiplier

        if (ltype[0] > 0)
        {
            lagr_inc[0] = solution.x[ltype[0] - 1];
        }
        else
        {
            lagr_inc[0] = 0.0;
        }
        if (ltype[1] > 0)
        {
            lagr_inc[1] = solution.x[ltype[1] - 1];
        }
        else
        {
            lagr_inc[1] = 0.0;
        }
        if (ltype[2] > 0)
        {
            lagr_inc[2] = solution.x[ltype[2] - 1];
        }
        else
        {
            lagr_inc[2] = 0.0;
        }

        // Current Lagrange multiplier

        lagr[0] += lagr_inc[0];
        lagr[1] += lagr_inc[1];
        lagr[2] += lagr_inc[2];
    }
}
int Node::AssignInTangent(unsigned int nJ)
{
    int PlaceInTangent = -1;
    for (unsigned int i1 = 0; i1 < ListNodes.size(); i1++)
    {
        if (ListNodes[i1] == nJ)
        {
            PlaceInTangent = i1 * ndf;
            i1 = ListNodes.size();
        }
    }
    if (PlaceInTangent == -1)
    {
        cout << "Node:: AsignInTangent: Wrong place in Tangent  " << endl;
        exit(0);
    }

    return PlaceInTangent;
}
int Node::AssignNodeInList(unsigned int nI, unsigned e)
{
    int PlaceInList = -1;
    for (unsigned int i = 0; i < element[e].connectivity.size(); i++)
    {
        unsigned int nP = element[e].connectivity[i];
        if (nI == nP)
        {
            PlaceInList = i;
            i = element[e].connectivity.size();
        }
    }
    if (PlaceInList == -1)
    {
        cout << "Node:: AssignNodeInList: Wrong place in ListNodes  " << endl;
        exit(0);
    }

    return PlaceInList;
}
void Node::NodalDomain(unsigned int n)
{

    // Initialization

    vector<unsigned int> ConnectedNodes(numnp, 0);
    ListNodes.clear();

    // Loop over neigboring elements of the corresponding node

    for (unsigned int i1 = 0; i1 < NeighborElements.size(); i1++)
    {

        // Neigboring element

        int e = NeighborElements[i1];

        // List of nodes in the neigboring element

        for (unsigned int i2 = 0; i2 < element[e].connectivity.size(); i2++)
        {

            // Node connected to current node

            int nn = element[e].connectivity[i2];
            if (ConnectedNodes[nn] == 0)
            {
                ListNodes.push_back(nn);
                ConnectedNodes[nn] = 1;
            }
        }
    }
}
void Node::InitialNormalVectorAndArea(unsigned int nI)
{

    // Definition of nodes in influence domain

    normal[0] = 0;
    normal[1] = 0;
    normal[2] = 0;
    for (unsigned int i1 = 0; i1 < NeighborElements.size(); i1++)
    {

        // Neigboring element

        unsigned int e = NeighborElements[i1];

        // Corresponing node number in element

        unsigned int nP = AssignNodeInList(nI, e);

        // Compute normal vector

        for (unsigned int l = 0; l < element[e].lint; l++)
        {
            normal[0] += element[e].shp[0][nP][l] * element[e].vol[l];
            normal[1] += element[e].shp[1][nP][l] * element[e].vol[l];
            normal[2] += element[e].shp[2][nP][l] * element[e].vol[l];
        }
    }

    // Nodal surface area

    area = sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
}
void Node::CurrentNormalVector(unsigned int nI)
{

    // Loop over influence domain

    for (unsigned int i1 = 0; i1 < NeighborElements.size(); i1++)
    {

        // Global material point

        unsigned int e = NeighborElements[i1];

        for (unsigned int l = 0; l < element[e].lint; l++)
        {

            unsigned int nP = AssignNodeInList(nI, e);

            // Normal vector

            normal[0] += element[e].shp[0][nP][l] * element[e].vol[l];
            normal[1] += element[e].shp[1][nP][l] * element[e].vol[l];
            normal[2] += element[e].shp[2][nP][l] * element[e].vol[l];
        }
    }
}
void Node::MaxArea(double &maxArea)
{
    // Update maximum nodal area

    if (area > maxArea)
    {
        maxArea = area;
    }
}
void Node::NodeOnSurface(double maxArea, double tol)
{
    // Relative nodal area

    double rel = area / maxArea;

    // Update maximum nodal area

    onSurface = 0;
    if (rel > tol)
    {
        onSurface = 1;
    }
}
void Node::DirichletBoundaryHomogenization()
{

    if (onSurface == 1)
    {
        bdisp[0] = icoor[0] * solution.HomBoundary[0] + 0.5 * icoor[1] * solution.HomBoundary[3] + 0.5 * icoor[2] * solution.HomBoundary[5];
        bdisp[1] = 0.5 * icoor[0] * solution.HomBoundary[3] + icoor[1] * solution.HomBoundary[1] + 0.5 * icoor[2] * solution.HomBoundary[4];
        bdisp[2] = 0.5 * icoor[0] * solution.HomBoundary[5] + 0.5 * icoor[1] * solution.HomBoundary[4] + icoor[2] * solution.HomBoundary[2];

        if (solution.HomMethod != "Lagrange")
        {
            btype[0] = solution.HomFixBoundary[0];
            btype[1] = solution.HomFixBoundary[1];
            btype[2] = solution.HomFixBoundary[2];
        }

        bprop[0] = solution.HomFixBoundary[0];
        bprop[1] = solution.HomFixBoundary[1];
        bprop[2] = solution.HomFixBoundary[2];
    }
}
void Node::LagrangeDomain(unsigned int nI)
{

    if (solution.HomType == "DisplacementStrain")
    {
        if (onSurface)
        {

            // Store connected nodes to Lagrange multiplier

            Nodes2Lagrange.push_back(nI);

            // Store active or unactive nodes

            ltype[0] = solution.HomFixBoundary[0];
            ltype[1] = solution.HomFixBoundary[1];
            ltype[2] = solution.HomFixBoundary[2];

            // Initialize values

            lagr[0] = 0.0;
            lagr[1] = 0.0;
            lagr[2] = 0.0;
        }
    }
    else if (solution.HomType == "DisplacementPeriodic")
    {

        // Node on positive plane z-direction

        for (unsigned int i1 = 0; i1 < node.size(); i1++)
        {
            if (normal[2] > 1e-6 && node[i1].normal[2] < 1e-6)
            {
                double x1 = fabs(icoor[0] - node[i1].icoor[0]);
                double y1 = fabs(icoor[1] - node[i1].icoor[1]);

                double nx1 = fabs(normal[0] - node[i1].normal[0]);
                double ny1 = fabs(normal[1] - node[i1].normal[1]);

                if (x1 < 1e-6 && y1 < 1e-6 && nx1 < 1e-6 && ny1 < 1e-6)
                {

                    // Node pair

                    Nodes2Lagrange.push_back(nI);
                    Nodes2Lagrange.push_back(i1);

                    // Store active or unactive nodes

                    ltype[0] = solution.HomFixBoundary[0];
                    ltype[1] = solution.HomFixBoundary[1];
                    ltype[2] = solution.HomFixBoundary[2];

                    // Initialize values

                    lagr[0] = 0.0;
                    lagr[1] = 0.0;
                    lagr[2] = 0.0;

                    i1 = node.size();
                }
            }

            // Node on positive plane y-direction

            if (normal[1] > 1e-6 && node[i1].normal[1] < 1e-6)
            {
                double x1 = fabs(icoor[0] - node[i1].icoor[0]);
                double z1 = fabs(icoor[2] - node[i1].icoor[2]);

                double nx1 = fabs(normal[0] - node[i1].normal[0]);
                double nz1 = fabs(normal[2] - node[i1].normal[2]);

                if (x1 < 1e-6 && z1 < 1e-6 && nx1 < 1e-6 && nz1 < 1e-6)
                {
                    // Node pair

                    Nodes2Lagrange.push_back(nI);
                    Nodes2Lagrange.push_back(i1);

                    // Store active or unactive nodes

                    ltype[0] = solution.HomFixBoundary[0];
                    ltype[1] = solution.HomFixBoundary[1];
                    ltype[2] = solution.HomFixBoundary[2];

                    // Initialize values

                    lagr[0] = 0.0;
                    lagr[1] = 0.0;
                    lagr[2] = 0.0;

                    i1 = node.size();
                }
            }

            // Node on positive plane x-direction

            if (normal[0] > 1e-6 && node[i1].normal[0] < 1e-6)
            {
                double y1 = fabs(icoor[1] - node[i1].icoor[1]);
                double z1 = fabs(icoor[2] - node[i1].icoor[2]);

                double ny1 = fabs(normal[1] - node[i1].normal[1]);
                double nz1 = fabs(normal[2] - node[i1].normal[2]);

                if (y1 < 1e-6 && z1 < 1e-6 && ny1 < 1e-6 && nz1 < 1e-6)
                {

                    // Node pair

                    Nodes2Lagrange.push_back(nI);
                    Nodes2Lagrange.push_back(i1);

                    // Store active or unactive nodes

                    ltype[0] = solution.HomFixBoundary[0];
                    ltype[1] = solution.HomFixBoundary[1];
                    ltype[2] = solution.HomFixBoundary[2];

                    // Initialize values

                    lagr[0] = 0.0;
                    lagr[1] = 0.0;
                    lagr[2] = 0.0;

                    i1 = node.size();
                }
            }
        }
    }
}
void Node::ListLagrange2Node(unsigned int nI)
{
    for (unsigned int i1 = 0; i1 < node.size(); i1++)
    {
        for (unsigned int i2 = 0; i2 < node[i1].Nodes2Lagrange.size(); i2++)
        {
            if (node[i1].Nodes2Lagrange[i2] == nI)
            {
                Lagrange2Node.push_back(i1);
            }
        }
    }
}
void Node::TangentAndResidualLagrange(unsigned int nI)
{
    // Initialize array

    unsigned int nn = Lagrange2Node.size();

    if (nn != 0)
    {
        vector<double> ForceU(ndf, 0);
        vector<vector<double>> TangentUL(ndf, vector<double>(ndf * nn, 0));

        // Loop over Lagrange multiplier linked to node

        for (unsigned int i1 = 0; i1 < Lagrange2Node.size(); i1++)
        {

            // Lagrange multiplier number

            unsigned int lm = Lagrange2Node[i1];

            // Determine factor

            double factor = 1.0;
            for (unsigned int i2 = 0; i2 < node[lm].Nodes2Lagrange.size(); i2++)
            {
                if (node[lm].Nodes2Lagrange[i2] == nI && i2 != 0)
                {
                    factor = -1.0;
                }
            }

            // Tangent - Displacement contribution

            for (unsigned int i2 = 0; i2 < ndf; i2++)
            {
                TangentUL[i2][ndf * i1 + i2] += factor;
            }

            // Residual vector - Displacement contribution

            for (unsigned int i2 = 0; i2 < ndf; i2++)
            {
                ForceU[i2] -= factor * node[lm].lagr[i2];
            }
        }

        solution.NodalAssemblyLagrange(0, nI, ndf, ForceU, TangentUL);
    }

    // Contributions to Lagrange multiplier test function

    nn = Nodes2Lagrange.size();

    if (nn != 0)
    {
        // Initialize arrays

        vector<double> ForceL(ndf, 0);
        vector<vector<double>> TangentLU(ndf, vector<double>(nn * ndf, 0));

        for (unsigned int i1 = 0; i1 < Nodes2Lagrange.size(); i1++)
        {
            // Displacement node number

            int nJ = Nodes2Lagrange[i1];
            double factor = 1.0;
            if (i1 == 1)
            {
                factor = -1.0;
            }

            // Tangent

            for (unsigned int i2 = 0; i2 < ndf; i2++)
            {
                TangentLU[i2][ndf * i1 + i2] += factor;
            }

            // Residuum vector lagrange mulitplier

            double prop = 0;
            for (unsigned int i2 = 0; i2 < ndf; i2++)
            {
                prop = solution.prop(solution.ptype, nJ, i2, Time);
                ForceL[i2] -= factor * (node[nJ].disp[i2] - prop * node[nJ].bdisp[i2]);
            }
        }
        // Assemble

        solution.NodalAssemblyLagrange(1, nI, ndf, ForceL, TangentLU);
    }
}
