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

// Includes
#ifndef Node_h
#define Node_h

#include <vector>
#include <string>

using namespace std;

class Node
{

public:
  Node() {}

  void initialize(int n, int t, int type, int isw);
  void initial_boundary(string type, int n, double dt);
  void initial(double dt, vector<double> x);
  void InitializeCRSFormat(unsigned int neq);
  void project(int n);
  void ContactRigid(unsigned int nI);
  void ForceVector(int nI, unsigned int typeLoad, vector<vector<double>> &Tangent);
  void initialForce(int n, double areaX, double areaY);
  void BoundaryForce(unsigned int nI);
  void ContactForce(unsigned int nI);
  void SolidStaticAceGen(unsigned int nI);
  void Update(unsigned int n, double dt, int type, int isw);
  int AssignInTangent(unsigned int nJ);
  int AssignNodeInList(unsigned int nI, unsigned e);
  void NodalDomain(unsigned int n);
  void InitialNormalVectorAndArea(unsigned int nI);
  void CurrentNormalVector(unsigned int nI);
  void MaxArea(double &maxArea);
  void NodeOnSurface(double maxArea, double tol);
  void DirichletBoundaryHomogenization();
  void LagrangeDomain(unsigned int nI);
  void ListLagrange2Node(unsigned int nI);
  void TangentAndResidualLagrange(unsigned int nI);

  int istf;
  unsigned int onSurface;
  double mass, stress[10], normal[3], Forc[3], vI, area;
  double icoor[3], acoor[3], pcoor[3], velo[3], acce[3], disp[3], du[3], force[3], pdisp[3], lagr[3];
  vector<unsigned int> NeighborElements;
  vector<unsigned int> ListNodes;
  vector<unsigned int> Nodes2Lagrange;
  vector<unsigned int> Lagrange2Node;
  vector<int> btype;
  vector<int> ltype;
  vector<double> bdisp;
  vector<double> bknown;
  vector<double> bforce;
  vector<int> bprop;
  vector<int> ptype;
  vector<double> penaltyN;
  vector<double> penaltyT;

private:
};

#endif
