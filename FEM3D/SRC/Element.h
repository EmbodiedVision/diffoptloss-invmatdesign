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
#ifndef Element_h
#define Element_h

// Includes
#include <vector>

using namespace std;

class Element
{

public:
  Element() : mtype(0) {}

  unsigned int nen, mtype, lint;
  int itype;
  vector<unsigned int> connectivity;
  vector<double> detF;
  vector<double> detdF;
  vector<vector<double>> Sigma;
  vector<double> vol;
  vector<vector<vector<double>>> tenF;
  vector<vector<vector<double>>> tendF;
  vector<vector<vector<double>>> shp;

  void InitArrays();
  void NodalNeighborElements(unsigned int e);
  void Kinematics();
  vector<double> push_vector(vector<double> shp_ini, double fi[][3]);
  void inverse3(double f[][3], double fi[][3]);
  void update_hist();
  void UpdateShapeFunction(unsigned int ngp, unsigned int nen);
  void UpdateVolume(unsigned int ngp);
  void NeighborElementsToNodes(unsigned int e);
  void Average(int p, double AverageStrain[6], double AverageStress[6], double &Volume);
  void TetShapeFunctions(unsigned int l, unsigned int nel, vector<vector<double>> xgl, vector<vector<double>> xn);
  void HexShapeFunctions(unsigned int l, unsigned int nel, vector<vector<double>> xgl, vector<vector<double>> xn);

private:
};

#endif
