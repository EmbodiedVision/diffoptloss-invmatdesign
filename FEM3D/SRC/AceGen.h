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
#ifndef AceGen_h
#define AceGen_h

using namespace std;

class AceGen
{

public:
  AceGen() : debug(0) {}

  void inverse4(double a[4][4], double ainv[4][4]);
  void inverse3(double a[3][3], double ainv[3][3]);
  void inverse2(double a[2][2], double ainv[2][2]);
  void NeoHooke3d(double shp[3], double F[3][3], double data[2], double ivol, double R[3], double T[3][3][3]);
  void LinearElasticity3d(double shp[3], double F[3][3], double data[2], double ivol, double R[3], double T[3][3][3]);
  void LinearElasticity3dTheta(double shp[3], double F[3][3], double data[2], double ivol, double T[3][2]);

  int debug;

private:
};

#endif
