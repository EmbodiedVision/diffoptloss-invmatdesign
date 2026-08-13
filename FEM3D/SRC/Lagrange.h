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
#ifndef Lagrange_h
#define Lagrange_h

// Includes

using namespace std;

class Lagrange
{

public:
  Lagrange() {}

  void TangentAndResidual(unsigned int nI);
  void Update(unsigned int nI);

  int ltype[3];
  double lagr[3];
  vector<unsigned int> ConnectedNodes;

private:
};

#endif
