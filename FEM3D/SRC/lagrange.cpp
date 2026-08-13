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
#include "Solution.h"
#include "Lagrange.h"
#include <vector>
#include <string>
#include <fstream>
#include <stdlib.h>

using namespace std;

extern vector<Node> node;
extern vector<Lagrange> lagrange;
extern Solution solution;
extern unsigned int ndf;
extern double dt, Time;

extern string filename;

void Lagrange::TangentAndResidual(unsigned int nI)
{
}
void Lagrange::Update(unsigned int nI)
{
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
