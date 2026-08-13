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
#include "Material.h"
#include "Solution.h"
#include <vector>
#include <string>
#include <fstream>
#include <stdlib.h>

using namespace std;

extern vector<Material> material;
extern vector<Element> element;
extern Solution solution;
extern double dt;

extern string filename;

void Material::stress(unsigned int m, unsigned int e, unsigned int lint, int istrt)
{

    if (mate == "NeoHooke")
    {
        for (unsigned int l = 0; l < lint; l++)
        {

            double J = element[e].detF[l];

            double lambda_J = lam / J;
            double mu_J = mu / J;

            // Left Cauchy Green vector

            double b[6];

            b[0] = element[e].tenF[0][0][l] * element[e].tenF[0][0][l] + element[e].tenF[0][1][l] * element[e].tenF[0][1][l] + element[e].tenF[0][2][l] * element[e].tenF[0][2][l];
            b[1] = element[e].tenF[1][0][l] * element[e].tenF[1][0][l] + element[e].tenF[1][1][l] * element[e].tenF[1][1][l] + element[e].tenF[1][2][l] * element[e].tenF[1][2][l];
            b[2] = element[e].tenF[2][0][l] * element[e].tenF[2][0][l] + element[e].tenF[2][1][l] * element[e].tenF[2][1][l] + element[e].tenF[2][2][l] * element[e].tenF[2][2][l];
            b[3] = element[e].tenF[0][0][l] * element[e].tenF[1][0][l] + element[e].tenF[0][1][l] * element[e].tenF[1][1][l] + element[e].tenF[0][2][l] * element[e].tenF[1][2][l];
            b[4] = element[e].tenF[1][0][l] * element[e].tenF[2][0][l] + element[e].tenF[1][1][l] * element[e].tenF[2][1][l] + element[e].tenF[1][2][l] * element[e].tenF[2][2][l];
            b[5] = element[e].tenF[0][0][l] * element[e].tenF[2][0][l] + element[e].tenF[0][1][l] * element[e].tenF[2][1][l] + element[e].tenF[0][2][l] * element[e].tenF[2][2][l];

            // Stress vector (Cauchy)

            element[e].Sigma[0][l] = 0.5 * lambda_J * (J * J - 1.0) + mu_J * (b[0] - 1.0);
            element[e].Sigma[1][l] = 0.5 * lambda_J * (J * J - 1.0) + mu_J * (b[1] - 1.0);
            element[e].Sigma[2][l] = 0.5 * lambda_J * (J * J - 1.0) + mu_J * (b[2] - 1.0);
            element[e].Sigma[3][l] = mu_J * b[3];
            element[e].Sigma[4][l] = mu_J * b[4];
            element[e].Sigma[5][l] = mu_J * b[5];
        }
    }
    else if (mate == "LinearElasticity")
    {

        for (unsigned int l = 0; l < lint; l++)
        {
            // Displacement gradient

            double H[3][3];

            H[0][0] = element[e].tenF[0][0][l] - 1.0;
            H[0][1] = element[e].tenF[0][1][l];
            H[0][2] = element[e].tenF[0][2][l];
            H[1][0] = element[e].tenF[1][0][l];
            H[1][1] = element[e].tenF[1][1][l] - 1.0;
            H[1][2] = element[e].tenF[1][2][l];
            H[2][0] = element[e].tenF[2][0][l];
            H[2][1] = element[e].tenF[2][1][l];
            H[2][2] = element[e].tenF[2][2][l] - 1.0;

            // Strain tensor

            double eps[6];
            eps[0] = H[0][0];
            eps[1] = H[1][1];
            eps[2] = H[2][2];
            eps[3] = H[0][1] + H[1][0];
            eps[4] = H[1][2] + H[2][1];
            eps[5] = H[0][2] + H[2][0];

            // Trace of strain

            double treps = eps[0] + eps[1] + eps[2];

            // Stress vector

            element[e].Sigma[0][l] = lam * treps + 2.0 * mu * eps[0];
            element[e].Sigma[1][l] = lam * treps + 2.0 * mu * eps[1];
            element[e].Sigma[2][l] = lam * treps + 2.0 * mu * eps[2];
            element[e].Sigma[3][l] = mu * eps[3];
            element[e].Sigma[4][l] = mu * eps[4];
            element[e].Sigma[5][l] = mu * eps[5];
        }
    }
}
void Material::init(string mat)
{

    if (mat == "NeoHooke" || mat == "LinearElasticity")
    {
        lam = matdata[0] * matdata[1] / ((1 + matdata[1]) * (1 - 2 * matdata[1]));
        mu = matdata[0] / (2 * (1 + matdata[1]));
        rho = matdata[2];

        solution.difftype = "Solid";
    }
    else if (mat == "NeoHookeL")
    {
        lam = matdata[0];
        mu = matdata[1];
        rho = matdata[2];

        mate = "NeoHooke";

        solution.difftype = "Solid";
    }
}
