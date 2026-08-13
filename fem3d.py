# Copyright 2026 University of Augsburg, Intelligent Perception in Technical Systems Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Code Author: Jens Kreber <jens.kreber@uni-a.de>


import sys
import numpy as np

UINT_TYPE = np.uint32
FLOAT_TYPE = np.float64


class Config3D:
    n_nodes: int
    n_elements: int
    n_materials: int
    n_nodes_per_element: int

    material_data: np.ndarray  # float,float (n_materials, 2): E, A
    boundaries: list
    coordinates: np.ndarray  # float n_nodes
    elements: (
        np.ndarray
    )  # int, int, ... (n_elements, n_nodes_per_element): node indices
    element_material: np.ndarray  # int (n_elements): element indices

    inverse_target: float  # optional: target for inverse design gradients
    homogenization: list[str]  # 4 lines of homogenization info, not parsed

    def __repr__(
        self,
    ):  # from https://docs.python.org/3/library/types.html#types.SimpleNamespace
        items = (f"{k}={v!r}" for k, v in self.__dict__.items())
        return "{}({})".format(type(self).__name__, ", ".join(items))


other_stuff_lines = {
    "DimensionNodalDegree": "3 3",
    "StepsPrint": "1",
    "TimeIncrement": "1",
    "TimeSteps": "1",
    "Loadtype": ["LinearProp 1", "0 0 1 1"],
    "TestFunction": "Hex",
    "TrialFunction": "Hex",
    "Formulation": "Standard",
    "Searchtype": ["None", "PointBased Sphere 1.5 1.2"],
    "Stabilization": "None",
    "DiffType": "Lagrange",
    "Theory": "Linear",
    "TimeSolution": ["Static", "Newmark Consistent 0.25 0.5", "Central RowSum"],
    "Gravity": "0 0 0",
}

ok_stuff_lines = ["Projection", "Vtk", "Print", "NoPrint"]


N_NODES_PER_EL = 8
EL_INT_TYPE = 2


def read_in_file_3d(path):
    c = Config3D()
    c.n_nodes_per_element = N_NODES_PER_EL
    c.boundaries = []
    c.inverse_target = None
    c.homogenization = None
    with open(path, "r") as f:
        it = iter(f)
        try:
            while True:
                line = next(it).strip()
                if not line:
                    continue
                if line == "NumberOfNodes":
                    c.n_nodes = int(next(it))
                elif line == "NumberOfElements":
                    c.n_elements = int(next(it))
                elif line == "Materials":
                    c.n_materials = int(next(it))

                elif line in other_stuff_lines:
                    should = other_stuff_lines[line]
                    if not isinstance(should, list):
                        should = [should]
                    for should_ in should:
                        line_ = ""
                        while not line_:
                            line_ = next(it).strip()
                        if line_ == should_:  # matches
                            continue
                        else:
                            print(
                                f"Warning: While reading input file, expected sub-entry of '{line}' to read '{should_}', but it reads '{line_}'. Ignoring."
                            )

                elif line in ok_stuff_lines:
                    continue

                elif line == "Materiallist":
                    c.material_data = (
                        np.ones((c.n_materials, 3), dtype=FLOAT_TYPE) * np.nan
                    )
                    for i in range(c.n_materials):
                        line_ = ""
                        while not line_:
                            line_ = next(it).strip()
                        assert line_ == "LinearElasticity"
                        line_ = next(it).strip()
                        E, Q, d = [float(q) for q in line_.split()]
                        c.material_data[i, :] = [E, Q, d]

                elif line in ["Boundary", "Displacement", "Proportional"]:
                    b = {"what": line}
                    things_read = 0
                    while things_read < 4:
                        line_ = next(it).strip()
                        thing, data = line_.split(" ", maxsplit=1)
                        b[thing] = data
                        things_read += 1
                    c.boundaries.append(b)

                elif line == "InverseDesign":
                    c.inverse_target = float(next(it).strip())

                elif line == "Homogenization":
                    c.homogenization = [next(it).strip() for _ in range(4)]

                elif line == "Node":
                    c.coordinates = np.ones((c.n_nodes, 3), dtype=FLOAT_TYPE) * np.nan
                    for i in range(c.n_nodes):
                        line_ = next(it).strip()
                        c.coordinates[i, :] = [float(q) for q in line_.split()]

                elif line == "NumberNodesElementIntegrationTypeMaterialNumber":
                    c.element_material = np.zeros((c.n_elements,), dtype=UINT_TYPE)
                    for i in range(c.n_elements):
                        line_ = next(it).strip()
                        n_nodes, int_type, mat = [int(q) for q in line_.split()]
                        assert n_nodes == N_NODES_PER_EL
                        assert int_type == EL_INT_TYPE
                        c.element_material[i] = mat - 1  # 0-based indexing

                elif line == "Element":
                    c.elements = np.zeros(
                        (c.n_elements, c.n_nodes_per_element), dtype=UINT_TYPE
                    )
                    for i in range(c.n_elements):
                        line_ = next(it).strip()
                        c.elements[i, :] = [
                            int(q) - 1 for q in line_.split()
                        ]  # 0-based indexing

                else:
                    raise ValueError(f"Cannot parse line '{line}'.")
        except StopIteration:
            return c


def write_in_file_3d(c: Config3D, path):
    if isinstance(path, str):
        f = open(path, "w")
    else:
        f = path
    f.write("NumberOfNodes\n")
    f.write("  " + str(c.n_nodes) + "\n")
    f.write("NumberOfElements\n")
    f.write("  " + str(c.n_elements) + "\n")
    f.write("Materials\n")
    f.write("  " + str(c.n_materials) + "\n")
    for stuff, morestuff in other_stuff_lines.items():
        if not isinstance(morestuff, list):
            morestuff = [morestuff]
        f.write(stuff + "\n")
        for morestuff_ in morestuff:
            f.write("  " + morestuff_ + "\n")
    f.write("\n")
    f.write("Materiallist\n")
    for i in range(c.n_materials):
        f.write("  LinearElasticity\n")
        f.write("  {:e}  {:e}  {:e}\n".format(*c.material_data[i]))
    f.write("\n")

    for b in c.boundaries:
        f.write(b["what"] + "\n")
        for thing, data in b.items():
            if thing == "what":
                continue
            f.write("  " + thing + " " + data + "\n")
        f.write("\n")

    if c.inverse_target is not None:
        f.write("InverseDesign\n")
        f.write("  " + str(c.inverse_target) + "\n")
        f.write("\n")

    if c.homogenization is not None:
        f.write("Homogenization\n")
        for hom_line in c.homogenization:
            f.write("  " + hom_line + "\n")
        f.write("\n")

    f.write("Node\n")
    for i in range(c.n_nodes):
        f.write("  {:e}  {:e}  {:e}\n".format(*c.coordinates[i]))
    f.write("\n")

    f.write("NumberNodesElementIntegrationTypeMaterialNumber\n")
    for i in range(c.n_elements):
        f.write(
            "  {:d}  {:d}  {:d}\n".format(
                N_NODES_PER_EL, EL_INT_TYPE, c.element_material[i] + 1
            )
        )  # 0-based -> 1-based indexing
    f.write("\n")

    f.write("Element\n")
    for i in range(c.n_elements):
        f.write(
            ("  {:4d}" * c.n_nodes_per_element + "\n").format(*c.elements[i] + 1)
        )  # 0-based -> 1-based indexing
    f.write("\n")

    f.write("NoPrint\n")

    f.close()


if __name__ == "__main__":
    c = read_in_file_3d(sys.argv[1])
    write_in_file_3d(c, sys.argv[1] + ".new")
