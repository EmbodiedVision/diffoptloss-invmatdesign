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


from collections import defaultdict
import json
import pickle
import sys, os
import tempfile
import subprocess
from queue import Queue, Empty
import threading

import numpy as np
import torch
from scipy.optimize import minimize, check_grad

from dataset_utils import pad, unpad, flatten_nodes, structure_nodes
from fem3d import read_in_file_3d, write_in_file_3d
from material_normalization import MaterialListData


class SolverInterface:
    def __init__(self, config, ds, n_el_per_dim):
        ### adjust config: set inverse target and set materials to full!
        self.matlist_data: MaterialListData = ds['matlist_data']
        self.padding = ds['padding']
        self.config = config
        self.n_el_per_dim = n_el_per_dim
        c = read_in_file_3d(config['base_config'])
        n_el = c.n_elements
        n_mat = n_el # full
        mat_ass = np.arange(n_el)
        c.n_materials = n_mat
        c.element_material = mat_ass
        c.material_data = np.ones((n_mat, 3)) * (-1) # Illegal value, but will anyhow set later!
        assert config['inverse_target'] is not None, f"No inverse target provided!"
        c.inverse_target = config['inverse_target']
        self.conf_file = tempfile.NamedTemporaryFile('w', prefix='/tmp/inverse_conf_', suffix=f'.ini', delete=False)
        write_in_file_3d(c, self.conf_file) # closes
        self.fem_config = c
        self.get_element_coordinates()

        # build fifos
        self.fifo_path = tempfile.mkstemp(prefix='diff_fem_pipe_', dir='/tmp')[1]
        os.unlink(self.fifo_path)
        os.mkfifo(self.fifo_path + '_tofem')
        os.mkfifo(self.fifo_path + '_topy')
        self.proc = subprocess.Popen([config['fem_path'], self.conf_file.name, f"--interactive_fifo={self.fifo_path}", "--calc_grad", f"--solver={config['solver_type']}",
                                      f"--abs_tol={config['solver_abs_tol']}", f"--rel_tol={config['solver_rel_tol']}"])
        self.default_abs_tol = config['solver_abs_tol']
        self.default_rel_tol = config['solver_rel_tol']
        self.precise_abs_tol = 0.
        self.precise_rel_tol = 1e-8
        self.fifo_tofem = open(self.fifo_path + '_tofem', 'wb')
        self.fifo_topy = open(self.fifo_path + '_topy', 'rb')
        self.fifo_tofem.write("hi.".encode())
        self.fifo_tofem.flush()
        check = self.fifo_topy.read(3).decode()
        assert check == 'hi.', f"expected hi., but got {check}, {check[0]}, {check[1]}, {check[2]}."
        self.last_control = None, None, None

    def cleanup(self):
        size = np.array(0, dtype=np.uint32)
        bts = size.tobytes()
        assert len(bts) == 4
        self.fifo_tofem.write(bts)
        self.fifo_tofem.flush()
        os.unlink(self.fifo_path + '_tofem')
        os.unlink(self.fifo_path + '_topy')
        os.unlink(self.conf_file.name)
        os.unlink(self.conf_file.name[:-len('.ini')] + '.out')
        # print('Cleaned up solver interface.')

    def prepare_send(self, x: np.ndarray): # batch dim
        x_for_later = None
        x = unpad(x, self.padding)
        x = flatten_nodes(x) # (bs,d,N)
        x = x.transpose((0, 2, 1)) # (bs, N, d)

        x = x.clip(-1, 1)
        x = self.matlist_data.unnormalize(x)
        x = x.transpose((0, 2, 1)) # (bs, d, N)
        return x, x_for_later

    def send_mat(self, x: np.ndarray, calc_grad=True, keep_sol=True, precise=False): # no batch dim: (d,N)
        assert len(x.shape) == 2 # no batch dim and flat
        x = x.T # (N, d)
        x = x.flatten() # (N*d)
        size = np.array(x.nbytes, dtype=np.uint32)
        bts = size.tobytes()
        assert len(bts) == 4
        self.fifo_tofem.write(bts)
        self.fifo_tofem.flush()

        if precise:
            abs_tol, rel_tol = self.precise_abs_tol, self.precise_rel_tol
        else:
            abs_tol, rel_tol = self.default_abs_tol, self.default_rel_tol

        control = np.array([calc_grad, keep_sol, keep_sol], dtype=np.uint8)
        self.last_control = calc_grad, keep_sol, keep_sol
        bts = control.tobytes()
        assert len(bts) == 3
        self.fifo_tofem.write(bts)
        self.fifo_tofem.flush()

        tolerances = np.array([abs_tol, rel_tol], dtype=np.float32)
        bts = tolerances.tobytes()
        assert len(bts) == 8
        self.fifo_tofem.write(bts)
        self.fifo_tofem.flush()

        bts = x.tobytes()
        self.fifo_tofem.write(bts)
        self.fifo_tofem.flush()


    def process_grad(self, grad: np.ndarray): # batched: (bs,d,N)
        # print('raw grad', grad)
        grad = structure_nodes(grad, self.n_el_per_dim) # (bs,d,w,h,dp)
        grad, pads = pad(grad)
        unnorm_scale = self.matlist_data.unnorm_scale(grad) # the "unnormalize" direction. makes stuff bigger.
        grad = grad * unnorm_scale[None, :, None, None, None] # 
        return grad

    def recv_grad(self):
        bts = self.fifo_topy.read(4 * 2)
        it_counts = np.frombuffer(bts, dtype=np.int32, count=2)
        # print('recv it counts', it_counts)
        it_count = it_counts.sum()
        bts = self.fifo_topy.read(8) # first, recv aveK as double
        K = np.frombuffer(bts, dtype=np.double, count=1).item()
        bts = self.fifo_topy.read(8) # first, recv J as double
        J = np.frombuffer(bts, dtype=np.double, count=1).item()
        bts = self.fifo_topy.read(4)
        size = np.frombuffer(bts, dtype=np.uint32, count=1).item()
        if size == 0:
            assert not self.last_control[0] # no calc grad
        # print('py: recv', size, 'bytes of grad')
        bts = self.fifo_topy.read(size)
        grad = np.frombuffer(bts, dtype=np.double)
        # print('py: recved array', grad.shape, grad.dtype)
        grad = grad.reshape(-1, 2) # (N, 2) # TODO ρ
        # print('some grad:', grad[0], grad[1], grad[10], grad[-1])
        grad = grad.T # (2,N)
        return grad, K, J, it_count # (N,d)

    def linesearch(self, stepsize):
        bts = np.array(stepsize, dtype=np.double).tobytes()
        assert len(bts) == 8
        self.fifo_tofem.write(bts)
        self.fifo_tofem.flush()
        if stepsize < 0:
            return
        bts = self.fifo_topy.read(8) # aveK
        obj_val = np.frombuffer(bts, dtype=np.double, count=1).item()
        return obj_val

    def get_element_coordinates(self):
        n_nodes_per_dim = {}
        for d in range(3):
            d_values = np.unique(self.fem_config.coordinates[:, d])
            n_nodes_per_dim[d] = len(d_values)
        _n_nodes = np.prod(list(n_nodes_per_dim.values()))
        assert _n_nodes == self.fem_config.n_nodes, f"Config says {self.fem_config.n_nodes}, but prod of per dim yields {_n_nodes}"
        n_el_per_dim = {d : n_nodes_per_dim[d] - 1 for d in range(3)}
        assert np.prod(list(n_el_per_dim.values())) == self.fem_config.n_elements

        # assert node order
        coords_structured = structure_nodes(self.fem_config.coordinates.T, n_nodes_per_dim) # (d=3, n_nodes) --> (d=3, w, h, depth)
        assert (coords_structured[:, :-1, :, :] <= coords_structured[:, 1:, :, :]).all()
        assert (coords_structured[:, :, :-1, :] <= coords_structured[:, :, 1:, :]).all()
        assert (coords_structured[:, :, :, :-1] <= coords_structured[:, :, :, 1:]).all()
        assert (flatten_nodes(coords_structured) == self.fem_config.coordinates.T).all()

        # element coordinates
        element_coordinates = self.fem_config.coordinates[self.fem_config.elements, :]
        assert element_coordinates.ndim == 3
        element_coordinates = element_coordinates.mean(axis=1) # mean over nodes per el --> (n_el, 3)
        # el_z_coordinates = np.unique(element_coordinates[:, 2])
        # assert len(el_z_coordinates) == n_el_per_dim[2]

        # assert element order (just use node functions)
        el_coords_struct = structure_nodes(element_coordinates.T, n_el_per_dim)
        assert (el_coords_struct[:, :-1, :, :] <= el_coords_struct[:, 1:, :, :]).all()
        assert (el_coords_struct[:, :, :-1, :] <= el_coords_struct[:, :, 1:, :]).all()
        assert (el_coords_struct[:, :, :, :-1] <= el_coords_struct[:, :, :, 1:]).all()
        self.n_el_per_dim = tuple(n_el_per_dim[k] for k in [0, 1, 2])
        
        # with np.printoptions(linewidth=500, threshold=sys.maxsize):
        #     print('mat ass orig', self.fem_config.element_material)
        self.mesh_scale = coords_structured[1, 0, -1, 0]
        assert coords_structured[2, 0, 0, -1] == self.mesh_scale # square

        if self.n_el_per_dim[0] == 1: ## -> 2d
            self.elem_coords_2d = el_coords_struct[1:, 0, :, :] # the 2D part
            self.element_coordinates_normalized = self.elem_coords_2d / self.mesh_scale
        else: ## -> 3d
            elem_coords_3d = el_coords_struct[:, :, :, :]
            self.element_coordinates_normalized = elem_coords_3d / self.mesh_scale




def communicate(x_prepared, aligned, calc_grad=True, keep_sol=True, precise=False):
    bs, d, n = x_prepared.shape
    assert d == 3
    Js = np.ones(bs) * float('nan')
    Ks = np.ones(bs) * float('nan')
    grads_raw = np.ones((bs, 2, n)) * float('nan')
    it_counts = np.ones(bs)  * float('nan')

    if not aligned:
        idx_queue = Queue()
        for idx in range(bs):
            idx_queue.put(idx)
    else:
        assert bs == len(INTERFACES), f"if doing aligned, number of items must match number of interfaces! ({bs=}, {len(INTERFACES)=})"
    exception = None

    def worker(interface: SolverInterface, idx=None):
        nonlocal exception
        if not aligned:
            assert idx is None
        try:
            while True:
                if not aligned:
                    idx = idx_queue.get(False)
                interface.send_mat(x_prepared[idx], calc_grad=calc_grad, keep_sol=keep_sol, precise=precise)
                grad, K, J, it_count = interface.recv_grad()
                interface.linesearch(-1) # nope.
                if grad.size > 0:
                    grads_raw[idx, :, :] = grad
                Js[idx] = J
                Ks[idx] = K
                it_counts[idx] = it_count
                if aligned:
                    return
        except Empty:
            pass
        except Exception as e:
            exception = e
            raise

    threads = [threading.Thread(target=worker, args=(interface, (i_int if aligned else None))) for i_int, interface in enumerate(INTERFACES)]

    for thread in threads: thread.start()
    for thread in threads: thread.join()
    if exception is not None:
        raise exception

    return grads_raw, Ks, Js, it_counts



INTERFACES: list[SolverInterface] = None