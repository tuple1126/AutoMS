"""
Integrated electrical conductivity analysis script.
Voxelizes OBJ files, writes NPY data, and computes homogenized electrical
conductivity values.
"""

import numpy as np
import open3d as o3d
import torch
import torch.sparse as ts
from linear_operator.utils.linear_cg import linear_cg
import argparse
import os
import sys
import csv
from tqdm import tqdm
import time
from pathlib import Path
from tools.project_paths import CACHE_DIR, RESULTS_DIR, WORKSHOP_DIR

# ============================= Configuration =============================
class Config:
    """Centralized configuration for all adjustable parameters."""

    # Voxelization parameters
    VOXEL_RESOLUTION = 64  # Voxel resolution
    VOXEL_MODE = 'solid'   # Either 'surface' or 'solid'
    ENABLE_VISUALIZATION = False  # Whether to generate visualization files

    # Electrical conductivity calculation parameters
    BASE_CONDUCTIVITY = 1.0        # Base electrical conductivity
    SOLVER_TOL = 1e-5      # Solver tolerance
    SOLVER_MAX_ITER = 5000 # Maximum solver iterations

    # File-path parameters
    INPUT_DIR = str(WORKSHOP_DIR)           # Input OBJ directory
    OUTPUT_DIR = str(RESULTS_DIR)           # Output directory
    TEMP_DIR = str(CACHE_DIR / 'electrical_voxels')

    # GPU settings
    USE_GPU = False                 # Whether to use a GPU
    GPU_DEVICE = 'cuda:0'          # GPU device

# ============================= Voxelization =============================
class VoxelGenerator:
    """Voxelizer for OBJ files."""

    def __init__(self, config):
        self.config = config

    def get_voxel_coo(self, res, max_point, min_point):
        """Return voxel coordinates."""
        index = np.arange(res**3)
        x_ = (index % res)[:, None]
        y_ = ((index // res) % res)[:, None]
        z_ = (index // (res**2))[:, None]
        query_point = np.concatenate((z_, y_, x_), axis=1).astype(np.float32)

        query_point[:, 0] = (query_point[:, 0] + 0.5) * ((max_point[0] - min_point[0]) / res) + min_point[0]
        query_point[:, 1] = (query_point[:, 1] + 0.5) * ((max_point[1] - min_point[1]) / res) + min_point[1]
        query_point[:, 2] = (query_point[:, 2] + 0.5) * ((max_point[2] - min_point[2]) / res) + min_point[2]
        query_point = o3d.core.Tensor(query_point, dtype=o3d.core.Dtype.Float32)
        return query_point

    def voxelize_mesh(self, mesh, res, mode):
        """
        Voxelize a mesh.
        
        Args:
            mesh: Open3D ``TriangleMesh`` object.
            res: Voxel resolution.
            mode: Either ``surface`` or ``solid``.
            
        Returns:
            voxel_array: Voxel array with shape ``(res, res, res)``.
        """
        if mode == 'surface':
            # Surface voxelization: mark only voxels intersecting the mesh.
            voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(
                mesh, voxel_size=(mesh.get_max_bound()[0] - mesh.get_min_bound()[0]) / res)
            all_voxel = voxel_grid.get_voxels()
            voxel_array = np.zeros((res, res, res), dtype=int)
            
            for voxel in all_voxel:
                x, y, z = voxel.grid_index
                if 0 <= x < res and 0 <= y < res and 0 <= z < res:
                    voxel_array[x, y, z] = 1
            
            return voxel_array
        
        else:  # mode == 'solid'
            # Solid voxelization: use ray casting to determine inside/outside.
            query_point = self.get_voxel_coo(res, mesh.get_max_bound(), mesh.get_min_bound())
            mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
            
            scene = o3d.t.geometry.RaycastingScene()
            scene.add_triangles(mesh_t)
            
            occupancy = scene.compute_occupancy(query_point)
            voxel_array = occupancy.numpy().reshape(res, res, res).astype(int)
            
            return voxel_array

    def process_obj_file(self, obj_path, output_dir):
        """Process one OBJ file."""
        try:
            os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists.
            
            mesh = o3d.io.read_triangle_mesh(obj_path)
            if len(mesh.vertices) == 0:
                print(f"Warning: {obj_path} could not be read or is empty")
                return None

            mesh.compute_vertex_normals()
            voxel = self.voxelize_mesh(mesh, self.config.VOXEL_RESOLUTION, self.config.VOXEL_MODE)

            # Save as an NPY file.
            filename = Path(obj_path).stem
            npy_path = os.path.join(output_dir, f"{filename}.npy")
            np.save(npy_path, voxel)

            # Optionally generate a visualization file.
            if self.config.ENABLE_VISUALIZATION:
                vis_dir = os.path.join(output_dir, 'visualization')
                os.makedirs(vis_dir, exist_ok=True)
                vis_path = os.path.join(vis_dir, f"{filename}_vis.obj")
                self._generate_visualization(voxel, vis_path)

            return npy_path

        except Exception as e:
            print(f"Error while processing {obj_path}: {e}")
            return None

    def _generate_visualization(self, voxel, vis_path):
        """Generate a voxel visualization file."""
        res = voxel.shape[0]
        with open(vis_path, 'w') as f:
            for i in range(res):
                for j in range(res):
                    for k in range(res):
                        if voxel[i, j, k]:
                            print('v', i, j, k, file=f)

# ============================= Electrical Conductivity =============================
class NumericalHomogenization:
    """Numerical homogenization for electrical conductivity."""

    def __init__(self, nelx, nely, nelz, lx, ly, lz, device, base_conductivity=1.0):
        self.device = device
        self.__lx = lx
        self.__ly = ly
        self.__lz = lz
        self.__nelx = nelx
        self.__nely = nely
        self.__nelz = nelz
        self.Ke, self.Fe, self.X0 = self.hexahedron(base_conductivity)

    def hexahedron(self, base_conductivity=1):
        """Compute the hexahedral element matrices."""
        conductivity_matrix = torch.eye(3, 3, dtype=torch.float64, device=self.device) * base_conductivity

        dx = self.__lx / self.__nelx / 2
        dy = self.__ly / self.__nely / 2
        dz = self.__lz / self.__nelz / 2

        pp = torch.as_tensor([-pow(3 / 5, 0.5), 0, pow(3 / 5, 0.5)], dtype=torch.float64, device=self.device)
        ww = torch.as_tensor([5. / 9, 8. / 9, 5. / 9], dtype=torch.float64, device=self.device)
        Ke = torch.zeros(8, 8, dtype=torch.float64, device=self.device)
        Fe = torch.zeros(8, 3, dtype=torch.float64, device=self.device)

        dxdydz = torch.as_tensor(
            [[-dx, dx, dx, -dx, -dx, dx, dx, -dx],
             [-dy, -dy, dy, dy, -dy, -dy, dy, dy],
             [-dz, -dz, -dz, -dz, dz, dz, dz, dz]], dtype=torch.float64, device=self.device).t()

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    x = pp[i]
                    y = pp[j]
                    z = pp[k]
                    qxqyqz = torch.as_tensor(
                        [[-((y - 1) * (z - 1)) / 8, ((y - 1) * (z - 1)) / 8, -((y + 1) * (z - 1)) / 8,
                          ((y + 1) * (z - 1)) / 8, ((y - 1) * (z + 1)) / 8, -((y - 1) * (z + 1)) / 8,
                          ((y + 1) * (z + 1)) / 8, -((y + 1) * (z + 1)) / 8],
                         [-((x - 1) * (z - 1)) / 8, ((x + 1) * (z - 1)) / 8, -((x + 1) * (z - 1)) / 8,
                          ((x - 1) * (z - 1)) / 8, ((x - 1) * (z + 1)) / 8, -((x + 1) * (z + 1)) / 8,
                          ((x + 1) * (z + 1)) / 8, -((x - 1) * (z + 1)) / 8],
                         [-((x - 1) * (y - 1)) / 8, ((x + 1) * (y - 1)) / 8, -((x + 1) * (y + 1)) / 8,
                          ((x - 1) * (y + 1)) / 8, ((x - 1) * (y - 1)) / 8, -((x + 1) * (y - 1)) / 8,
                          ((x + 1) * (y + 1)) / 8, -((x - 1) * (y + 1)) / 8]], dtype=torch.float64, device=self.device)

                    J = qxqyqz @ dxdydz
                    qxyz = torch.inverse(J) @ qxqyqz

                    weight = J.det() * ww[i] * ww[j] * ww[k]
                    Ke = Ke + weight * qxyz.transpose(0, 1) @ conductivity_matrix @ qxyz
                    Fe = Fe + weight * qxyz.transpose(0, 1) @ conductivity_matrix

        idx = torch.ones(8, dtype=torch.bool, device=self.device)
        idx[0] = False

        X0 = torch.zeros(8, 3, dtype=torch.float64, device=self.device)
        X0[idx, :] = torch.inverse(Ke[idx, :][:, idx]) @ Fe[idx, :]

        return Ke, Fe, X0

    def assembly(self, voxel, Ke, Fe, anchor=False):
        """Assemble the stiffness matrix and load matrix."""
        voxel_numel = torch.count_nonzero(voxel)
        voxel_coo = torch.nonzero(voxel, as_tuple=False)
        grid_shape = torch.as_tensor(voxel.shape, dtype=voxel_coo.dtype, device=voxel.device)
        node = voxel.clone().long()
        hex = torch.tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                           [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=torch.int, device=voxel.device)
        
        # Assign node identifiers for every voxel.
        for i in hex:
            tmp = (voxel_coo + i) % grid_shape
            node[tmp[:, 0], tmp[:, 1], tmp[:, 2]] = 1
        
        node_numel = torch.count_nonzero(node)
        node_coo = torch.nonzero(node, as_tuple=False)
        nodeidx = torch.arange(0, node_numel, device=self.device)
        node[node_coo[:, 0], node_coo[:, 1], node_coo[:, 2]] = nodeidx
        
        dof = torch.zeros(voxel_numel, 8, device=self.device)
        kij = torch.zeros(2, voxel_numel * 8 * 8, dtype=torch.float64, device=self.device)
        
        for i in range(0, 8):
            now_node_coo = (voxel_coo + hex[i]) % grid_shape
            dof[:, i] = node[now_node_coo[:, 0], now_node_coo[:, 1], now_node_coo[:, 2]]
        
        kij[0, :] = dof.reshape(-1, 1).repeat(1, 8).reshape(-1)
        kij[1, :] = dof.repeat_interleave(8, dim=0).reshape(-1)
        
        fij = torch.zeros(2, voxel_numel * 8 * 3, dtype=torch.float64, device=self.device)
        fij[0, :] = dof.reshape(-1, 1).repeat(1, 3).reshape(-1)
        fij[1, :] = torch.arange(0, 3, device=self.device).reshape(1, 1, -1).repeat(voxel_numel, 8, 1).contiguous().view(-1)
        
        vK = Ke.repeat(voxel_numel, 1, 1).reshape(-1)
        vF = Fe.repeat(voxel_numel, 1, 1).reshape(-1)
        
        if anchor:
            mask = (torch.logical_or(kij[0, :] == 0, kij[1, :] == 0))
            vK[mask] = 0
            mask = (fij[0, :] == 0)
            vF[mask] = 0
        
        K = torch.sparse_coo_tensor(kij, vK.contiguous().view(-1), (node_numel, node_numel), device=self.device).coalesce()
        F = torch.sparse_coo_tensor(fij, vF.contiguous().view(-1), (node_numel, 3), device=self.device).coalesce().to_dense()
        
        K = 0.5 * (K.transpose(0, 1) + K)
        
        return K, F, node_coo

    def solve(self, K, F, tol=1e-5, max_iter=5000):
        """Solve the linear system."""
        def Kmm(rhs): 
            return ts.mm(K, rhs)
        
        X = linear_cg(Kmm, F, tolerance=tol, max_iter=max_iter)
        return X
    
    def solve_by_torch(self, voxel, tol=1e-5, maxit=5000):
        """Solve with PyTorch."""
        K, f, node_coo = self.assembly(voxel, self.Ke, self.Fe)
        
        def Kmm(rhs): 
            return ts.mm(K, rhs)
        
        X = linear_cg(Kmm, f, tolerance=tol, max_iter=maxit)
        X = X.reshape(-1, 3)
        u = torch.zeros(3, voxel.shape[0], voxel.shape[1], voxel.shape[2], dtype=torch.float64, device=voxel.device)
        u[:, node_coo[:, 0], node_coo[:, 1], node_coo[:, 2]] = X.t()
        
        return u
    
    def homogenized(self, voxel, U, ke_hard, X0):
        """Compute the homogenized conductivity matrix."""
        base_coo = torch.nonzero(voxel, as_tuple=False)
        grid_shape = torch.as_tensor(voxel.shape, dtype=base_coo.dtype, device=voxel.device)
        hex = torch.tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                           [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=torch.int, device=voxel.device)
        
        voxel_u = torch.zeros(base_coo.shape[0], 8, 3, device=voxel.device)
        for i in range(0, 8):
            tmp = (base_coo + hex[i]) % grid_shape
            voxel_u[:, i, :] = U[:, tmp[:, 0], tmp[:, 1], tmp[:, 2]].t()
        
        volume = self.__lx * self.__ly * self.__lz
        u = voxel_u
        CH = torch.zeros(3, 3, dtype=ke_hard.dtype, device=self.device)
        
        L = X0 - u
        CH = torch.einsum('bij,ik,bkl-> jl', L, ke_hard, L)
        CH = 1 / volume * CH
        return CH

# ============================= Main Analyzer =============================
class ElectricalConductivityAnalyzer:
    """Main electrical conductivity analyzer."""

    def __init__(self, config):
        self.config = config
        if self.config.USE_GPU and not torch.cuda.is_available():
            print("CUDA is unavailable; electrical analysis will use CPU.")
            self.config.USE_GPU = False
            self.config.GPU_DEVICE = 'cpu'
        self.device = torch.device(self.config.GPU_DEVICE if self.config.USE_GPU else 'cpu')
        self.voxel_generator = VoxelGenerator(config)
        os.makedirs(self.config.TEMP_DIR, exist_ok=True)  # Ensure the temporary directory exists.

    def process_single_file(self, obj_path):
        """Process one OBJ file."""
        try:
            # Voxelize the input mesh.
            npy_path = self.voxel_generator.process_obj_file(obj_path, self.config.TEMP_DIR)
            if npy_path is None:
                return None

            # Load voxel data.
            voxel = np.load(npy_path)
            voxel = torch.from_numpy(voxel).to(self.device)

            # Perform homogenization.
            homogenizer = NumericalHomogenization(
                voxel.shape[0], voxel.shape[1], voxel.shape[2],
                1.0, 1.0, 1.0,  # Normalized dimensions
                voxel.device,
                self.config.BASE_CONDUCTIVITY
            )

            Ke, Fe, X0 = homogenizer.hexahedron(self.config.BASE_CONDUCTIVITY)
            U = homogenizer.solve_by_torch(voxel, self.config.SOLVER_TOL, self.config.SOLVER_MAX_ITER)
            CH = homogenizer.homogenized(voxel, U, Ke, X0)

            # Calculate simplified results: volume fraction and mean conductivity.
            eff_cond_np = CH.cpu().numpy()
            total_voxels = voxel.shape[0] * voxel.shape[1] * voxel.shape[2]
            solid_voxels = int(torch.count_nonzero(voxel).item())
            volume_fraction = solid_voxels / total_voxels
            
            # For an isotropic material, use the mean diagonal value.
            avg_conductivity = (float(eff_cond_np[0, 0]) + float(eff_cond_np[1, 1]) + float(eff_cond_np[2, 2])) / 3.0

            result = {
                'filename': Path(obj_path).name,
                'volume_fraction': round(volume_fraction, 4),
                'electrical_conductivity': round(avg_conductivity, 3),
                'device': str(voxel.device),
            }

            return result

        except Exception as e:
            print(f"Error while processing file {obj_path}: {e}")
            return None

    def _save_results(self, results):
        """Save simplified results to files."""
        try:
            os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)  # Ensure the output directory exists.
            
            # Save detailed results.
            np.save(os.path.join(self.config.OUTPUT_DIR, 'detailed_results.npy'), results)

            # Save the CSV summary.
            csv_path = os.path.join(self.config.OUTPUT_DIR, 'electrical_conductivity_summary.csv')
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['filename', 'volume_fraction', 'electrical_conductivity']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for result in results:
                    writer.writerow({
                        'filename': result['filename'],
                        'volume_fraction': result['volume_fraction'],
                        'electrical_conductivity': result['electrical_conductivity']
                    })

        except Exception as e:
            print(f"Error while saving results: {e}")


def main():
    """Run the command-line interface."""
    parser = argparse.ArgumentParser(description='Electrical conductivity analysis tool')
    parser.add_argument('--input', '-i', type=str, required=True, help='Input OBJ file or directory')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output directory')
    parser.add_argument('--resolution', '-r', type=int, default=64, help='Voxel resolution')
    parser.add_argument('--base-conductivity', type=float, default=1.0, help='Base electrical conductivity')
    parser.add_argument('--gpu', action='store_true', help='Use GPU acceleration')
    parser.add_argument('--visualization', action='store_true', help='Generate visualization files')

    args = parser.parse_args()

    # Update the configuration.
    Config.VOXEL_RESOLUTION = args.resolution
    Config.BASE_CONDUCTIVITY = args.base_conductivity
    Config.USE_GPU = args.gpu
    Config.ENABLE_VISUALIZATION = args.visualization
    Config.OUTPUT_DIR = args.output or Config.OUTPUT_DIR

    # Create output directories.
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.TEMP_DIR, exist_ok=True)

    # Create the analyzer.
    analyzer = ElectricalConductivityAnalyzer(Config)

    # Process the input.
    if os.path.isfile(args.input):
        result = analyzer.process_single_file(args.input)
        if result:
            analyzer._save_results([result])
            print(f"Processing complete. Results saved to {Config.OUTPUT_DIR}")
        else:
            print("Processing failed")
    elif os.path.isdir(args.input):
        obj_files = [f for f in Path(args.input).glob('*.obj')]
        results = []
        for obj_file in tqdm(obj_files, desc="Processing files"):
            result = analyzer.process_single_file(str(obj_file))
            if result:
                results.append(result)

        if results:
            analyzer._save_results(results)
            print(f"Batch processing complete: {len(results)} files processed. Results saved to {Config.OUTPUT_DIR}")
        else:
            print("No files were processed successfully")
    else:
        print(f"Invalid input path: {args.input}")


if __name__ == "__main__":
    main()
