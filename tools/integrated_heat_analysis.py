"""
Integrated thermal conductivity analysis script.
Voxelizes OBJ files, writes NPY data, and computes homogenized thermal
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
    VOXEL_RESOLUTION = 128  # Voxel resolution
    VOXEL_MODE = 'solid'   # Either 'surface' or 'solid'
    ENABLE_VISUALIZATION = False  # Whether to generate visualization files
    
    # Thermal conductivity calculation parameters
    BASE_HEAT = 1.0        # Base thermal conductivity
    SOLVER_TOL = 1e-5      # Solver tolerance
    SOLVER_MAX_ITER = 5000 # Maximum solver iterations
    
    # File-path parameters
    INPUT_DIR = str(WORKSHOP_DIR)           # Input OBJ directory
    OUTPUT_DIR = str(RESULTS_DIR)           # Output directory
    TEMP_DIR = str(CACHE_DIR / 'heat_voxels')
    
    # GPU settings
    USE_GPU = True                  # Whether to use a GPU
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

# ============================= Thermal Conductivity =============================
class NumericalHomogenization:
    """Numerical homogenization for thermal conductivity."""
    
    def __init__(self, nelx, nely, nelz, lx, ly, lz, device, base_heat=1.0):
        self.device = device
        self.__lx = lx
        self.__ly = ly
        self.__lz = lz
        self.__nelx = nelx
        self.__nely = nely
        self.__nelz = nelz
        self.Ke, self.Fe, self.X0 = self.hexahedron(base_heat)
    
    def hexahedron(self, base_heat=1):
        """Compute the hexahedral element matrices."""
        heat_matrix = torch.eye(3, 3, dtype=torch.float64, device=self.device) * base_heat
        
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
                    Ke = Ke + weight * qxyz.transpose(0, 1) @ heat_matrix @ qxyz
                    Fe = Fe + weight * qxyz.transpose(0, 1) @ heat_matrix
        
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
        
        return K.coalesce(), F, node_coo
    
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
        """Compute the homogenized thermal conductivity matrix."""
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
class HeatConductivityAnalyzer:
    """Main thermal conductivity analyzer."""
    
    def __init__(self, config):
        self.config = config
        self.device = self._setup_device()
        self.voxel_generator = VoxelGenerator(config)
        
        # Create output directories.
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(config.TEMP_DIR, exist_ok=True)
    
    def _setup_device(self):
        """Configure the compute device."""
        if self.config.USE_GPU and torch.cuda.is_available():
            device = torch.device(self.config.GPU_DEVICE)
            print(f"Using GPU: {device}")
        else:
            device = torch.device('cpu')
            print("Using CPU")
        return device
    
    # def process_single_file(self, obj_path):
    #     """Process one OBJ file."""
    #     filename = Path(obj_path).stem
    #     print(f"\nProcessing file: {filename}")
        
    #     try:
    #         # 1. Voxelization
    #         print("  - Voxelizing...")
    #         npy_path = self.voxel_generator.process_obj_file(obj_path, self.config.TEMP_DIR)
    #         if npy_path is None:
    #             return None
            
    #         # 2. Load voxel data
    #         voxel_data = np.load(npy_path)
    #         resx, resy, resz = voxel_data.shape
    #         print(f"  - Voxel resolution: {resx}x{resy}x{resz}")
    #         print(f"  - Solid voxels: {np.sum(voxel_data)}")
            
    #         # 3. Thermal conductivity calculation
    #         print("  - Calculating thermal conductivity...")
    #         sol = NumericalHomogenization(resx, resy, resz, 1, 1, 1, self.device, self.config.BASE_HEAT)
    #         voxel_tensor = torch.from_numpy(voxel_data).to(self.device).float().reshape(resx, resy, resz)
            
    #         Ke, Fe, X0 = sol.hexahedron(self.config.BASE_HEAT)
    #         U = sol.solve_by_torch(voxel_tensor, self.config.SOLVER_TOL, self.config.SOLVER_MAX_ITER)
    #         CH = sol.homogenized(voxel_tensor, U, Ke, X0)
            
    #         # 4. Simplified results: volume fraction and mean conductivity
    #         total_voxels = resx * resy * resz
    #         solid_voxels = int(np.sum(voxel_data))
    #         volume_fraction = solid_voxels / total_voxels
            
    #         # For an isotropic material, use the mean diagonal value.
    #         avg_conductivity = (float(CH[0, 0].cpu()) + float(CH[1, 1].cpu()) + float(CH[2, 2].cpu())) / 3.0
            
    #         result = {
    #             'filename': filename,
    #             'volume_fraction': round(volume_fraction, 4),
    #             'thermal_conductivity': round(avg_conductivity, 3)
    #         }
            
    #         print(f"  - Volume fraction: {result['volume_fraction']:.4f}")
    #         print(f"  - Thermal conductivity: {result['thermal_conductivity']:.3f} W/(m*K)")
            
    #         return result
            
    #     except Exception as e:
    #         print(f"  - Error: {e}")
    #         return None
        
    #     finally:
    #         # Clean up temporary files.
    #         if 'npy_path' in locals() and os.path.exists(npy_path):
    #             os.remove(npy_path)
    def process_single_file(self, obj_path):
        """Process one OBJ file."""
        filename = Path(obj_path).stem
        # print(f"\nProcessing file: {filename}")
        
        try:
            # 1. Voxelization
            # print("  - Voxelizing...")
            npy_path = self.voxel_generator.process_obj_file(obj_path, self.config.TEMP_DIR)
            if npy_path is None:
                return None
            
            # 2. Load voxel data
            voxel_data = np.load(npy_path)
            resx, resy, resz = voxel_data.shape
            # print(f"  - Voxel resolution: {resx}x{resy}x{resz}")
            # print(f"  - Solid voxels: {np.sum(voxel_data)}")
            
            # 3. Thermal conductivity calculation
            # print("  - Calculating thermal conductivity...")
            sol = NumericalHomogenization(resx, resy, resz, 1, 1, 1, self.device, self.config.BASE_HEAT)
            voxel_tensor = torch.from_numpy(voxel_data).to(self.device).float().reshape(resx, resy, resz)
            
            Ke, Fe, X0 = sol.hexahedron(self.config.BASE_HEAT)
            U = sol.solve_by_torch(voxel_tensor, self.config.SOLVER_TOL, self.config.SOLVER_MAX_ITER)
            CH = sol.homogenized(voxel_tensor, U, Ke, X0)
            
            # 4. Simplified results: volume fraction and mean conductivity
            total_voxels = resx * resy * resz
            solid_voxels = int(np.sum(voxel_data))
            volume_fraction = solid_voxels / total_voxels
            
            # For an isotropic material, use the mean diagonal value.
            avg_conductivity = (float(CH[0, 0].cpu()) + float(CH[1, 1].cpu()) + float(CH[2, 2].cpu())) / 3.0
            
            result = {
                'filename': filename,
                'volume_fraction': round(volume_fraction, 4),
                'thermal_conductivity': round(avg_conductivity, 3)
            }
            
            # print(f"  - Volume fraction: {result['volume_fraction']:.4f}")
            # print(f"  - Thermal conductivity: {result['thermal_conductivity']:.3f} W/(m*K)")
            
            return result
            
        except Exception as e:
            # print(f"  - Error: {e}")
            return None
        
        finally:
            # Clean up temporary files.
            if 'npy_path' in locals() and os.path.exists(npy_path):
                os.remove(npy_path)
    
    
    def process_directory(self, input_dir):
        """Process every OBJ file in a directory."""
        if not os.path.exists(input_dir):
            print(f"Input directory does not exist: {input_dir}")
            return
        
        # Find all OBJ files.
        obj_files = []
        for ext in ['*.obj', '*.OBJ']:
            obj_files.extend(Path(input_dir).glob(ext))
        
        if not obj_files:
            print(f"No OBJ files found in {input_dir}")
            return
        
        print(f"Found {len(obj_files)} OBJ files")
        
        # Process all files.
        results = []
        pbar = tqdm(obj_files, desc="Processing progress")
        
        for obj_file in pbar:
            pbar.set_description(f"Processing: {obj_file.name}")
            result = self.process_single_file(str(obj_file))
            if result:
                results.append(result)
        
        # Save summary results.
        # self._save_results(results)
        
        print(f"\nProcessing complete. Successfully processed {len(results)}/{len(obj_files)} files")
        print(f"Results saved in: {self.config.OUTPUT_DIR}")
    
    def _save_results(self, results):
        """Save results."""
        # if not results:
        #     return
        
        # # Save detailed results in NPY format.
        # detailed_path = os.path.join(self.config.OUTPUT_DIR, 'detailed_results.npy')
        # np.save(detailed_path, results)
        
        # # Save a simplified CSV summary.
        # csv_path = os.path.join(self.config.OUTPUT_DIR, 'heat_conductivity_summary.csv')
        # with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(['filename', 'volume_fraction', 'thermal_conductivity (W/m*K)'])
            
        #     for result in results:
        #         writer.writerow([
        #             result['filename'],
        #             f"{result['volume_fraction']:.4f}",
        #             f"{result['thermal_conductivity']:.3f}"
        #         ])

def main():
    """Run the command-line interface."""
    parser = argparse.ArgumentParser(description="Integrated thermal conductivity analysis script")
    parser.add_argument('--input_dir', type=str, default=Config.INPUT_DIR, help='Input OBJ directory')
    parser.add_argument('--output_dir', type=str, default=Config.OUTPUT_DIR, help='Output directory')
    parser.add_argument('--resolution', type=int, default=Config.VOXEL_RESOLUTION, help='Voxel resolution')
    parser.add_argument('--mode', type=str, default=Config.VOXEL_MODE, choices=['surface', 'solid'], help='Voxelization mode')
    parser.add_argument('--base_heat', type=float, default=Config.BASE_HEAT, help='Base thermal conductivity')
    parser.add_argument('--gpu', default=True,action='store_true', help='Use GPU acceleration')
    parser.add_argument('--visualization', action='store_true', help='Generate voxel visualization files')
    
    args = parser.parse_args()
    
    # Update the configuration.
    Config.INPUT_DIR = args.input_dir
    Config.OUTPUT_DIR = args.output_dir
    Config.VOXEL_RESOLUTION = args.resolution
    Config.VOXEL_MODE = args.mode
    Config.BASE_HEAT = args.base_heat
    Config.USE_GPU = args.gpu
    Config.ENABLE_VISUALIZATION = args.visualization
    
    # Print the configuration.
    print("=" * 60)
    print("Thermal conductivity analysis configuration:")
    print(f"  Input directory: {Config.INPUT_DIR}")
    print(f"  Output directory: {Config.OUTPUT_DIR}")
    print(f"  Voxel resolution: {Config.VOXEL_RESOLUTION}")
    print(f"  Voxelization mode: {Config.VOXEL_MODE}")
    print(f"  Base thermal conductivity: {Config.BASE_HEAT}")
    print(f"  Use GPU: {Config.USE_GPU}")
    print(f"  Generate visualization: {Config.ENABLE_VISUALIZATION}")
    print("=" * 60)
    
    # Create and run the analyzer.
    analyzer = HeatConductivityAnalyzer(Config)
    analyzer.process_directory(Config.INPUT_DIR)

if __name__ == '__main__':
    main()
