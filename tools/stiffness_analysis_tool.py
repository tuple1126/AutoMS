"""
Stiffness matrix analysis tool.
Computes an effective stiffness matrix for microstructures using voxelization
and homogenization.
"""

import torch
import numpy as np
import open3d as o3d
import os
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime


def get_voxel_coo(res: int, max_point: np.ndarray, min_point: np.ndarray) -> o3d.core.Tensor:
    """
    Generate voxel coordinates.
    
    Args:
        res: Voxel resolution.
        max_point: Maximum mesh-bound point.
        min_point: Minimum mesh-bound point.
        
    Returns:
        Voxel query-point coordinates.
    """
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


def voxelization_solid(mesh: o3d.geometry.TriangleMesh, res: int, mode: str = 'solid') -> np.ndarray:
    """
    Voxelize a triangle mesh.
    
    Args:
        mesh: Triangle mesh.
        res: Voxel resolution.
        mode: Voxelization mode (``surface`` or ``solid``).
        
    Returns:
        Voxel array.
    """
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(
        mesh,
        voxel_size=(mesh.get_max_bound()[0] - mesh.get_min_bound()[0]) / res
    )
    all_voxel = voxel_grid.get_voxels()
    voxel_array = np.zeros((res, res, res), dtype=int)
    
    for i in range(len(all_voxel)):
        voxel_array[
            all_voxel[i].grid_index[0] - 1,
            all_voxel[i].grid_index[1] - 1,
            all_voxel[i].grid_index[2] - 1
        ] = 1
    
    if mode == 'surface':
        return voxel_array
    
    query_point = get_voxel_coo(res, mesh.get_max_bound(), mesh.get_min_bound())
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh_t)
    occupancy = scene.compute_occupancy(query_point)
    voxel = occupancy.numpy().reshape(res, res, res)
    return voxel.astype(np.int32)
    #return np.logical_or(voxel, voxel_array).astype(np.int32)


def isotropic_elastic_tensor(E: float, v: float) -> torch.Tensor:
    """
    Generate an isotropic elastic tensor.
    
    Args:
        E: Young's modulus.
        v: Poisson's ratio.
        
    Returns:
        6x6 elastic tensor.
    """
    Lambda = v / (1. + v) / (1 - 2. * v) * E
    Mu = 1. / (2. * (1. + v)) * E
    return torch.as_tensor([
        [Lambda + 2 * Mu, Lambda, Lambda, 0, 0, 0],
        [Lambda, Lambda + 2 * Mu, Lambda, 0, 0, 0],
        [Lambda, Lambda, Lambda + 2 * Mu, 0, 0, 0],
        [0, 0, 0, Mu, 0, 0],
        [0, 0, 0, 0, Mu, 0],
        [0, 0, 0, 0, 0, Mu]
    ])


class NumericalHomogenization:
    """Numerical homogenization implementation."""
    
    def __init__(self, nelx: int, nely: int, nelz: int, lx: float, ly: float, lz: float, device: str):
        """Initialize numerical homogenization."""
        self.device = device
        self.__lx = lx
        self.__ly = ly
        self.__lz = lz
        self.__nelx = nelx
        self.__nely = nely
        self.__nelz = nelz
        nel = nelx * nely * nelz
        
        nodeidx = torch.arange(nel, device=self.device).view(nelx, nely, nelz)
        
        index = torch.as_tensor([0], device=self.device)
        nodeidx = torch.cat((nodeidx, torch.index_select(nodeidx, 0, index)), 0)
        nodeidx = torch.cat((nodeidx, torch.index_select(nodeidx, 1, index)), 1)
        nodeidx = torch.cat((nodeidx, torch.index_select(nodeidx, 2, index)), 2)
        
        node_list = [
            nodeidx[0:nelx, 0:nely, 0:nelz].reshape((nel, 1)),
            nodeidx[1:nelx + 1, 0:nely, 0:nelz].reshape((nel, 1)),
            nodeidx[1:nelx + 1, 1:nely + 1, 0:nelz].reshape((nel, 1)),
            nodeidx[0:nelx, 1:nely + 1, 0:nelz].reshape((nel, 1)),
            nodeidx[0:nelx, 0:nely, 1:nelz + 1].reshape((nel, 1)),
            nodeidx[1:nelx + 1, 0:nely, 1:nelz + 1].reshape((nel, 1)),
            nodeidx[1:nelx + 1, 1:nely + 1, 1:nelz + 1].reshape((nel, 1)),
            nodeidx[0:nelx, 1:nely + 1, 1:nelz + 1].reshape((nel, 1))
        ]
        
        self.__cellidx = torch.zeros(8, 3, nel, device=self.device, dtype=torch.int64)
        self.__cellseq = torch.zeros(nel, 8, device=self.device, dtype=torch.int64)
        
        for i in range(8):
            self.__cellidx[i] = self.index2xyz(node_list[i])
            self.__cellseq[:, i] = node_list[i].view(-1)
            
        self.__nodeidx = nodeidx
    
    def index2xyz(self, index: torch.Tensor) -> torch.Tensor:
        """Convert an index to xyz coordinates."""
        x = index.div(self.__nely * self.__nelz, rounding_mode='floor')
        temp = index.remainder(self.__nely * self.__nelz)
        y = temp.div(self.__nelz, rounding_mode='floor')
        z = temp.remainder(self.__nelz)
        xyz = torch.cat((x, y, z), 1)
        return xyz.t()
    
    def solid_cell(self, voxel: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return solid-cell information."""
        voxelidx = torch.arange(0, voxel.numel(), device=self.device)
        voxelidx = torch.masked_select(voxelidx, voxel.type(torch.bool).contiguous().view(-1))
        solid_cell = self.__cellidx[:, :, voxelidx]
        solid_seq = self.__cellseq[voxelidx, :]
        anchor = self.__nodeidx[solid_cell[0, 0, 0], solid_cell[0, 1, 0], solid_cell[0, 2, 0]]
        mask = torch.eq(solid_seq, anchor)
        anchor_index = torch.arange(0, voxelidx.numel(), device=self.device)
        anchor_cell = anchor_index.masked_select(mask.sum(1).type(torch.bool))
        anchor_mask = mask[anchor_cell, :]
        if anchor_mask.dim() == 1:
            anchor_mask.unsqueeze_(0)
        anchor_mask = ~anchor_mask.unsqueeze(2).repeat(1, 1, 3).reshape(-1, 24).unsqueeze(2)
        anchor_mask = torch.as_tensor(anchor_mask, dtype=torch.float64, device=self.device)
        K_mask = anchor_mask.bmm(anchor_mask.transpose(1, 2))
        K_diag = torch.eye(24, 24, dtype=K_mask.dtype, device=self.device).unsqueeze(0).repeat(K_mask.shape[0], 1, 1)
        K_mask = torch.logical_or(K_mask, K_diag)
        F_mask = anchor_mask.repeat(1, 1, 6)
        return solid_cell, anchor_cell, K_mask, F_mask
    
    def hexahedron(self, C: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the hexahedral element stiffness matrix."""
        dx = self.__lx / self.__nelx / 2
        dy = self.__ly / self.__nely / 2
        dz = self.__lz / self.__nelz / 2
        
        pp = torch.as_tensor([-pow(3 / 5, 0.5), 0, pow(3 / 5, 0.5)], dtype=C.dtype, device=self.device)
        ww = torch.as_tensor([5 / 9, 8 / 9, 5 / 9], dtype=C.dtype, device=self.device)
        Ke = torch.zeros(24, 24, dtype=C.dtype, device=self.device)
        Fe = torch.zeros(24, 6, dtype=C.dtype, device=self.device)
        
        dxdydz = torch.as_tensor([
            [-dx, dx, dx, -dx, -dx, dx, dx, -dx], 
            [-dy, -dy, dy, dy, -dy, -dy, dy, dy],
            [-dz, -dz, -dz, -dz, dz, dz, dz, dz]
        ], dtype=C.dtype, device=self.device).t()
        
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    x = pp[i]
                    y = pp[j]
                    z = pp[k]
                    qxqyqz = torch.as_tensor([
                        [-((y - 1) * (z - 1)) / 8, ((y - 1) * (z - 1)) / 8, -((y + 1) * (z - 1)) / 8,
                         ((y + 1) * (z - 1)) / 8, ((y - 1) * (z + 1)) / 8, -((y - 1) * (z + 1)) / 8,
                         ((y + 1) * (z + 1)) / 8, -((y + 1) * (z + 1)) / 8],
                        [-((x - 1) * (z - 1)) / 8, ((x + 1) * (z - 1)) / 8, -((x + 1) * (z - 1)) / 8,
                         ((x - 1) * (z - 1)) / 8, ((x - 1) * (z + 1)) / 8, -((x + 1) * (z + 1)) / 8,
                         ((x + 1) * (z + 1)) / 8, -((x - 1) * (z + 1)) / 8],
                        [-((x - 1) * (y - 1)) / 8, ((x + 1) * (y - 1)) / 8, -((x + 1) * (y + 1)) / 8,
                         ((x - 1) * (y + 1)) / 8, ((x - 1) * (y - 1)) / 8, -((x + 1) * (y - 1)) / 8,
                         ((x + 1) * (y + 1)) / 8, -((x - 1) * (y + 1)) / 8]
                    ], dtype=C.dtype, device=self.device)
                    
                    J = qxqyqz @ dxdydz
                    invJ = torch.inverse(J)
                    qxyz = invJ @ qxqyqz
                    B = torch.zeros(6, 24, dtype=C.dtype, device=self.device)
                    
                    for i_B in range(8):
                        B[:, i_B * 3:(i_B + 1) * 3] = torch.as_tensor([
                            [qxyz[0, i_B], 0, 0],
                            [0, qxyz[1, i_B], 0],
                            [0, 0, qxyz[2, i_B]],
                            [qxyz[1, i_B], qxyz[0, i_B], 0],
                            [0, qxyz[2, i_B], qxyz[1, i_B]],
                            [qxyz[2, i_B], 0, qxyz[0, i_B]]
                        ], dtype=C.dtype, device=self.device)
                    
                    weight = J.det() * ww[i] * ww[j] * ww[k]
                    Ke = Ke + weight * B.transpose(0, 1) @ C @ B
                    Fe = Fe + weight * B.transpose(0, 1) @ C
        
        return Ke, Fe
    
    def voxel_XKF(self, elastic_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute voxel stiffness-matrix parameters."""
        idx = torch.ones(24, dtype=torch.bool, device=self.device)
        idx[[0, 1, 2, 4, 5, 11]] = False
        ke, fe = self.hexahedron(elastic_tensor)
        X0 = torch.zeros(24, 6, dtype=elastic_tensor.dtype, device=self.device)
        X0[idx, :] = torch.inverse(ke[idx, :][:, idx]) @ fe[idx, :]
        return X0, ke, fe
    
    def homogenized(self, voxel: torch.Tensor, U: torch.Tensor, 
                    ke_hard: torch.Tensor, X0: torch.Tensor) -> torch.Tensor:
        """Compute the homogenized stiffness matrix."""
        solid_seq = self.__cellseq[voxel.type(torch.bool).contiguous().view(-1), :]
        n = solid_seq.shape[0]
        volume = self.__lx * self.__ly * self.__lz
        u_ = U.contiguous().view(6, 3, -1).transpose(1, 2)
        u_ = u_.contiguous().view(6, -1).t()
        index_u = torch.empty(n, 8, 3, dtype=torch.int64, device=self.device)
        for i in range(3):
            index_u[:, :, i] = 3 * solid_seq + i
        index_u = index_u.contiguous().view(-1)
        u = u_[index_u, :].contiguous().view(n, 24, 6)
        del index_u
        CH = torch.zeros(6, 6, dtype=ke_hard.dtype, device=self.device)
        L = X0 - u
        CH = torch.einsum('bij,ik,bkl->jl', L, ke_hard, L)
        CH = 1 / volume * CH
        return CH
    
    def assembly(self, voxel: torch.Tensor, Ke: torch.Tensor, Fe: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Assemble the global stiffness matrix."""
        voxelidx = torch.arange(0, voxel.numel(), device=self.device)
        voxelidx = torch.masked_select(voxelidx, voxel.type(torch.bool).contiguous().view(-1))
        solid_seq = self.__cellseq[voxelidx, :]
        _, anchor_cell, K_mask, F_mask = self.solid_cell(voxel)
        indices, inv_indices = torch.unique(solid_seq, sorted=True, return_inverse=True)
        nv = indices.shape[0]
        nc = inv_indices.shape[0]
        dof = torch.zeros(nv, 3, device=self.device)
        dof_indices = torch.zeros(nc, 24, device=self.device, dtype=torch.int64)
        for i in range(3):
            dof[:, i] = 3 * indices + i
            dof_indices[:, torch.arange(0, 8) * 3 + i] = inv_indices * 3 + i
        dof = torch.as_tensor(dof.contiguous().view(-1), dtype=torch.int64, device=self.device)
        dof_indices = torch.as_tensor(dof_indices.unsqueeze(2), dtype=Ke.dtype, device=self.device)
        vK = Ke.repeat(nc, 1, 1)
        vF = Fe.repeat(nc, 1, 1)
        for idx in range(K_mask.shape[0]):
            vK[anchor_cell[idx], :, :] = vK[anchor_cell[idx], :, :] * K_mask[idx, :, :]
            vF[anchor_cell[idx], :, :] = vF[anchor_cell[idx], :, :] * F_mask[idx, :, :]
        Kij = torch.zeros(2, 24 * 24 * inv_indices.shape[0], device=self.device)
        temp = torch.ones(inv_indices.shape[0], 24, dtype=Ke.dtype, device=self.device).unsqueeze(2)
        Kij[0, :] = dof_indices.bmm(temp.transpose(1, 2)).contiguous().view(-1)
        Kij[1, :] = temp.bmm(dof_indices.transpose(1, 2)).contiguous().view(-1)
        K = torch.sparse_coo_tensor(Kij, vK.contiguous().view(-1), (3 * nv, 3 * nv), device=self.device).coalesce()
        Fij = torch.zeros(2, 24 * 6 * inv_indices.shape[0], device=self.device)
        temp_F = torch.ones(inv_indices.shape[0], 6, dtype=Ke.dtype, device=self.device).unsqueeze(2)
        Fij[0, :] = dof_indices.bmm(temp_F.transpose(1, 2)).contiguous().view(-1)
        Fij[1, :] = torch.arange(0, 6, device=self.device).unsqueeze(0).unsqueeze(0).repeat(inv_indices.shape[0], 24, 1).contiguous().view(-1)
        F = torch.sparse_coo_tensor(Fij, vF.contiguous().view(-1), (3 * nv, 6), device=self.device).coalesce().to_dense()
        return K, F, dof
    
    def solve_by_torch(self, voxel: torch.Tensor, Ke: torch.Tensor, Fe: torch.Tensor, 
                       tol: float = 1e-4, maxit: int = 5000) -> torch.Tensor:
        """Solve the displacement field."""
        K, F, dof = self.assembly(voxel, Ke, Fe)
        def Kmm(rhs: torch.Tensor) -> torch.Tensor: 
            return torch.sparse.mm(K, rhs)
        from linear_operator.utils.linear_cg import linear_cg
        X = linear_cg(Kmm, F, tolerance=tol, max_iter=maxit)
        u = torch.zeros(3 * self.__cellseq.shape[0], 6, dtype=X.dtype, device=self.device)
        u[dof, :] = X
        u = u.t().contiguous().view(6, -1, 3)
        u = u.transpose(1, 2).contiguous().view(18, self.__nelx, self.__nely, self.__nelz)
        return u


def run_stiffness_analysis(
    obj_file: str,
    resolution: int = 64,
    voxel_mode: str = 'solid',
    youngs_modulus: float = 1.0,
    poisson_ratio: float = 0.35,
    device: str = 'cuda:0',
    tolerance: float = 1e-4,
    max_iterations: int = 5000,
    output_dir: Optional[str] = None,
    save_results: bool = True,
    silent: bool = False
) -> Dict[str, Any]:
    """
    Run stiffness-matrix analysis.
    
    Args:
        obj_file: OBJ file path.
        resolution: Voxel resolution.
        voxel_mode: Voxelization mode (``surface`` or ``solid``).
        youngs_modulus: Young's modulus.
        poisson_ratio: Poisson's ratio.
        device: Compute device.
        tolerance: Solver tolerance.
        max_iterations: Maximum solver iterations.
        output_dir: Optional output directory.
        save_results: Whether to save results.
        silent: Whether to suppress progress output.
        
    Returns:
        Dictionary containing the stiffness matrix and analysis information.
    """
    try:
        # Treat a directory input as a batch request.
        if os.path.isdir(obj_file):
            print(f"Directory input detected: {obj_file}; switching to batch analysis mode...")
            batch_results = batch_stiffness_analysis(
                directory=obj_file,
                resolution=resolution,
                voxel_mode=voxel_mode,
                youngs_modulus=youngs_modulus,
                poisson_ratio=poisson_ratio,
                device=device,
                tolerance=tolerance,
                max_iterations=max_iterations,
                output_dir=output_dir,
                save_results=save_results,
                silent=silent
            )
            return {
                "success": True,
                "analysis_type": "batch",
                "input_path": obj_file,
                "total_files": len(batch_results),
                "results": batch_results
            }

        # Check whether the file exists.
        if not os.path.exists(obj_file):
            return {
                "success": False,
                "error": f"OBJ file does not exist: {obj_file}",
                "file": obj_file
            }
        
        # Check device availability.
        if device.startswith('cuda') and not torch.cuda.is_available():
            device = 'cpu'
            if not silent:
                print("CUDA is unavailable; switching to CPU computation")
        
        if not silent:
            print(f"Starting analysis for: {obj_file}")
            print(f"Resolution: {resolution}, mode: {voxel_mode}, device: {device}")
        
        # Load the mesh.
        mesh = o3d.io.read_triangle_mesh(obj_file)
        mesh.compute_vertex_normals()
        
        # Voxelize the mesh.
        if not silent:
            print("Voxelizing...")
        voxel_np = voxelization_solid(mesh, resolution, voxel_mode)
        voxel = torch.from_numpy(voxel_np).to(device)
        
        # Calculate the solid volume fraction.
        solid_fraction = torch.sum(voxel).item() / (resolution**3)
        if not silent:
            print(f"Solid volume fraction: {solid_fraction:.4f}")
        
        # Perform homogenization.
        if not silent:
            print("Performing homogenization...")
        homogenizer = NumericalHomogenization(resolution, resolution, resolution, 1, 1, 1, device)
        
        # Generate the elastic tensor.
        C_H = isotropic_elastic_tensor(youngs_modulus, poisson_ratio).to(torch.double).to(device)
        
        # Compute the stiffness matrix.
        X0, Ke, Fe = homogenizer.voxel_XKF(C_H)
        U = homogenizer.solve_by_torch(voxel, Ke, Fe, tolerance, max_iterations)
        CH = homogenizer.homogenized(voxel, U, Ke, X0)
        
        # Convert to a NumPy array and round values.
        stiffness_matrix = np.round(CH.cpu().numpy(), 3)
        
        # Calculate effective elastic constants.
        effective_properties = calculate_effective_properties(stiffness_matrix)
        
        # Prepare results.
        results = {
            "success": True,
            "file": obj_file,
            # "parameters": {
            #     "resolution": resolution,
            #     "voxel_mode": voxel_mode,
            #     "youngs_modulus": youngs_modulus,
            #     "poisson_ratio": poisson_ratio,
            #     "device": device,
            #     "tolerance": tolerance,
            #     "max_iterations": max_iterations
            # },
            # "solid_fraction": solid_fraction,
            # "stiffness_matrix": stiffness_matrix.tolist(),
            "effective_properties": effective_properties,
            # "analysis_time": datetime.now().isoformat()
        }
        
        # Save results.
        if save_results and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.splitext(os.path.basename(obj_file))[0]
            output_file = os.path.join(output_dir, f"{filename}_stiffness_analysis.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            results["output_file"] = output_file
            if not silent:
                print(f"Results saved to: {output_file}")
        
        if not silent:
            print("Stiffness-matrix analysis complete")
        return results
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "file": obj_file,
            "traceback": str(e)
        }


def calculate_effective_properties(stiffness_matrix: np.ndarray) -> Dict[str, float]:
    """
    Calculate effective elastic properties from a stiffness matrix.
    
    Args:
        stiffness_matrix: 6x6 stiffness matrix.
        
    Returns:
        Dictionary of effective elastic properties.
    """
    try:
        C = stiffness_matrix
        
        # Compute the compliance matrix.
        S = np.linalg.inv(C)
        
        # Effective Young's moduli.
        E1 = 1 / S[0, 0]
        E2 = 1 / S[1, 1] 
        E3 = 1 / S[2, 2]
        
        # Effective Poisson's ratios.
        nu12 = -S[0, 1] / S[0, 0]
        nu13 = -S[0, 2] / S[0, 0]
        nu23 = -S[1, 2] / S[1, 1]
        
        # Effective shear moduli.
        G12 = 1 / S[3, 3]
        G13 = 1 / S[4, 4]
        G23 = 1 / S[5, 5]
        
        # Mean values for an isotropic material.
        E_avg = (E1 + E2 + E3) / 3
        nu_avg = (nu12 + nu13 + nu23) / 3
        G_avg = (G12 + G13 + G23) / 3
        
        # Bulk modulus and Lame parameter.
        K = E_avg / (3 * (1 - 2 * nu_avg))  # Bulk modulus
        lambda_lame = nu_avg * E_avg / ((1 + nu_avg) * (1 - 2 * nu_avg))  # Lame parameter
        
        return {
            "E_avg": round(float(E_avg), 3),
            "nu_avg": round(float(nu_avg), 3),
            "G_avg": round(float(G_avg), 3),
            "K": round(float(K), 3)
        }
    except Exception as e:
        return {"error": f"Error while calculating effective properties: {str(e)}"}


def get_obj_files(directory: Optional[str] = None) -> List[str]:
    """
    Return OBJ files from a directory.
    
    Args:
        directory: Directory path. Defaults to ``data/workshop``.
        
    Returns:
        OBJ file paths.
    """
    if directory is None:
        # Search ``data/workshop`` by default.
        current_dir = os.path.dirname(os.path.abspath(__file__))
        directory = os.path.join(os.path.dirname(current_dir), 'data', 'workshop')
    
    if not os.path.exists(directory):
        return []
    
    obj_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.obj'):
                obj_files.append(os.path.join(root, file))
    
    return sorted(obj_files)


def batch_stiffness_analysis(
    obj_files: Optional[List[str]] = None,
    directory: Optional[str] = None,
    silent: bool = False,
    **kwargs: Any
) -> List[Dict[str, Any]]:
    """
    Run batch stiffness-matrix analysis.
    
    Args:
        obj_files: Optional OBJ file list.
        directory: Optional directory to search.
        silent: Whether to suppress progress output.
        **kwargs: Additional arguments passed to ``run_stiffness_analysis``.
        
    Returns:
        Analysis results.
    """
    if obj_files is None:
        obj_files = get_obj_files(directory)
    
    if not obj_files:
        return [{"success": False, "error": "No OBJ files found"}]
    
    # Retain only parameters accepted by ``run_stiffness_analysis``.
    valid_params = ['resolution', 'voxel_mode', 'youngs_modulus', 'poisson_ratio', 'device', 'tolerance', 'max_iterations', 'output_dir', 'save_results', 'silent']
    valid_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    
    results = []
    for obj_file in obj_files:
        if not silent:
            print(f"\nProcessing file {len(results)+1}/{len(obj_files)}: {os.path.basename(obj_file)}")
        result = run_stiffness_analysis(obj_file, **valid_kwargs)
        results.append(result)
    
    return results


if __name__ == "__main__":
    # Smoke test.
    test_files = get_obj_files()
    if test_files:
        print(f"Found {len(test_files)} OBJ files")
        # Test the first file.
        result = run_stiffness_analysis(
            test_files[0],
            resolution=64,
            device='cpu'
        )
        print("Test result:", result.get("success", False))
    else:
        print("No test files found")
