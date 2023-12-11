import torch
import torch.nn as nn
from pcdet.utils import transform_utils

try:
    from kornia.utils.grid import create_meshgrid3d
    from kornia.geometry.linalg import transform_points
except Exception as e:
    # Note: Kornia team will fix this import issue to try to allow the usage of lower torch versions.
    print('Warning: kornia is not installed correctly, please ignore this warning if you do not use CaDDN. Otherwise, it is recommended to use torch version greater than 1.2 to use kornia properly.')



class Point2ImageProjection(nn.Module):
    def __init__(self, grid_size, pc_range, fuse_mode, stride_dict, fuse_layer, device="cuda"):
        """
        Initializes Grid Generator for frustum features
        """
        super().__init__()
        self.grid_size = torch.as_tensor(grid_size, dtype=torch.float32).to(device)
        self.pc_range = pc_range
        self.fuse_mode = fuse_mode
        self.stride_dict = stride_dict
        self.fuse_layer = fuse_layer

        # Calculate voxel size
        pc_range = torch.as_tensor(pc_range, dtype=torch.float32).reshape(2, 3)
        self.pc_min = pc_range[0].to(device)
        self.pc_max = pc_range[1].to(device)
        self.voxel_size = ((self.pc_max - self.pc_min) / self.grid_size).to(device)
        
        self.grid_to_lidar_dict = {}
        for _layer in self.fuse_layer:
            self.grid_to_lidar_dict[_layer] = self.grid_to_lidar_unproject(pc_min=self.pc_min,
                                                                           voxel_size=self.voxel_size * self.stride_dict[_layer],
                                                                           device=device)

        # Create voxel grid
        if 'ray' in self.fuse_mode:
            self.voxel_grid_dict = {}
            for _layer in self.fuse_layer:
                depth, width, height = (self.grid_size/self.stride_dict[_layer]).ceil().int()
                voxel_grid = create_meshgrid3d(depth=depth, height=height, width=width,
                                               normalized_coordinates=False,
                                               device=device,
                                               dtype=torch.float32)
                voxel_grid = voxel_grid.permute(0, 1, 3, 2, 4)  # XZY-> XYZ
                # Add offsets to center of voxel
                self.voxel_grid_dict[_layer] = voxel_grid + 0.5

    def grid_to_lidar_unproject(self, pc_min, voxel_size, device, dtype=torch.float32):
        """
        Calculate grid to LiDAR unprojection for each plane
        Args:
            pc_min: [x_min, y_min, z_min], Minimum of point cloud range (m)
            voxel_size: [x, y, z], Size of each voxel (m)
        Returns:
            unproject: (4, 4), Voxel grid to LiDAR unprojection matrix
        """
        x_size, y_size, z_size = voxel_size
        x_min, y_min, z_min = pc_min
        unproject = torch.tensor([[x_size, 0, 0, x_min],
                                  [0, y_size, 0, y_min],
                                  [0,  0, z_size, z_min],
                                  [0,  0, 0, 1]],
                                 dtype=dtype,
                                 device=device)  # (4, 4)

        return unproject

    def transform_grid(self, voxel_coords=None, voxel_grid=None, batch_dict=None, layer_name=None):
        """
        Transforms voxel sampling grid into frustum sampling grid
        Args:
            voxel_coords: (B, N, 3), Voxel sampling coordinates
            voxel_grid: (B, X, Y, Z, 3), Voxel sampling grid
            batch_dict
                grid_to_lidar: (4, 4), Voxel grid to LiDAR unprojection matrix
                lidar_to_cam: (B, 4, 4), LiDAR to camera frame transformation
                cam_to_img: (B, 3, 4), Camera projection matrix
        """
        lidar_to_cam=batch_dict["trans_lidar_to_cam"]
        cam_to_img=batch_dict["trans_cam_to_img"]
        fuse_stride = self.stride_dict[layer_name]
        B = lidar_to_cam.shape[0]
        grid_to_lidar = self.grid_to_lidar_dict[layer_name]
        # Create transformation matricies
        V_G = grid_to_lidar  # Voxel Grid -> LiDAR (4, 4)
        C_V = lidar_to_cam  # LiDAR -> Camera (B, 4, 4)
        I_C = cam_to_img  # Camera -> Image (B, 3, 4)
        # Given Voxel coords
        if voxel_coords is not None:
            # Transform to LiDAR
            voxel_coords = voxel_coords[:,[0,3,2,1]] # B,Z,Y,X -> B,X,Y,Z
            point_grid = transform_points(trans_01=V_G.unsqueeze(0), points_1=voxel_coords[:,1:].unsqueeze(0))
            point_grid = point_grid.squeeze()
            batch_idx = voxel_coords[:,0]
            point_count = batch_idx.unique(return_counts=True)
            batch_voxel = torch.zeros(B, max(point_count[1]), 3).to(device=lidar_to_cam.device)
            point_inv = torch.zeros(B, max(point_count[1]), 3).to(device=lidar_to_cam.device)
            batch_mask = torch.zeros(B, max(point_count[1])).to(device=lidar_to_cam.device)
            for _idx in range(B):
                # project points to the non-augment one
                if self.training and 'aug_matrix' in batch_dict.keys():
                    aug_mat_inv = torch.inverse(batch_dict['aug_matrix'][_idx])
                else:
                    aug_mat_inv = torch.eye(3).to(device=lidar_to_cam.device)
                point_inv[_idx,:point_count[1][_idx]] = torch.matmul(aug_mat_inv, point_grid[batch_idx==_idx].t()).t()  
                batch_voxel[_idx,:point_count[1][_idx]] = voxel_coords[batch_idx==_idx][:,1:]
                batch_mask[_idx,:point_count[1][_idx]] = 1

            # Transform to camera frame
            camera_grid = transform_points(trans_01=C_V, points_1=point_inv)
            # Project to image
            image_grid, image_depths = transform_utils.project_to_image(project=I_C, points=camera_grid)
            image_grid = image_grid//fuse_stride
            return image_grid, image_depths, batch_voxel, batch_mask
        # Given voxel grid    
        elif voxel_grid is not None:
            # Reshape to match dimensions
            voxel_grid = voxel_grid.repeat_interleave(repeats=B, dim=0)
            # Transform to LiDAR
            voxel_size = self.voxel_size.view(1,1,1,1,3)
            pc_min = self.pc_min.view(1,1,1,1,3)
            lidar_grid = voxel_grid * voxel_size + pc_min

            # project points to the non-augment one
            if self.training and 'aug_matrix' in batch_dict.keys():
                aug_mat_inv = torch.inverse(batch_dict['aug_matrix'])
                aug_mat_inv = aug_mat_inv.transpose(1,2).reshape(B,1,1,3,3)
                point_grid = torch.matmul(lidar_grid, aug_mat_inv)
            else:
                point_grid = lidar_grid

            # Transform to camera frame
            camera_grid = transform_points(trans_01=C_V, points_1=point_grid)
            I_C = I_C.reshape(B, 1, 1, 3, 4)
            image_grid, image_depths = transform_utils.project_to_image(project=I_C, points=camera_grid)
            image_grid = image_grid//fuse_stride
            return image_grid, image_depths, voxel_grid, lidar_grid


    def forward(self, voxel_coords, batch_dict, layer_name):
        """
        Generates sampling grid for frustum features
        Args:
            voxel_coords: (N, 4), Voxel coordinates
            batch_dict:
                lidar_to_cam: (B, 4, 4), LiDAR to camera frame transformation
                cam_to_img: (B, 3, 4), Camera projection matrix
                image_shape: (B, 2), Image shape [H, W]
        Returns:
            projection_dict: 
                image_grid: (B, N, 2), Image coordinates in X,Y of image plane
                image_depths: (B, N), Image depth
                batch_voxel: (B, N, 3), Voxel coordinates in X,Y,Z of point plane
                point_mask: (B, N), Useful points indictor
        """
        fuse_stride = self.stride_dict[layer_name]
        image_shape=batch_dict["image_shape"]//fuse_stride
        image_grid, image_depths, batch_voxel, batch_mask = self.transform_grid(voxel_coords=voxel_coords, 
                                                                                batch_dict=batch_dict,
                                                                                layer_name=layer_name)

        # Drop points out of range
        point_mask = (image_grid[...,0]>0) & (image_grid[...,0]<image_shape[:,1].unsqueeze(-1)-1) & \
                     (image_grid[...,1]>0) & (image_grid[...,1]<image_shape[:,0].unsqueeze(-1)-1)
        point_mask = point_mask & batch_mask.bool()
        image_grid[~point_mask] = 0
        image_depths[~point_mask] = 0
        batch_voxel[~point_mask] = 0
        projection_dict = {}
        projection_dict['image_grid'] = image_grid.long()
        projection_dict['image_depths'] = image_depths
        projection_dict['batch_voxel'] = batch_voxel.long()
        projection_dict['point_mask'] = point_mask

        # This is a naive version for ray contruction.
        # A more efficient verison should contruct ray from pixel to voxel.
        if 'ray' in self.fuse_mode:
            # Assume Voxel Grid has been auged.
            voxel_grid = self.voxel_grid_dict[layer_name]
            ray_grid, ray_depths, voxel_grid, lidar_grid = self.transform_grid(voxel_grid=voxel_grid, 
                                                                            batch_dict=batch_dict, 
                                                                            layer_name=layer_name)
            ray_mask = (ray_grid[...,0]>0) & (ray_grid[...,0]<image_shape[:,1].reshape(-1,1,1,1)-1) & \
                       (ray_grid[...,1]>0) & (ray_grid[...,1]<image_shape[:,0].reshape(-1,1,1,1)-1) & \
                       (ray_depths > self.pc_range[0])

            projection_dict['ray_grid'] = ray_grid.reshape(ray_grid.shape[0], -1, 2).long()
            projection_dict['ray_depths'] = ray_depths.reshape(ray_depths.shape[0], -1)
            projection_dict['ray_mask'] = ray_mask.reshape(ray_mask.shape[0], -1)
            projection_dict['voxel_grid'] = voxel_grid.reshape(voxel_grid.shape[0], -1, 3).long()
            projection_dict['lidar_grid'] = lidar_grid.reshape(lidar_grid.shape[0], -1, 3)

        return projection_dict
