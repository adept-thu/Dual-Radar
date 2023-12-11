from typing_extensions import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .point_to_image_projection import Point2ImageProjection
try:
    from kornia.losses.focal import BinaryFocalLossWithLogits
except:
    pass 

class VoxelFieldFusion(nn.Module):
    def __init__(self, model_cfg, grid_size, pc_range, disc_cfg=None, device="cuda"):
        """
        Initializes module to transform frustum features to voxel features via 3D transformation and sampling
        Args:
            model_cfg: EasyDict, Module configuration
            grid_size: [X, Y, Z], Voxel grid size
            pc_range: [x_min, y_min, z_min, x_max, y_max, z_max], Voxelization point cloud range (m)
        """
        super().__init__()
        self.model_cfg = model_cfg
        self.grid_size = grid_size
        self.pc_range = pc_range
        self.fuse_mode = model_cfg.get('FUSE', None)
        self.fuse_stride = model_cfg.get('STRIDE', {})
        self.image_interp = model_cfg.get('INTERPOLATE', True)
        self.loss_cfg = model_cfg.get('LOSS', None)
        self.point_projector = Point2ImageProjection(grid_size=grid_size,
                                                     pc_range=pc_range,
                                                     fuse_mode=self.fuse_mode,
                                                     stride_dict=self.fuse_stride,
                                                     fuse_layer=model_cfg.LAYER_CHANNEL.keys())
        if 'ray' in self.fuse_mode:
            self.fuse_thres = model_cfg.get('FUSE_THRES', 0.5)
            self.depth_thres = model_cfg.get('DEPTH_THRES', 70)
            self.ray_thres = model_cfg.get('RAY_THRES', 1)
            self.block_num = model_cfg.get('BLOCK_NUM', 1)
            self.ray_sample = model_cfg.get('SAMPLE', {'METHOD': 'naive'})
            self.topk_ratio = model_cfg.get('TOPK_RATIO', 0.25)
            self.position_type = model_cfg.get('POSITION_TYPE', None)
            self.ray_blocks = nn.ModuleDict()
            self.img_blocks = nn.ModuleDict()
            self.sample_blocks = nn.ModuleDict()
            self.fuse_blocks = nn.ModuleDict()
            self.judge_voxels = {}
            self.kernel_size = self.loss_cfg.get("GT_KERNEL", 1)
            if 'FocalLoss' in self.loss_cfg.NAME:
                self.loss_func = BinaryFocalLossWithLogits(**self.loss_cfg.ARGS)
            else: raise NotImplementedError
            if 'BCELoss' in self.ray_sample.get("LOSS", "BCELoss"):
                self.sample_loss = torch.nn.BCEWithLogitsLoss()
            else: raise NotImplementedError

            for _layer in model_cfg.LAYER_CHANNEL.keys():
                ray_block = OrderedDict()
                img_block = OrderedDict()
                sample_blocks = OrderedDict()
                ray_in_channel, img_in_channel = 3, model_cfg.LAYER_CHANNEL[_layer]
                out_channel = model_cfg.LAYER_CHANNEL[_layer]
                sparse_shape = np.ceil(self.grid_size/self.fuse_stride[_layer]).astype(int)
                sparse_shape = sparse_shape[::-1] + [1, 0, 0]
                self.judge_voxels[_layer] = torch.zeros(*sparse_shape).to(device=device)
                if self.position_type is not None:
                    img_in_channel = img_in_channel + 2
                for _block in range(self.block_num):
                    ray_block['ray_{}_conv_{}'.format(_layer,_block)] = nn.Linear(in_features=ray_in_channel,
                                                                                  out_features=out_channel,
                                                                                  bias=True)
                    img_block['img_{}_conv_{}'.format(_layer,_block)] = nn.Conv2d(in_channels=img_in_channel,
                                                                                  out_channels=out_channel,
                                                                                  kernel_size=1,
                                                                                  stride=1,
                                                                                  padding=0,
                                                                                  bias=True)
                    if "learnable" in self.ray_sample.METHOD:
                        sample_blocks['sample_{}_conv_{}'.format(_layer,_block)] = nn.Conv2d(in_channels=img_in_channel,
                                                                                    out_channels=out_channel if _block<self.block_num-1 else 1,
                                                                                    kernel_size=3,
                                                                                    stride=1,
                                                                                    padding=1,
                                                                                    bias=True)
                    if _block < self.block_num - 1:
                        ray_block['ray_{}_relu_{}'.format(_layer, _block)] = nn.ReLU()
                        img_block['img_{}_bn_{}'.format(_layer,_block)] = nn.BatchNorm2d(out_channel)
                        img_block['img_{}_relu_{}'.format(_layer,_block)] = nn.ReLU()
                        if "learnable" in self.ray_sample.METHOD:
                            sample_blocks['sample_{}_bn_{}'.format(_layer,_block)] = nn.BatchNorm2d(out_channel)
                            sample_blocks['sample_{}_relu_{}'.format(_layer,_block)] = nn.ReLU()
                    ray_in_channel = out_channel
                    img_in_channel = out_channel
                
                # weight init
                for _ray in ray_block:
                    if 'relu' in _ray or 'bn' in _ray: continue
                    nn.init.normal_(ray_block[_ray].weight, mean=0, std=0.01)
                    if ray_block[_ray].bias is not None:
                        nn.init.constant_(ray_block[_ray].bias, 0)
                for _img in img_block:
                    if 'relu' in _img or 'bn' in _img: continue
                    nn.init.normal_(img_block[_img].weight, mean=0, std=0.01)
                    if img_block[_img].bias is not None:
                        nn.init.constant_(img_block[_img].bias, 0)
                if "learnable" in self.ray_sample.METHOD:
                    for _sample in sample_blocks:
                        if 'relu' in _sample or 'bn' in _sample: continue
                        nn.init.normal_(sample_blocks[_sample].weight, mean=0, std=0.01)
                        if sample_blocks[_sample].bias is not None:
                            nn.init.constant_(sample_blocks[_sample].bias, 0)
                    self.sample_blocks[_layer] = nn.Sequential(sample_blocks)
                
                self.ray_blocks[_layer] = nn.Sequential(ray_block)
                self.img_blocks[_layer] = nn.Sequential(img_block)
                self.fuse_blocks[_layer] = nn.Sequential(nn.Linear(in_features=out_channel*2,
                                                                   out_features=out_channel,
                                                                   bias=True),
                                                         nn.ReLU())
                
    
    def position_encoding(self, H, W):
        if self.position_type == "absolute":
            min_value=(0, 0)
            max_value=(W-1, H-1)
        elif self.position_type == "relative":
            min_value=(-1.0, -1.0)
            max_value=(1.0, 1.0)

        loc_w = torch.linspace(min_value[0], max_value[0], W).cuda().unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(min_value[1], max_value[1], H).cuda().unsqueeze(1).repeat(1, W)
        loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
        return loc

    def fusion(self, image_feat, voxel_feat, image_grid):
        """
        Fuses voxel features and image features
        Args:
            image_feat: (C, H, W), Encoded image features
            voxel_feat: (N, C), Encoded voxel features
            image_grid: (N, 2), Image coordinates in X,Y of image plane
        Returns:
            voxel_feat: (N, C), Fused voxel features
        """
        image_grid = image_grid[:,[1,0]] # X,Y -> Y,X

        if 'sum' in self.fuse_mode:
            #fuse_feat = image_feat[:,image_grid[:,0],image_grid[:,1]]
            voxel_feat =  voxel_feat + image_feat[:,image_grid[:,0],image_grid[:,1]].permute(1,0)
            #import pdb;pdb.set_trace()
        elif 'mean' in self.fuse_mode:
            fuse_feat = image_feat[:,image_grid[:,0],image_grid[:,1]]
            voxel_feat = (voxel_feat + fuse_feat.permute(1,0)) / 2
        else:
            raise NotImplementedError
        
        return voxel_feat

    def forward(self, batch_dict, encoded_voxel=None, encoded_feat2d=None, layer_name=None):
        """
        Generates voxel features via 3D transformation and sampling
        """
        # Generate sampling grid for frustum volume
        projection_dict = self.point_projector(voxel_coords=encoded_voxel.indices.float(),
                                               batch_dict=batch_dict,
                                               layer_name=layer_name)

        ray_pred, ray_gt, ray_multi, sample_pred, sample_gt = [], [], [], [], []
        for _idx in range(len(batch_dict['image_shape'])):
            if encoded_feat2d is None:
                image_feat = batch_dict['image_features'][_idx]
            else:
                image_feat = encoded_feat2d[_idx]
            raw_shape = tuple(batch_dict['image_shape'][_idx].cpu().numpy()//self.fuse_stride[layer_name])
            feat_shape = image_feat.shape[-2:]
            if self.image_interp:
                image_feat = F.interpolate(image_feat.unsqueeze(0), size=raw_shape, mode='bilinear')[0]
            index_mask = encoded_voxel.indices[:,0]==_idx
            voxel_feat = encoded_voxel.features[index_mask]
            image_grid = projection_dict['image_grid'][_idx]
            point_mask = projection_dict['point_mask'][_idx]
            image_depth = projection_dict['image_depths'][_idx]
            # Fuse 3D LiDAR point with 2D image feature
            # point_mask[len(voxel_feat):] -> 0 for batch construction
            voxel_mask = point_mask[:len(voxel_feat)]
            if self.training and 'overlap_mask' in batch_dict.keys():
                overlap_mask = batch_dict['overlap_mask'][_idx]
                is_overlap = overlap_mask[image_grid[:,1], image_grid[:,0]].bool()
                if 'depth_mask' in batch_dict.keys():
                    depth_mask = batch_dict['depth_mask'][_idx]
                    depth_range = depth_mask[image_grid[:,1], image_grid[:,0]]
                    is_inrange = (image_depth > depth_range[:,0]) & (image_depth < depth_range[:,1])
                    is_overlap = is_overlap & (~is_inrange)

                image_grid = image_grid[~is_overlap]
                point_mask = point_mask[~is_overlap]
                voxel_mask = voxel_mask & (~is_overlap[:len(voxel_feat)])
            if not self.image_interp:
                image_grid = image_grid.float()
                image_grid[:,0] *= (feat_shape[1]/raw_shape[1])
                image_grid[:,1] *= (feat_shape[0]/raw_shape[0])
                image_grid = image_grid.long()
            voxel_feat[voxel_mask] = self.fusion(image_feat, voxel_feat[voxel_mask], 
                                                    image_grid[point_mask])

            encoded_voxel.features[index_mask] = voxel_feat

            # Predict 3D Ray from 2D image feature
            if 'ray' in self.fuse_mode:
                # Get projected variables for ray rendering
                ray_mask = projection_dict['ray_mask'][_idx]
                ray_depth = projection_dict['ray_depths'][_idx]
                ray_mask = ray_mask & (ray_depth<self.depth_thres)
                ray_voxel = projection_dict['voxel_grid'][_idx][ray_mask]
                ray_grid = projection_dict['ray_grid'][_idx][ray_mask]
                lidar_grid = projection_dict['lidar_grid'][_idx][ray_mask]

                # Get shape of render voxel and grid
                render_shape = batch_dict['image_shape'][_idx]//self.fuse_stride[layer_name]
                render_shape = render_shape.flip(dims=[0]).unsqueeze(0)
                # Add positional embedding if needed
                if self.position_type is not None:
                    H, W = image_feat.shape[-2:]
                    if self.position_type is not None:
                        pos_embedding = self.position_encoding(H=H, W=W)
                    image_feat = torch.cat([image_feat, pos_embedding.squeeze()], dim=0)

                # Paint GT Voxel
                voxel_indices = encoded_voxel.indices[index_mask][:,1:].long()
                judge_voxel = self.judge_voxels[layer_name] * 0
                judge_voxel[voxel_indices[:,0], voxel_indices[:,1], voxel_indices[:,2]] = 1

                # Select TOP number
                topk_num = int(encoded_voxel.indices.shape[0] * self.topk_ratio)
                render_feat, ray_logit, sample_mask, ray_mask, grid_prob = self.ray_render(ray_grid, lidar_grid, image_grid[point_mask], 
                                                                                           image_feat.unsqueeze(0), 
                                                                                           render_shape, layer_name, topk_num)

                # Find the pair of rendered voxel and orignial voxel
                render_indices = ray_voxel[sample_mask][ray_mask][:,[2,1,0]].long()
                render_mask = judge_voxel[render_indices[:,0], render_indices[:,1], render_indices[:,2]].bool()
                judge_voxel = judge_voxel * 0
                judge_voxel[render_indices[render_mask][:,0], render_indices[render_mask][:,1], render_indices[render_mask][:,2]] = 1
                voxel_mask = judge_voxel[voxel_indices[:,0], voxel_indices[:,1], voxel_indices[:,2]].bool()

                # Add rendered point to Sparse Tensor
                render_indices = render_indices[~render_mask].int()
                render_feat = render_feat[~render_mask]
                render_batch = _idx * torch.ones(len(render_indices),1).to(device=render_indices.device).int()
                render_indices = torch.cat([render_batch, render_indices], dim=1)
                encoded_voxel.indices = torch.cat([encoded_voxel.indices, render_indices], dim=0)
                #encoded_voxel.features = torch.cat([encoded_voxel.features, render_feat], dim=0)
                encoded_voxel = encoded_voxel.replace_feature(torch.cat([encoded_voxel.features, render_feat], dim=0))

                if not self.training:
                    continue

                # Find the points in GT ray
                grid_mask = torch.zeros(tuple(render_shape[0].cpu().numpy())).to(device=image_grid.device)
                grid_mask[image_grid[point_mask][:,0], image_grid[point_mask][:,1]] = 1
                identity_mask = grid_mask[ray_grid[sample_mask][:,0], ray_grid[sample_mask][:,1]].bool()

                # Find the pair of rendered voxel and orignial voxel
                judge_voxel = judge_voxel * 0
                if self.kernel_size > 1:
                    judge_voxel = self.gaussian3D(judge_voxel, voxel_indices)
                else:
                    judge_voxel[voxel_indices[:,0], voxel_indices[:,1], voxel_indices[:,2]] = 1
                
                
                render_indices = ray_voxel[sample_mask][identity_mask][:,[2,1,0]].long()
                render_mask = judge_voxel[render_indices[:,0], render_indices[:,1], render_indices[:,2]].float()

                if identity_mask.sum() > 0:
                    ray_logit = ray_logit[identity_mask]
                else:
                    # Avoid Loss Nan
                    ray_logit = ray_logit.sum()[None] * 0
                    render_mask = render_mask.sum()[None] * 0
                ray_pred.append(ray_logit)
                ray_gt.append(render_mask)

                if grid_prob is not None:
                    grid_prob = grid_prob[0]
                    grid_gt = torch.zeros_like(grid_prob)
                    box2d_gt = batch_dict['gt_boxes2d'][_idx]

                    if "affine_matrix" in batch_dict:
                        box2d_gt = box2d_gt.reshape(-1, 2, 2)
                        box2d_gt = torch.cat([box2d_gt, torch.ones(box2d_gt.shape[0],2,1).to(box2d_gt.device)], dim=-1)
                        box2d_T = box2d_gt @ batch_dict["affine_matrix"][_idx, :2].T
                        norm_T = box2d_gt @ batch_dict["affine_matrix"][_idx, -1].T
                        box2d_gt = (box2d_T / norm_T[...,None])
                        box2d_gt[...,0] = box2d_gt[...,0].clip(min=0, max=raw_shape[1])
                        box2d_gt[...,1] = box2d_gt[...,1].clip(min=0, max=raw_shape[0])
                        box2d_gt = box2d_gt.reshape(-1, 4)

                    box2d_gt = (box2d_gt//self.fuse_stride[layer_name]).long()
                    if "gaussian" in self.ray_sample.get("GT_TYPE", "box"):
                        grid_gt = self.gaussian2D(grid_gt, box2d_gt)
                    else:
                        for _box in box2d_gt:
                            grid_gt[:,_box[1]:_box[3],_box[0]:_box[2]] = 1
                    sample_pred.append(grid_prob)
                    sample_gt.append(grid_gt)

        if self.training and self.loss_cfg is not None:
            ray_pred = torch.cat(ray_pred, dim=0)
            ray_gt = torch.cat(ray_gt, dim=0)
            if len(ray_multi) > 0:
                ray_multi = torch.cat(ray_multi, dim=0)
            if len(sample_pred) > 0:
                sample_pred = torch.cat(sample_pred, dim=0)
                sample_gt = torch.cat(sample_gt, dim=0)
            loss_dict = self.get_loss(ray_pred, ray_gt, ray_multi, sample_pred, sample_gt)
            for _key in loss_dict:
                batch_dict[_key+'_'+layer_name] = loss_dict[_key]

        return encoded_voxel, batch_dict


    def ray_render(self, ray_grid, ray_feat, image_grid, image_feat, shape, layer_name, topk_num, min_n=-1, max_n=1):   
        grid_prob = None
        window_size = self.ray_sample.WINDOW // self.fuse_stride[layer_name]
        grid_x = torch.arange(0, ((shape[0,0]/window_size).ceil()+1)*window_size+1, step=window_size)
        range_x = torch.stack([grid_x[:-1], grid_x[1:]-1]).transpose(0,1).to(device=image_grid.device)
        grid_y = torch.arange(0, ((shape[0,1]/window_size).ceil()+1)*window_size+1, step=window_size)
        range_y = torch.stack([grid_y[:-1], grid_y[1:]-1]).transpose(0,1).to(device=image_grid.device)
        mask_x = (image_grid[:,0][None,:] >= range_x[:,0][:,None]) & \
                    (image_grid[:,0][None,:] <= range_x[:,1][:,None])
        mask_y = (image_grid[:,1][None,:] >= range_y[:,0][:,None]) & \
                    (image_grid[:,1][None,:] <= range_y[:,1][:,None])
        grid_mask = mask_x[:,None,:] & mask_y[None,:,:]
        grid_count = grid_mask.sum(-1)

        if "uniform" in self.ray_sample.METHOD:
            sample_num = len(image_grid) * self.ray_sample.RATIO
            grid_count = (grid_count > 0) * sample_num // (grid_count > 0).sum()
        elif "density" in self.ray_sample.METHOD: 
            grid_count = grid_count * self.ray_sample.RATIO
        elif "sparsity" in self.ray_sample.METHOD:
            sample_count = grid_count[grid_count > 0]
            sample_num, sample_idx = (sample_count * self.ray_sample.RATIO).sort()
            sample_count[sample_idx] = sample_num.long().flip(0)
            grid_count[grid_count > 0] = sample_count

        if "all" in self.ray_sample.METHOD:
            grid_sample = torch.ones(grid_count.shape[0]*window_size, grid_count.shape[1]*window_size).bool().to(device=image_grid.device)
        else:
            grid_sample = torch.rand(*grid_count.shape, window_size, window_size).to(device=image_grid.device)
            grid_ratio = grid_count/(window_size**2)
            grid_sample = grid_sample < grid_ratio[...,None,None]
            grid_sample = grid_sample.permute(0, 2, 1, 3)
            grid_sample = grid_sample.reshape(grid_sample.shape[0]*window_size, -1)

        if "learnable" in self.ray_sample.METHOD:
            grid_prob = self.sample_blocks[layer_name](image_feat)
            grid_mask = (grid_prob.sigmoid() > self.ray_sample.THRES).squeeze()
            grid_mask = grid_mask.transpose(0,1)
            grid_sample[:grid_mask.shape[0],:grid_mask.shape[1]] = grid_mask & grid_sample[:grid_mask.shape[0],:grid_mask.shape[1]]

        sample_mask = grid_sample[ray_grid[:,0], ray_grid[:,1]]
        ray_grid = ray_grid[sample_mask]
        ray_feat = ray_feat[sample_mask]

        # Get feature embedding
        image_feat = self.img_blocks[layer_name](image_feat)
        ray_feat = self.ray_blocks[layer_name](ray_feat)
        # Subtract 1 since pixel indexing from [0, shape - 1]
        norm_coords = ray_grid / (shape - 1) * (max_n - min_n) + min_n
        norm_coords = norm_coords.reshape(1,1,-1,2)
        grid_feat = F.grid_sample(input=image_feat, grid=norm_coords, mode="bilinear", padding_mode="zeros")
        grid_feat = grid_feat[0,:,0].transpose(0,1)
        ray_logit = (ray_feat * grid_feat).sum(-1)
        ray_prob = ray_logit.sigmoid()

        if self.training:
            if len(ray_prob) > topk_num:
                ray_topk = torch.topk(ray_prob, topk_num)[1]
                ray_mask = torch.zeros_like(ray_prob).bool()
                ray_mask[ray_topk] = True
            else:
                ray_mask = torch.ones_like(ray_prob).bool()
        else:
            ray_mask = (ray_prob > self.fuse_thres)
            if ray_mask.sum() > topk_num:
                ray_topk = torch.topk(ray_prob, topk_num)[1]
                top_mask = torch.zeros_like(ray_prob).bool()
                top_mask[ray_topk] = True
                ray_mask = ray_mask & top_mask
        ray_prob = ray_prob[ray_mask]
        render_feat = torch.cat([ray_feat[ray_mask], grid_feat[ray_mask]], dim=1)
        render_feat = self.fuse_blocks[layer_name](render_feat)
        render_feat = render_feat * ray_prob.unsqueeze(-1)
        
        return render_feat, ray_logit, sample_mask, ray_mask, grid_prob

    def get_loss(self, ray_pred, ray_gt, ray_multi, sample_pred, sample_gt):
        loss_dict = {}
        loss_ray = self.loss_func(ray_pred[None,None,:], ray_gt[None,None,:])
        if self.loss_cfg.ARGS["reduction"] == "sum":
            loss_ray = loss_ray / max((ray_gt==1).sum(), 1)
        if len(ray_multi) > 0:
            loss_dict["ray_loss"] = self.loss_cfg.WEIGHT * (ray_multi * loss_ray.squeeze()).mean()
        else:
            loss_dict["ray_loss"] = self.loss_cfg.WEIGHT * loss_ray
        if len(sample_pred) > 0:
            loss_sample = self.sample_loss(sample_pred, sample_gt)
            loss_dict["sample_loss"] = self.ray_sample.WEIGHT * loss_sample
        return loss_dict

    def gaussian3D(self, voxel, grid, sigma_factor=3):
        kernel = self.kernel_size//2
        sigma = self.kernel_size / sigma_factor
        loc_range = np.linspace(-kernel, kernel, self.kernel_size).astype(np.int)
        max_shape = voxel.shape
        for _x in loc_range:
            for _y in loc_range:
                for _z in loc_range:
                    # generate gaussian-like GT with a factor sigma
                    gauss = np.exp(-(_x * _x + _y * _y + _z * _z) / ((2 * sigma * sigma) * sigma))
                    voxel[(grid[:,0]+_x).clip(min=0,max=max_shape[0]-1), 
                          (grid[:,1]+_y).clip(min=0,max=max_shape[1]-1), 
                          (grid[:,2]+_z).clip(min=0,max=max_shape[2]-1)] = gauss
        
        return voxel

    def gaussian2D(self, grid, boxes, sigma_factor=3, device="cuda"):        
        box_wh = (boxes[:,-2:] - boxes[:,:2]).cpu().numpy()
        _keep = (box_wh>0).all(-1)
        boxes = boxes[_keep]
        box_wh = box_wh[_keep]
        for _i, _box in enumerate(boxes):
            w, h = box_wh[_i]//2
            y = torch.arange(-h, h+1)[:,None].to(device=device)
            x = torch.arange(-w, w+1)[None,:].to(device=device)
            sigma_x, sigma_y = (2*w)/sigma_factor, (2*h)/sigma_factor
            # generate gaussian-like GT
            gauss = torch.exp(-(x * x + y * y) / (2 * sigma_x * sigma_y))
            h = min(box_wh[_i][1], gauss.shape[0])
            w = min(box_wh[_i][0], gauss.shape[1])
            gauss_area = grid[0,_box[1]:_box[1]+h,_box[0]:_box[0]+w]
            gauss_mask = gauss[:h, :w] > gauss_area[:h, :w]
            gauss_area[gauss_mask] = gauss[:h, :w][gauss_mask]
            grid[0,_box[1]:_box[1]+h,_box[0]:_box[0]+w] = gauss_area

        return grid
