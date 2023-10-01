import copy
import pickle
import json

import numpy as np
from skimage import io

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, common_utils
from .object3d_astyx import Object3dAstyx, inv_trans
from ..dataset import DatasetTemplate


class AstyxDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

        self.astyx_infos = []
        self.include_astyx_data(self.mode)

        #self.pc_type = self.dataset_cfg.POINT_CLOUD_TYPE[0]
        if 'radar' in self.dataset_cfg.POINT_CLOUD_TYPE and 'lidar' in self.dataset_cfg.POINT_CLOUD_TYPE :
            self.pc_type = 'fusion'
        else:
            self.pc_type = self.dataset_cfg.POINT_CLOUD_TYPE[0]

    def include_astyx_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading Astyx dataset')
        astyx_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                astyx_infos.extend(infos)

        self.astyx_infos.extend(astyx_infos)

        if self.logger is not None:
            self.logger.info('Total samples for Astyx dataset: %d' % (len(astyx_infos)))

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training,
            root_path=self.root_path, logger=self.logger
        )
        self.split = split
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

    def get_lidar(self, idx):
        lidar_file = self.root_split_path / 'lidar_vlp16' / ('%s.txt' % idx)
        assert lidar_file.exists()
        return np.loadtxt(str(lidar_file), dtype=np.float32, skiprows=1, usecols=(0,1,2,3))

    def get_radar(self, idx):
        radar_file = self.root_split_path / 'radar_6455' / ('%s.txt' % idx)
        assert radar_file.exists()
        radar_points = np.loadtxt(str(radar_file), dtype=np.float32, skiprows=2, usecols=(0,1,2,3))
        x = radar_points[:, 0]
        z = radar_points[:, 2]
        x = np.sqrt(x*x + z*z)*np.cos(20/96*np.arctan2(z, x))
        z = np.sqrt(x*x + z*z)*np.sin(20/96*np.arctan2(z, x))
        radar_points[:, 0] = x
        radar_points[:, 2] = z
        return radar_points

    def get_pointcloud(self, idx, pc_type):
        if pc_type == 'lidar':
            lidar_file = self.root_split_path / 'lidar_vlp16' / ('%s.txt' % idx)
            assert lidar_file.exists()
            return np.loadtxt(str(lidar_file), dtype=np.float32, skiprows=1, usecols=(0, 1, 2, 3))
        elif pc_type == 'radar':
            radar_file = self.root_split_path / 'radar_6455' / ('%s.txt' % idx)
            assert radar_file.exists()
            #return np.loadtxt(str(radar_file), dtype=np.float32, skiprows=2, usecols=(0, 1, 2, 3, 4))
            pc = np.loadtxt(str(radar_file), dtype=np.float32, skiprows=2, usecols=(0, 1, 2, 3, 4))
            pc[:, -1] -= 45
            x = pc[:, 0]
            z = pc[:, 2]
            x = np.sqrt(x*x + z*z)*np.cos(20/96*np.arctan2(z, x))
            z = np.sqrt(x*x + z*z)*np.sin(20/96*np.arctan2(z, x))
            pc[:, 0] = x
            pc[:, 2] = z
            return pc
        else:
            pass

    def get_image_shape(self, idx):
        img_file = self.root_split_path / 'camera_front' / ('%s.jpg' % idx)
        assert img_file.exists()
        return np.array(io.imread(img_file).shape[:2], dtype=np.int32)

    def get_label(self, idx):
        label_file = self.root_split_path / 'groundtruth_obj3d' / ('%s.json' % idx)
        assert label_file.exists()
        with open(label_file, 'r') as f:
            data = json.load(f)
        objects = [Object3dAstyx.from_label(obj) for obj in data['objects']]
        return objects

    def get_calib(self, idx):
        calib_file = self.root_split_path / 'calibration' / ('%s.json' % idx)
        assert calib_file.exists()
        with open(calib_file, 'r') as f:
            data = json.load(f)

        T_from_lidar_to_radar = np.array(data['sensors'][1]['calib_data']['T_to_ref_COS'])
        T_from_camera_to_radar = np.array(data['sensors'][2]['calib_data']['T_to_ref_COS'])
        K = np.array(data['sensors'][2]['calib_data']['K'])
        T_from_radar_to_lidar = inv_trans(T_from_lidar_to_radar)
        T_from_radar_to_camera = inv_trans(T_from_camera_to_radar)
        return {'T_from_radar_to_lidar': T_from_radar_to_lidar,
                'T_from_radar_to_camera': T_from_radar_to_camera,
                'T_from_lidar_to_radar': T_from_lidar_to_radar,
                'T_from_camera_to_radar': T_from_camera_to_radar,
                'K': K}

    def get_road_plane(self, idx):
        plane_file = self.root_split_path / 'planes' / ('%s.txt' % idx)
        if not plane_file.exists():
            return None

        with open(plane_file, 'r') as f:
            lines = f.readlines()
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinate
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane

    @staticmethod
    def get_fov_flag(pts_rect, img_shape, calib):
        """
        Args:
            pts_rect:
            img_shape:
            calib:

        Returns:

        """
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        return pts_valid_flag

    def get_infos(self, num_workers=4, has_label=True, count_inside_pts=True, sample_id_list=None):
        import concurrent.futures as futures

        def process_single_scene(sample_idx):
            print('%s sample_idx: %s' % (self.split, sample_idx))
            info = {}
            pc_info = {'num_features': 4, 'pc_idx': sample_idx}
            info['point_cloud'] = pc_info

            image_info = {'image_idx': sample_idx, 'image_shape': self.get_image_shape(sample_idx)}
            info['image'] = image_info
            calib = self.get_calib(sample_idx)
            info['calib'] = calib

            if has_label:
                obj_list = self.get_label(sample_idx)
                for obj in obj_list:
                    obj.from_radar_to_camera(calib)
                    obj.from_radar_to_image(calib)
                    ###############################
                    # print(f'\n%s:' % sample_idx)
                    #
                    # # print(obj.loc, obj.orient)
                    # # obj.from_lidar_to_radar(calib)
                    # # print(obj.loc, obj.orient)
                    #
                    # # print(obj.loc_camera, obj.rot_camera)
                    # # obj.from_lidar_to_camera(calib)
                    # # print(obj.loc_camera, obj.rot_camera)
                    #
                    # print(obj.box2d)
                    # obj.from_lidar_to_image(calib)
                    # print(obj.box2d)
                    ###############################

                annotations = {'name': np.array([obj.cls_type for obj in obj_list]),
                               'occluded': np.array([obj.occlusion for obj in obj_list]),
                               'alpha': np.array([-np.arctan2(obj.loc[1], obj.loc[0])
                                                  + obj.rot_camera for obj in obj_list]),
                               'bbox': np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0),
                               'dimensions': np.array([[obj.l, obj.h, obj.w] for obj in obj_list]),
                               'location': np.concatenate([obj.loc_camera.reshape(1, 3) for obj in obj_list], axis=0),
                               'rotation_y': np.array([obj.rot_camera for obj in obj_list]),
                               'score': np.array([obj.score for obj in obj_list]),
                               'difficulty': np.array([obj.level for obj in obj_list], np.int32),
                               'truncated': -np.ones(len(obj_list))}

                num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
                num_gt = len(annotations['name'])
                index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
                annotations['index'] = np.array(index, dtype=np.int32)
                if self.pc_type == 'radar':
                    gt_boxes = np.array([[*obj.loc, obj.w, obj.l, obj.h, obj.rot] for obj in obj_list])
                else:
                    for obj in obj_list:
                        obj.from_radar_to_lidar(calib)
                    gt_boxes = np.array([[*obj.loc_lidar, obj.w, obj.l, obj.h, obj.rot_lidar] for obj in obj_list])
                annotations['gt_boxes'] = gt_boxes

                info['annos'] = annotations

                if count_inside_pts:
                    points = self.get_pointcloud(sample_idx, self.pc_type)
                    calib = self.get_calib(sample_idx)
                    #pts_rect = calib.lidar_to_rect(points[:, 0:3])

                    #fov_flag = self.get_fov_flag(pts_rect, info['image']['image_shape'], calib)
                    #pts_fov = points[fov_flag]
                    corners = box_utils.boxes_to_corners_3d(gt_boxes)
                    num_points_in_gt = -np.ones(num_gt, dtype=np.int32)

                    for k in range(num_objects):
                        flag = box_utils.in_hull(points[:, 0:3], corners[k])
                        num_points_in_gt[k] = flag.sum()
                    annotations['num_points_in_gt'] = num_points_in_gt

            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)

    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        import torch

        # database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        # db_info_save_path = Path(self.root_path) / ('astyx_dbinfos_%s.pkl' % split)
        database_save_path = Path(self.root_path) / (('gt_database_%s' % self.pc_type) if split == 'train'
                                                     else ('gt_database_%s_%s' % (split, self.pc_type)))
        db_info_save_path = Path(self.root_path) / ('astyx_dbinfos_%s_%s.pkl' % (split, self.pc_type))

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        for k in range(len(infos)):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_idx = info['point_cloud']['pc_idx']
            points = self.get_pointcloud(sample_idx, self.pc_type)
            annos = info['annos']
            names = annos['name']
            difficulty = annos['difficulty']
            bbox = annos['bbox']
            gt_boxes = annos['gt_boxes']

            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0],
                               'difficulty': difficulty[i], 'bbox': bbox[i], 'score': annos['score'][i]}
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

    def generate_prediction_dicts(self, batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_3d': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            calib = batch_dict['calib'][batch_index]
            image_shape = batch_dict['image_shape'][batch_index]

            obj_list = [Object3dAstyx.from_prediction(box, label, score, self.pc_type) for box, label, score
                        in zip(pred_boxes,pred_labels,pred_scores)]
            for i, obj in enumerate(obj_list):
                if self.pc_type == 'radar':
                    obj.from_radar_to_camera(calib)
                    obj.from_radar_to_image(calib)
                    loc = obj.loc
                else:
                    obj.from_lidar_to_camera(calib)
                    obj.from_lidar_to_image(calib)
                    loc = obj.loc_lidar
                pred_dict['dimensions'][i, :] = np.array([obj.l, obj.h, obj.w])
                pred_dict['location'][i, :] = np.array(obj.loc_camera)
                pred_dict['rotation_y'][i] = np.array(obj.rot_camera)
                pred_dict['alpha'][i] = -np.arctan2(loc[1], loc[0]) + obj.rot_camera
                pred_dict['bbox'][i, :] = np.array(obj.box2d)

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_3d'] = pred_boxes

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]

            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                cur_det_file = output_path / ('%s.txt' % frame_id)
                with open(cur_det_file, 'w') as f:
                    bbox = single_pred_dict['bbox']
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
                                 bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                 dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                 loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
                                 single_pred_dict['score'][idx]), file=f)

        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.astyx_infos[0].keys():
            return None, {}

        from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval  # use the same kitti eval package

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.astyx_infos]
        ##################################################################
        # print(f'gt annos:')
        # print(len(eval_gt_annos))
        # for key, value in eval_gt_annos[0].items():
        #     print(key, type(value), value.shape)
        # print(eval_gt_annos[0])

        # print(f'det annos:')
        # print(len(eval_det_annos))
        # for key, value in eval_det_annos[0].items():
        #     print(key, type(value), value.shape)
        # print(eval_det_annos[0])

        # print(f'class names:')
        # print(class_names)
        ##################################################################
        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)

        return ap_result_str, ap_dict

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.astyx_infos) * self.total_epochs

        return len(self.astyx_infos)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.astyx_infos)

        info = copy.deepcopy(self.astyx_infos[index])

        ##################################################################
        # print(f'info annos:')
        # print(len(info['point_cloud']))
        # for key, value in info['point_cloud'].items():
        #     print(key)
        #################################################################
        sample_idx = info['point_cloud']['pc_idx']

        points = self.get_pointcloud(sample_idx, self.pc_type)
        calib = info['calib']

        img_shape = info['image']['image_shape']
        if self.dataset_cfg.FOV_POINTS_ONLY:
            pts_rect = calib.lidar_to_rect(points[:, 0:3])
            fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
            points = points[fov_flag]

        input_dict = {
            'points': points,
            'frame_id': sample_idx,
            'calib': calib,
        }
        if 'annos' in info:
            annos = info['annos']
            # ##################################################################
            # print(f'info annos:')
            # print(len(annos))
            # for key, value in annos.items():
            #     print(key, type(value), value.shape)
            # ##################################################################
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            gt_names = annos['name']
            # loc, dims, rots = annos['location'], annos['dimensions'], annos['orientation']
            # gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
            # gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)
            gt_boxes = annos['gt_boxes']

            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes
            })
            road_plane = self.get_road_plane(sample_idx)
            if road_plane is not None:
                input_dict['road_plane'] = road_plane

        # print('%%%%%%%%%%%%%%%%%%%%%%%% Before prepare_data ')
        # print(input_dict['points'].shape)
        # print(input_dict['points'][:10, :])
        data_dict = self.prepare_data(data_dict=input_dict)

        data_dict['image_shape'] = img_shape
        # print('%%%%%%%%%%%%%%%%%%%%%%%% After prepare_data ')
        # print(data_dict['points'].shape)
        # print(data_dict['points'][:10, :])
        return data_dict


def create_astyx_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    dataset = AstyxDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)
    train_split, val_split = 'train', 'val'

    # train_filename = save_path / ('astyx_infos_%s.pkl' % train_split)
    # val_filename = save_path / ('astyx_infos_%s.pkl' % val_split)
    # trainval_filename = save_path / 'astyx_infos_trainval.pkl'
    # test_filename = save_path / 'astyx_infos_test.pkl'
    train_filename = save_path / ('astyx_infos_%s_%s.pkl' % (train_split, dataset.pc_type))
    val_filename = save_path / ('astyx_infos_%s_%s.pkl' % (val_split, dataset.pc_type))
    trainval_filename = save_path / ('astyx_infos_trainval_%s.pkl' % dataset.pc_type)
    test_filename = save_path / ('astyx_infos_test_%s.pkl' % dataset.pc_type)

    print('---------------Start to generate data infos---------------')
    print(f'point cloud type: %s' % dataset.pc_type)

    dataset.set_split(train_split)
    astyx_infos_train = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(train_filename, 'wb') as f:
        pickle.dump(astyx_infos_train, f)
    print('Astyx info train file is saved to %s' % train_filename)

    dataset.set_split(val_split)
    astyx_infos_val = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(val_filename, 'wb') as f:
        pickle.dump(astyx_infos_val, f)
    print('Astyx info val file is saved to %s' % val_filename)

    with open(trainval_filename, 'wb') as f:
        pickle.dump(astyx_infos_train + astyx_infos_val, f)
    print('Astyx info trainval file is saved to %s' % trainval_filename)

    dataset.set_split('test')
    astyx_infos_test = dataset.get_infos(num_workers=workers, has_label=False, count_inside_pts=False)
    with open(test_filename, 'wb') as f:
        pickle.dump(astyx_infos_test, f)
    print('Astyx info test file is saved to %s' % test_filename)

    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)

    print('---------------Data preparation Done---------------')


if __name__ == '__main__':
    import sys
    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_astyx_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.full_load(open(sys.argv[2])))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        create_astyx_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Car', 'Pedestrian', 'Cyclist'],
            data_path=ROOT_DIR / 'data' / 'astyx',
            save_path=ROOT_DIR / 'data' / 'astyx'
        )
