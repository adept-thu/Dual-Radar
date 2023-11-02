import argparse
import glob
from pathlib import Path

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
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
        #print(dataset_cfg.DATA_AUGMENTOR)
        self.use_data_type = dataset_cfg.DATA_AUGMENTOR.get('USE_DATA_TYPE', None)
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            print(self.use_data_type)
            if self.use_data_type == 'lidar':
               points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 6)
               #points = points[:,:4]
            else:
               points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 5)
               #points = points[:,:4] 
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='/ai/volume/Dual-Radar-master/tools/cfgs/dual_radar_models/pointpillar_lidar.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='/ai/volume/Dual-Radar-master/data/dual_radar/lidar/training/velodyne/000000.bin',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default='/ai/volume/Dual-Radar-master/models/pointpillars_liadr_80.pth', help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
    logger.info('Demo done.')


if __name__ == '__main__':
    main()
