import torch
#根据类别分别生成anchors并且每个anchor都被分配了一个分类目标的one-hot向量、一个7-vector的框回归目标和一个方向分类目标的one-hot向量

class AnchorGenerator(object):
    def __init__(self, anchor_range, anchor_generator_config):
        super().__init__()
        self.anchor_generator_cfg = anchor_generator_config
        self.anchor_range = anchor_range # [0, -39.68, -3, 69.12, 39.68, 1]
        # car:[[3.9, 1.6, 1.56]] ，Pedestrian:[[0.8, 0.6, 1.73]]，Cyclist:[[1.76, 0.6, 1.73]]
        self.anchor_sizes = [config['anchor_sizes'] for config in anchor_generator_config]
        # [0, 1.57],[0, 1.57],[0, 1.57]  每个类别的anchor都有两个方向角为0度和90度
        self.anchor_rotations = [config['anchor_rotations'] for config in anchor_generator_config]
        # [-1.78],[-0.6],[-0.6] 以 z = − 1.78 m  z = − 0.6 m  z = − 0.6 m为中心
        self.anchor_heights = [config['anchor_bottom_heights'] for config in anchor_generator_config]
        # False,False,False
        self.align_center = [config.get('align_center', False) for config in anchor_generator_config]

        assert len(self.anchor_sizes) == len(self.anchor_rotations) == len(self.anchor_heights)
        self.num_of_anchor_sets = len(self.anchor_sizes) # 3

    def generate_anchors(self, grid_sizes):
        #grid_sizes [array([128, 128]), array([128, 128]), array([128, 128])]
        assert len(grid_sizes) == self.num_of_anchor_sets
        # print(grid_sizes)#########
        # 1.初始化
        all_anchors = []
        num_anchors_per_location = []
        # 2.三个类别的anchors逐个生成
        for grid_size, anchor_size, anchor_rotation, anchor_height, align_center in zip(
                grid_sizes, self.anchor_sizes, self.anchor_rotations, self.anchor_heights, self.align_center):
            # 2 = 2x1x1 --> 每个位置产生2个anchor
            num_anchors_per_location.append(len(anchor_rotation) * len(anchor_size) * len(anchor_height))
            if align_center:
                x_stride = (self.anchor_range[3] - self.anchor_range[0]) / grid_size[0]
                y_stride = (self.anchor_range[4] - self.anchor_range[1]) / grid_size[1]
                x_offset, y_offset = x_stride / 2, y_stride / 2
            else:
                # 2.1计算每个网格的实际大小
                x_stride = (self.anchor_range[3] - self.anchor_range[0]) / (grid_size[0] - 1) # (69.12 - 0) / (216 - 1) = 0.321488
                y_stride = (self.anchor_range[4] - self.anchor_range[1]) / (grid_size[1] - 1) # (39.68 - (-39.68)) / (248 -1 ) = 0.321295
                x_offset, y_offset = 0, 0
            # 2.2 生成单个维度x_shifts，y_shifts和z_shifts
            # 以x_stride为step，在self.anchor_range[0] + x_offset和self.anchor_range[3] + 1e-5，产生x坐标 --> 216个点 [0, 69.12]
            x_shifts = torch.arange(
                self.anchor_range[0] + x_offset, self.anchor_range[3] + 1e-5, step=x_stride, dtype=torch.float32,
            ).cuda()
            # 248个点 [-39.68, 39.68]
            y_shifts = torch.arange(
                self.anchor_range[1] + y_offset, self.anchor_range[4] + 1e-5, step=y_stride, dtype=torch.float32,
            ).cuda()
            z_shifts = x_shifts.new_tensor(anchor_height)# [-1.78]车类别的高度  其他为[-0.6]

            num_anchor_size, num_anchor_rotation = anchor_size.__len__(), anchor_rotation.__len__() # 1,2
            anchor_rotation = x_shifts.new_tensor(anchor_rotation) # tensor([0.0000, 1.5700])
            anchor_size = x_shifts.new_tensor(anchor_size) # tensor([[3.9000, 1.6000, 1.5600]])
            # # 2.3 调用meshgrid生成网格坐标
            # torch.meshgrid()的功能是生成网格，可以用于生成坐标
            # 维度：[第一个输入张量的元素个数，第二个输入张量的元素个数,第三个输入张量的元素个数]
            # 第一个输出张量填充第一个输入张量中的元素，第二个输出张量填充第二个输入张量中的元素,第三个输出张量填充第三个输入张量中的元素
            x_shifts, y_shifts, z_shifts = torch.meshgrid([
                x_shifts, y_shifts, z_shifts
            ])   # [x_grid, y_grid, z_grid] torch.Size([216, 248, 1]) torch.Size([216, 248, 1]) torch.Size([216, 248, 1])
            # meshgrid可以理解为在原来的维度上进行扩展,例如:
            # x原来为（216，）-->（216，1, 1）--> (216,248,1) 每个元素为对应格子x坐标
            # y原来为（248，）--> (1，248，1）--> (216,248,1) 每个元素为对应格子y坐标
            # z原来为 (1,) --> (1,1,1) --> (216,248,1) 每个元素为对应格子z坐标

            # 2.4.anchor各个维度堆叠组合，生成最终anchor(1,248,216,1,2,7）
            # 2.4.1.堆叠anchor的位置
            # [x, y, z, 3]  # torch.Size([216, 248, 1, 3])
            """
            anchors tensor([[[[  0.0000, -39.6800,  -1.7800]],

                           [[  0.0000, -39.3587,  -1.7800]],

                           [[  0.0000, -39.0374,  -1.7800]],

                           ...,
            """
            anchors = torch.stack((x_shifts, y_shifts, z_shifts), dim=-1)  # [x, y, z, 3] # torch.Size([216, 248, 1, 3]) x,y,z坐标
            # 2.4.2.将anchor的位置和大小进行组合，编程套路为将anchor扩展并复制为相同维度（除了最后一维），然后进行组合
            anchors = anchors[:, :, :, None, :].repeat(1, 1, 1, anchor_size.shape[0], 1) # (216,248,1,1,3)-->(216,248,1,1,3) 增加了anchor_size的维度1，对应[[3.9, 1.6, 1.56]]
            anchor_size = anchor_size.view(1, 1, 1, -1, 3).repeat([*anchors.shape[0:3], 1, 1]) # (1,1,1,1,3)-->(216,248,1,1,3) 对应anchor_size [[3.9, 1.6, 1.56]]
            anchors = torch.cat((anchors, anchor_size), dim=-1) # anchors的位置+大小 --> (216,248,1,1,6)
            # 2.4.3.将anchor的位置和大小和旋转角进行组合
            anchors = anchors[:, :, :, :, None, :].repeat(1, 1, 1, 1, num_anchor_rotation, 1) # (216, 248, 1, 1，1, 6)-->(216, 248, 1, 1, 2, 6) 增加了anchor_rotation的维度2，对应[0, 1.57]
            anchor_rotation = anchor_rotation.view(1, 1, 1, 1, -1, 1).repeat([*anchors.shape[0:3], num_anchor_size, 1, 1]) #(1,1,1,1,2,1)--> (216, 248, 1, 1, 2, 1)
            anchors = torch.cat((anchors, anchor_rotation), dim=-1)  # anchors的位置+大小+旋转方向 --> [x, y, z, num_size, num_rot, 7] --> (216,248,1,1,2,7)
            # 2.5 调整anchor的维度 anchors.shape:torch.Size([128, 128, 1, 1, 2, 7])
            # print(anchors.shape) #torch.Size([128, 128, 1, 1, 2, 7])
            anchors = anchors.permute(2, 1, 0, 3, 4, 5).contiguous() #（1，248，216，1，2，7）--> [x, y, z, dx, dy, dz, rot] 把Z轴移到第0个维度，X,Y的维度在第1，2个维度
            #anchors = anchors.view(-1, anchors.shape[-1])
            anchors[..., 2] += anchors[..., 5] / 2  # shift to box centers # z轴方向-->shift to box centers
            all_anchors.append(anchors)
        return all_anchors, num_anchors_per_location # list:3 all_anchors:[(1，248，216，1，2，7），(1，248，216，1，2，7），(1，248，216，1，2，7）],num_anchors_per_location: [2,2,2]
        #车、行人、自行车的anchors,对应每个anchors的x、y、z坐标和anchors框的长宽高、还有角度，每个位置对应两个anchors(两个角度)

if __name__ == '__main__':
    from easydict import EasyDict
    config = [
        EasyDict({
            'anchor_sizes': [[2.1, 4.7, 1.7], [0.86, 0.91, 1.73], [0.84, 1.78, 1.78]],
            'anchor_rotations': [0, 1.57],
            'anchor_heights': [0, 0.5]
        })
    ]

    A = AnchorGenerator(
        anchor_range=[-75.2, -75.2, -2, 75.2, 75.2, 4],
        anchor_generator_config=config
    )
    import pdb
    pdb.set_trace()
    A.generate_anchors([[188, 188]])
