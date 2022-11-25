import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


class StackedGCN(torch.nn.Module):
    """
    Multi-layer GCN model.
    """
    def __init__(self, args, input_channels, output_channels):
        """
        args: 构建模型所需参数
        input_channels: 输入层
        output_channels: 输出层
        args: Arguments object.
        input_channels: Number of features.
        output_channels: Number of target features.
        """
        super(StackedGCN, self).__init__()
        self.layers = None
        self.args = args
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.setup_layers()

    def setup_layers(self):
        """
        根据传入参数设置GCN层
        Creating the layers based on the args.
        """
        self.layers = []
        '''
        layers == [input_channels, layers, output_channels]
        构建了input_channels个输入，output_channels个输出，中间层结构为[layers]，默认为[16,16,16]的GCN模型架构
        '''
        self.args.layers = [self.input_channels] + self.args.layers + [self.output_channels]

        if self.args.debug:
            print(f'Before setup:layers={self.args.layers}')

        # 创建对应数量的GCN卷积层并存入list
        # 除输出层外都是卷积层
        for i, _ in enumerate(self.args.layers[:-1]):
            self.layers.append(GCNConv(self.args.layers[i], self.args.layers[i + 1]))
        # 将GCN卷积层实际加入模型
        self.layers = ListModule(*self.layers)

        if self.args.debug:
            print(f'After setup:layers={self.layers}')

    def forward(self, edges, features):
        """
        前向传播
        Making a forward pass.
        :param edges: Edge list LongTensor.
        :param features: Feature matrix input FLoatTensor.
        :return predictions: Prediction matrix output FLoatTensor.
        """
        # 逐层卷积，激活函数为ReLU
        for i, _ in enumerate(self.args.layers[:-2]):
            features = F.relu(self.layers[i](features, edges))
            # 除输入层外，其余卷积层使用dropout
            if i > 1:
                features = F.dropout(features, p=self.args.dropout, training=self.training)
        # 最后一层卷积层使用线性激活函数
        features = self.layers[i + 1](features, edges)
        # 使用Softmax函数预测，不在最后一层使用Softmax是为了提高计算精度
        predictions = F.log_softmax(features, dim=1)
        return predictions


class ListModule(torch.nn.Module):
    """
    Abstract list layer class.

    """

    def __init__(self, *args):
        """
        Module initializing.
        """
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        """
        Getting the indexed layer.
        """
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        """
        Iterating on the layers.
        """
        return iter(self._modules.values())

    def __len__(self):
        """
        Number of layers.
        """
        return len(self._modules)
