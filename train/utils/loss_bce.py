import torch
from torch import nn
class DiceLoss(nn.Module):
    def __init__(self, smooth=1., dims=(-2, -1)):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.dims = dims

    def forward(self, x, y):
        tp = (x * y).sum(self.dims)
        fp = (x * (1 - y)).sum(self.dims)
        fn = ((1 - x) * y).sum(self.dims)
        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        dc = dc.mean()
        return 1 - dc
class MIoULoss(nn.Module):
    def __init__(self, smooth=1., dims=(-2, -1)):
        super(MIoULoss, self).__init__()
        self.smooth = smooth
        self.dims = dims

    def forward(self, x, y):
        # x: 预测概率图，shape (batch, class, height, width)，需已通过softmax/sigmoid
        # y: 真实标签的one-hot编码，shape 与 x 相同
        tp = (x * y).sum(self.dims)          # 真正例，shape (batch, class)
        fp = (x * (1 - y)).sum(self.dims)    # 假正例
        fn = ((1 - x) * y).sum(self.dims)    # 假负例
        
        # 计算每个类别的 IoU，避免除零
        iou = (tp + self.smooth) / (tp + fp + fn + self.smooth)  # shape (batch, class)
        miou = iou.mean(dim=-1)  # 对类别维度取平均，shape (batch,)
        miou = miou.mean()       # 对批次取平均，得到标量
        return 1 - miou
class bce_dice(nn.Module):
    def __init__(self, args):
        super(bce_dice, self).__init__()
        # 不再使用 BCEWithLogitsLoss，而是手动计算
        self.dice_fn = DiceLoss()   # 保持原样
        self.miou_fn = MIoULoss()
        self.args = args
        self.eps = 1e-8   # 防止除零

    def forward(self, y_pred, y_true):
        C = y_pred.size(0)
        dice_sum = 0.0
        
        for c in range(C):
            pred_c = y_pred[c, :, :]   # logits
            true_c = y_true[c, :, :]   # 0/1

            prob_c = torch.sigmoid(pred_c)
            dice_c = self.dice_fn(prob_c, true_c)

            dice_sum += dice_c

        dice_avg = dice_sum / C
        
        return dice_avg