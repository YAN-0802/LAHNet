import torch.nn as nn

class Make_Criterion(nn.Module):
    def __init__(self, train_args):
        super(Make_Criterion, self).__init__()
        self.aux_num = train_args['deep_supervise']
        self.batch_size = train_args['batch_size']
        self.bceloss = nn.BCEWithLogitsLoss().cuda()

    def forward(self, pred, label):
        
        if self.aux_num == 1:
            loss_bce = self.bceloss(pred, label) / self.batch_size
        else:
            loss_bce = sum([self.bceloss(pred_i, label) for pred_i in pred]) / self.batch_size
        return loss_bce