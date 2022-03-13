import torch
import torch.nn as nn
from torchvision import models

class Feat2d(nn.Module):
    def __init__(self, touch_emb_dim, do_bn=True):
        """
            @touch_emb_dim: dimension of the embedding dim for the VGGNet
        """
        super(Feat2d, self).__init__()
        self.do_bn = do_bn
        if self.do_bn:
            x = models.vgg11_bn(pretrained=False, progress=False)
            feats = x.features
            feats = list(feats)[1:14]
        else:
            x = models.vgg11(pretrained=False, progress=False)
            feats = x.features
            feats = list(feats)[1:10]
        # since I will be passing depth images
        feats.insert(0, nn.Conv2d(1, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1)))
        self.model = nn.Sequential(*feats)

        # now the above produces a feature channel of size 512 we want less
        self.feats_dim_reducer = nn.Conv2d(256, touch_emb_dim, kernel_size=(4,4), stride=(1,1))

        nn.init.kaiming_normal_(self.feats_dim_reducer.weight, mode='fan_out', nonlinearity='relu')
        if self.feats_dim_reducer.bias is not None:
            nn.init.constant_(self.feats_dim_reducer.bias, 0.0)

    def forward(self, x):
        out = self.model(x)
        out = self.feats_dim_reducer(out)
        # x.shape[0] will give me the batch size
        return out.view(x.shape[0], -1)

if __name__ == '__main__':
    # create an instance of the model
    net = Feat2d(32, do_bn=False)
    x = torch.randn(10, 1, 16, 16)
    out = net(x)
    from IPython import embed; embed()
