from torch import nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss
from image_segmentation_project.config import Config


class SegmentatiomModel(nn.Module):

    def __init__(self):
        super(SegmentatiomModel, self).__init__()

        self.arc = smp.Unet(
            encoder_name=Config.ENCODER,
            encoder_weights=Config.WEIGHS,
            in_channels=3,
            classes=1,
            activation=None
        )

    def forward(self, images, masks=None):
        logits = self.arc(images)

        if masks != None:
            loss1 = DiceLoss(mode='binary')(logits, masks)
            loss2 = nn.BCEWithLogitsLoss()(logits, masks)
            return logits, loss1 + loss2

        return logits
