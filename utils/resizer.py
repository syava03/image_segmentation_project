from image_segmentation_project.config import Config
import cv2
import numpy as np
import torch


def img_resize(img):
    dim = (Config.IMAGE_SIZE, Config.IMAGE_SIZE)
    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    image = np.transpose(resized, (2, 0, 1)).astype(np.float32)
    image = torch.Tensor(image) / 255.0
    return image
