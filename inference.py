from image_segmentation_project.utils.resizer import *
from image_segmentation_project.model import SegmentatiomModel
from image_segmentation_project.utils.helper import show_image


def inference_image(img: str) -> None:
    img = cv2.imread(img, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = img_resize(img)
    model = SegmentatiomModel()
    model.load_state_dict(torch.load('D:/pythonProject/OOP/image_segmentation_project/best_model.pt',
                                     map_location='cpu'))
    logits_mask = model(image.to(Config.DEVICE).unsqueeze(0))  # (C, H, W) -> (1, C, H, W)
    pred_mask = torch.sigmoid(logits_mask)
    pred_mask = (pred_mask > 0.5) * 1.0

    show_image(image, pred_mask.detach().cpu().squeeze(0))

inference_image('D:/pythonProject/OOP/image_segmentation_project/Human-Segmentation-Dataset-master/Training_Images/7.jpg')
