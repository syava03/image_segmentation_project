class Config:
    CSV_FILE = 'D:/pythonProject/OOP/image_segmentation_project/Human-Segmentation-Dataset-master/train.csv'
    DATA_DIR = '/image_segmentation_project/'

    DEVICE = 'cpu'

    EPOCHS = 25
    LR = 0.003
    IMAGE_SIZE = 320
    BATCH_SIZE = 16

    ENCODER = 'timm-efficientnet-b0'
    WEIGHS = 'imagenet'
