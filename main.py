import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from image_segmentation_project.dataset import SegmentationDataset
from image_segmentation_project.model import SegmentatiomModel
from image_segmentation_project.utils.augmentation import *
from image_segmentation_project.trainer import *
from image_segmentation_project.utils.helper import show_image


def main() -> None:
    df = pd.read_csv(Config.CSV_FILE)
    df.head()

    train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)

    trainset = SegmentationDataset(train_df, get_train_augs())
    validset = SegmentationDataset(valid_df, get_valid_augs())

    trainloader = DataLoader(trainset, batch_size=Config.BATCH_SIZE, shuffle=True)
    validloader = DataLoader(validset, batch_size=Config.BATCH_SIZE)

    model = SegmentatiomModel()
    model.to(Config.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR)
    best_valid_loss = np.Inf
    

    for i in range(Config.EPOCHS):
        train_loss = train_fn(trainloader, model, optimizer)
        valid_loss = eval_fn(validloader, model)

        if valid_loss < best_valid_loss:
            torch.save(model.state_dict(), 'best_model.pt')
            print("SAVED-MODEL")
            best_valid_loss = valid_loss

        print(f"Epoch : {i + 1} Train_loss : {train_loss} Valid_loss: {valid_loss}")


if __name__ == "__main__":
    main()
