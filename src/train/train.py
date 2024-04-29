import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from const.const_values import DEVICE, ROOT
from model.unet import Unet
from dataset.coco.TorchCocoDataset import CustomCocoDataset


def root_path(path_from_root: str) -> str:
    return os.path.join(ROOT, path_from_root)


if __name__ == '__main__':

    transforms4data = transforms.Compose([
        transforms.Resize((512, 512)),
    ])
    transforms4label = transforms.Compose([
        transforms.Resize((512, 512)),
    ])

    dataset = CustomCocoDataset(
        image_dir=root_path("datasets/coco-2017/validation/data"),
        annotation_file=root_path("datasets/coco-2017/validation/labels.json"),
        data_transforms=transforms4data,
        label_transforms=transforms4label,
        include_categories=["cat", "dog"]
    )

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset=dataset,
                                              lengths=[0.9, 0.1],
                                              generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=5)
    val_loader = DataLoader(val_dataset, batch_size=5)

    # tensorboardログ
    log_dir = root_path("logs")
    writer = SummaryWriter(os.path.join(log_dir, "log"))
    log_loss = {}

    model = Unet(num_class=2)
    model = model.to(DEVICE, dtype=torch.float32)
    criterion = nn.MSELoss()  # TODO 適切な損失関数を設定する
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)  # TODO 適切なオプティマイザを設定する
    scheduler = MultiStepLR(optimizer, milestones=[7, 14, 21, 28, 32, 40], gamma=0.75)  # TODO 適切なスケジューラを設定する

    epochs = 1

    p = tqdm(total=epochs, desc=f"epoch")
    for epoch in range(1, epochs + 1):
        train_loss = 0.0
        for images, target in train_loader:
            # prepare data
            masks = target["masks"]
            images = images.to(DEVICE, dtype=torch.float32)
            masks = masks.to(DEVICE, dtype=torch.float32)

            # forward
            output = model(images)

            # loss
            loss = criterion(output, masks)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # logging
        train_loss /= len(train_loader)
        log_loss["train"] = train_loss

        # scheduler
        scheduler.step()

        # validation
        with torch.no_grad():
            val_loss = 0.0
            for images, target in val_loader:
                masks = target["masks"]
                images = images.to(DEVICE, dtype=torch.float32)
                masks = masks.to(DEVICE, dtype=torch.float32)

                # forward
                output = model(images)
                # loss
                loss = criterion(output, masks)

                val_loss += loss.item()

            # logging
            val_loss /= len(val_loader)
            log_loss["val"] = val_loss

        if epoch % 10 == 0:
            # save checkpoint
            save_dir = root_path("checkpoints")
            model.save_checkpoint(save_dir, epoch)

        # tensorboard logging
        writer.add_scalar("loss", log_loss, epoch)

        p.update()
    writer.close()
