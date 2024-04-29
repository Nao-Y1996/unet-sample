import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Conv2d, MaxPool2d, ConvTranspose2d


class DoubleConv(nn.Module):
    def __init__(self, in_channel=3, out_channel=64, *args, **kwargs, ):
        super().__init__(*args, **kwargs)
        self.conv1 = Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        return x


class UpSample(nn.Module):
    """
    UpSampling Layer
    空間方向に2倍に拡大する.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # inputに対して空間サイズが2倍になるようなパラメータ（一般的な例）
        kernel_size = 4
        stride = 2
        padding = 1
        output_padding = 0
        dilation = 1

        self.conv_transpose = ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                              kernel_size=kernel_size, stride=stride,
                                              padding=padding, output_padding=output_padding, dilation=dilation)

    def forward(self, x):
        x = self.conv_transpose(x)
        return x


class DownSample(nn.Module):
    """
    DownSampling Layer
    空間方向に1/2に縮小する.
    """

    def __init__(self):
        super().__init__()
        self.pool = MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.pool(x)
        return x


class BottleNeck(nn.Module):
    def __init__(self, in_channel, out_channel, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv = DoubleConv(in_channel=in_channel, out_channel=out_channel)

    def forward(self, x):
        return self.conv(x)


class Unet(nn.Module):
    def __init__(self, num_class: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_class = num_class

        self.down_conv1 = DoubleConv(in_channel=3, out_channel=64)
        self.down1 = DownSample()

        self.down_conv2 = DoubleConv(in_channel=64, out_channel=128)
        self.down2 = DownSample()

        self.down_conv3 = DoubleConv(in_channel=128, out_channel=256)
        self.down3 = DownSample()

        self.down_conv4 = DoubleConv(in_channel=256, out_channel=512)
        self.down4 = DownSample()

        self.bottleneck = BottleNeck(in_channel=512, out_channel=1024)

        self.up4 = UpSample(in_channels=1024, out_channels=512)
        self.up_conv4 = DoubleConv(in_channel=1024, out_channel=512)

        self.up3 = UpSample(in_channels=512, out_channels=256)
        self.up_conv3 = DoubleConv(in_channel=512, out_channel=256)

        self.up2 = UpSample(in_channels=256, out_channels=128)
        self.up_conv2 = DoubleConv(in_channel=256, out_channel=128)

        self.up1 = UpSample(in_channels=128, out_channels=64)
        self.up_conv1 = DoubleConv(in_channel=128, out_channel=64)

        self.classifier = Conv2d(in_channels=64, out_channels=num_class, kernel_size=3, stride=1, padding=1)

    def forward(self, x):

        # print("--------エンコーダー-------")
        x = self.down_conv1(x)
        _x1 = x.detach().clone()
        x = self.down1(x)
        # print(x.shape)

        x = self.down_conv2(x)
        _x2 = x.detach().clone()
        x = self.down2(x)
        # print(x.shape)

        x = self.down_conv3(x)
        _x3 = x.detach().clone()
        x = self.down3(x)
        # print(x.shape)

        x = self.down_conv4(x)
        _x4 = x.detach().clone()
        x = self.down4(x)
        # print(x.shape)
        # print("-------------------------")

        # ボトルネック層
        # print("--------ボトルネック-------")
        x = self.bottleneck(x)
        # print(x.shape)
        # print("-------------------------")

        # print("--------デコーダー-------")
        # アップサンプリング1
        x = self.up4(x)
        x = torch.cat([_x4, x], dim=1)  # skip connection チャンネル方向に連結
        x = self.up_conv4(x)
        # print(x.shape)

        x = self.up3(x)
        x = torch.cat([_x3, x], dim=1)  # skip connection
        x = self.up_conv3(x)
        # print(x.shape)

        x = self.up2(x)
        x = torch.cat([_x2, x], dim=1)  # skip connection
        x = self.up_conv2(x)
        # print(x.shape)

        x = self.up1(x)
        x = torch.cat([_x1, x], dim=1)  # skip connection
        x = self.up_conv1(x)
        # print(x.shape)
        # print("-------------------------")

        x = self.classifier(x)
        # print(x.shape)

        # TODO
        #  - [x] UpSamplingの実装 OK!
        #  - [x] skipConnectionの実装
        #  - [ ] データセットの準備
        #  - [ ] 学習の実装
        #  - [ ] 学習
        #  - [ ] 可視化
        #  - [ ] ファイルごとの評価の実装（評価はIoU、

        return x

    def save_checkpoint(self, save_dir: str, epoch: int) -> str:
        """
        Save model to checkpoint file.
        To load checkpoint file, use load_checkpoint method.

        Parameters
        ----------
        save_dir: str
            directory path to save checkpoint file
        epoch: int
            epoch number

        Returns
        -------
        save_path: str
            saved checkpoint file path
        """
        os.makedirs(save_dir, exist_ok=True)
        prefix = "model"
        file_name = os.path.join(save_dir, f"{prefix}_epoch{epoch}.pth")
        save_path = os.path.join(save_dir, file_name)
        data = {"model_state_dict": self.state_dict(),
                "num_class": self.num_class}
        torch.save(data, save_path)
        return save_path

    @classmethod
    def load_checkpoint(cls, ckp_path: str, device: torch.device) -> 'Unet':
        """
        load checkpoint file and return model.
        model is loaded to device.
        checkpoint file must contain 3 keys such as "model_state_dict", "encoder_name", "c_out".
        To save checkpoint file, use save_checkpoint method.

        Parameters
        ----------
        ckp_path: str
            checkpoint file path
        device: torch.device
            device to load model

        Returns
        -------
        _model: UNet
            loaded model
        """
        checkpoint = torch.load(ckp_path, map_location=device)
        num_class = int(checkpoint["num_class"])
        _model = cls(num_class=num_class)
        _model.load_state_dict(checkpoint["model_state_dict"])
        _model = _model.to(device, dtype=torch.float32)
        return _model


if __name__ == '__main__':

    num_class = 5
    batch_size = 4

    unet = Unet(num_class=num_class)
    input = torch.randn(batch_size, 3, 32, 32)
    output = unet(input)
    assert output.shape == (batch_size, num_class, 32, 32)
