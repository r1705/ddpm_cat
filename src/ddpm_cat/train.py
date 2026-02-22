import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from model import DDPM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(
    T: int = 1000,
    epochs: int = 10000,
    batch_size: int = 128,
    beta_min: float = 1e-4,
    beta_max: float = 0.02,
):
    # CIFAR10の読み込み
    cifar10_train = datasets.CIFAR10(
        "./data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )

    # 猫画像の抽出
    cat_label_idx = cifar10_train.class_to_idx["cat"]
    cat_indices = [i for i, y in enumerate(cifar10_train.targets) if y == cat_label_idx]
    cat_train = Subset(cifar10_train, cat_indices)
    train_loader = DataLoader(cat_train, batch_size=batch_size, shuffle=True)

    model = DDPM(img_ch=3, ch_base=128, ch_multi=[1, 2, 2, 2], num_resblocks=2).to(
        device
    )
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    # DDPMのノイズスケジュール
    betas = torch.linspace(beta_min, beta_max, steps=T).to(device)
    alphas = 1.0 - betas
    bar_alphas = torch.cumprod(alphas, dim=0)

    # DDPMの学習
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss_sum = 0.0
        num_samples = 0
        for batch in tqdm(train_loader):
            x0, _ = batch
            x0 = x0.to(device)
            t = torch.randint(0, T, size=(x0.size(0),)).to(device)
            eps = torch.normal(0, 1, size=x0.size()).to(device)

            bar_alpha = bar_alphas[t]
            x = (
                torch.sqrt(bar_alpha)[:, None, None, None] * x0
                + torch.sqrt(1 - bar_alpha)[:, None, None, None] * eps
            )

            eps_pred = model(x, t)
            loss = mse_loss(eps_pred, eps)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss_sum += loss.item() * x0.size(0)
            num_samples += x0.size(0)

        mean_loss = epoch_loss_sum / num_samples
        logger.info(f"epoch={epoch}, loss={mean_loss:.5f}")

    # モデルの保存
    output_path = "./ddpm.pth"
    torch.save(model.state_dict(), output_path)


if __name__ == "__main__":
    train()
