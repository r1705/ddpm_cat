import logging
import os

import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

from model import DDPM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def q_sample(x0, t_idx, bar_alphas, eps=None):
    if eps is None:
        eps = torch.randn_like(x0)
    bar_alpha = bar_alphas[t_idx]
    return torch.sqrt(bar_alpha) * x0 + torch.sqrt(1 - bar_alpha) * eps


def p_sample_loop(model, x_t, t_start, alphas, bar_alphas, betas, zs=None):
    x = x_t
    with torch.inference_mode():
        for t in tqdm(reversed(range(t_start + 1)), total=t_start + 1):
            t_batch = torch.full((x.size(0),), t, device=device, dtype=torch.long)
            eps = model(x, t_batch)

            alpha = alphas[t]
            bar_alpha = bar_alphas[t]
            beta = betas[t]
            z = (
                zs[t]
                if (t > 0 and zs is not None)
                else (torch.randn_like(x) if t > 0 else torch.zeros_like(x))
            )

            x = (x - ((1 - alpha) / torch.sqrt(1 - bar_alpha)) * eps) / torch.sqrt(
                alpha
            ) + torch.sqrt(beta) * z
    return x


def interpolate(
    model_path="./ddpm.pth",
    T=1000,
    t_interp=200,
    n_lambdas=9,
    beta_min=1e-4,
    beta_max=0.02,
    idx_a=0,
    idx_b=2,
    out_path="interp.png",
):
    if not (1 <= t_interp <= T):
        raise ValueError(f"t_interp must be in [1, {T}], but got {t_interp}")
    t_idx = t_interp - 1

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    cifar10_train = datasets.CIFAR10(
        "./data", train=True, download=True, transform=transform
    )
    cat_label_idx = cifar10_train.class_to_idx["cat"]
    cat_indices = [i for i, y in enumerate(cifar10_train.targets) if y == cat_label_idx]
    cat_train = Subset(cifar10_train, cat_indices)

    x0_a, _ = cat_train[idx_a]
    x0_b, _ = cat_train[idx_b]
    x0_a = x0_a.unsqueeze(0).to(device)
    x0_b = x0_b.unsqueeze(0).to(device)

    model = DDPM(img_ch=3, ch_base=128, ch_multi=[1, 2, 2, 2], num_resblocks=2).to(
        device
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    betas = torch.linspace(beta_min, beta_max, steps=T, device=device)
    alphas = 1 - betas
    bar_alphas = torch.cumprod(alphas, dim=0)

    # lambda の違いだけを見るため、前向きノイズは固定する
    eps_a = torch.randn_like(x0_a)
    eps_b = torch.randn_like(x0_b)
    x_t_a = q_sample(x0_a, t_idx, bar_alphas, eps=eps_a)
    x_t_b = q_sample(x0_b, t_idx, bar_alphas, eps=eps_b)

    # 論文図8に合わせ、逆拡散ノイズも lambda 間で固定する
    zs = [None] * T
    for t in range(1, T):
        zs[t] = torch.randn_like(x0_a)

    lambdas = torch.linspace(0.0, 1.0, steps=n_lambdas, device=device)
    outputs = []
    for n, lam in enumerate(lambdas):
        logger.info(f"n={n + 1}/{n_lambdas}, lam={lam}")
        x_t_mix = (1 - lam) * x_t_a + lam * x_t_b
        x0_hat = p_sample_loop(model, x_t_mix, t_idx, alphas, bar_alphas, betas, zs=zs)
        outputs.append(x0_hat)

    imgs = torch.cat(outputs, dim=0)
    imgs = (imgs.clamp(-1, 1) + 1) / 2
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    save_image(imgs, out_path, nrow=n_lambdas)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    interpolate()
