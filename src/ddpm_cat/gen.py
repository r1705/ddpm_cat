import torch
from model import DDPM
from torchvision.utils import save_image
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gen(
    model_path="./ddpm.pth",
    n_img: int = 8,
    T: int = 1000,
    beta_min: float = 1e-4,
    beta_max: float = 0.02,
):
    size = (n_img, 3, 32, 32)

    # モデルの読み込み
    model = DDPM(img_ch=3, ch_base=128, ch_multi=[1, 2, 2, 2], num_resblocks=2).to(
        device
    )
    model.load_state_dict(torch.load(model_path, map_location=device))

    # DDPMのノイズスケジュール
    betas = torch.linspace(beta_min, beta_max, steps=T, device=device)
    alphas = 1 - betas
    bar_alphas = torch.cumprod(alphas, dim=0)

    # 初期値
    x = torch.randn(size, device=device)

    # 画像の生成
    model.eval()
    with torch.inference_mode():
        for t in tqdm(reversed(range(T)), total=T):
            t_batch = torch.full((x.size(0),), t, device=device, dtype=torch.long)

            eps = model(x, t_batch)

            alpha = alphas[t]
            bar_alpha = bar_alphas[t]
            beta = betas[t]

            z = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
            x = (
                1
                / torch.sqrt(alpha)
                * (x - (1 - alpha) / torch.sqrt(1 - bar_alpha) * eps)
                + torch.sqrt(beta) * z
            )

    # 画像の保存
    save_image(((x.clamp(-1, 1) + 1) / 2), "sample.png")


if __name__ == "__main__":
    gen()
