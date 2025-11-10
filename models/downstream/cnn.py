# models/downstream/cnn.py
import torch
import torch.nn as nn
from einops import rearrange


class CNN1DEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 24,
        widths: tuple[int, int, int] = (16, 32, 64),
        kernels: tuple[int, int, int] = (11, 7, 5),
        p_drop: float = 0.1,
        pool_after: tuple[int, ...] = (0, 1),
        preserve_time: bool = False,
        out_time_len: int = 8,
    ):
        super().__init__()
        assert len(widths) == len(kernels), "widths and kernels must match"

        layers: list[nn.Module] = []
        c_in = in_channels

        for idx, (c_out, k) in enumerate(zip(widths, kernels)):
            pad = k // 2
            layers += [
                nn.Conv1d(c_in, c_out, kernel_size=k, padding=pad, bias=False),
                nn.BatchNorm1d(c_out),
                nn.ReLU(inplace=True),
            ]
            if idx in pool_after:
                layers.append(nn.MaxPool1d(kernel_size=2))
            c_in = c_out

        layers.append(nn.Dropout(p_drop))
        self.feat = nn.Sequential(*layers)
        self.preserve_time = bool(preserve_time)
        self.out_channels = widths[-1]
        if self.preserve_time:
            if out_time_len <= 0:
                raise ValueError("out_time_len must be positive when preserve_time is True.")
            self.out_time = int(out_time_len)
        else:
            self.out_time = 1
        self.out_dim = self.out_channels * self.out_time

    @staticmethod
    def _segment_sizes(length: int, segments: int) -> list[int]:
        if segments > length:
            raise ValueError(
                f"Cannot preserve {segments} temporal tokens when sequence length is only {length}."
            )
        base = length // segments
        remainder = length % segments
        sizes = [base + 1] * remainder + [base] * (segments - remainder)
        return sizes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*, C, T)
        h = self.feat(x)  # (B*, F, T')
        if self.preserve_time:
            target = self.out_time
            cur = h.shape[-1]
            if target != cur:
                sizes = self._segment_sizes(cur, target)
                chunks = torch.split(h, sizes, dim=-1)
                h = torch.stack([chunk.mean(dim=-1) for chunk in chunks], dim=-1)
            else:
                h = h
        else:
            h = h.mean(dim=-1, keepdim=True)
        if not self.preserve_time:
            h = h.squeeze(-1)  # (B*, F)
        return h


class CNN(nn.Module):
    def __init__(
        self,
        n_outputs: int,
        in_channels_eeg: int = 24,
        input_windows: int = 5,
        time_len: int = 300,
        width: tuple[int, int, int] = (16, 32, 64),
        kernels: tuple[int, int, int] = (11, 7, 5),
        p_drop: float = 0.1,
        pool_after: tuple[int, ...] = (0, 1),
    ):
        super().__init__()
        self.encoder = CNN1DEncoder(
            in_channels=in_channels_eeg,
            widths=width,
            kernels=kernels,
            p_drop=p_drop,
            pool_after=pool_after,
        )
        self.head = nn.Linear(self.encoder.out_dim * input_windows, n_outputs)
        self.out_act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, W, C, T)
        B, W, C, T = x.shape
        z = rearrange(x, "B W C T -> (B W) C T")
        e = self.encoder(z)  # (B*W, F)
        e = rearrange(e, "(B W) F -> B (W F)", W=W)
        y = self.head(e)  # (B, n_outputs)
        return self.out_act(y)  # (B, n_outputs) in [0,1]