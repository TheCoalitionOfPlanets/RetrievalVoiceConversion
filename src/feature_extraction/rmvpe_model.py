"""RMVPE: a neural pitch estimator (DeepUnet + BiGRU over a mel spectrogram).

Self-contained port of the RMVPE checkpoint used by the upstream RVC project
(the fairseq/RVC-WebUI `rmvpe.pt` weights, ~173MB). Far more accurate than
classical pyworld estimators (harvest/dio) on expressive or noisy speech,
at the cost of needing the extra checkpoint and a GPU for real-time use.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel


class BiGRU(nn.Module):
    def __init__(self, input_features, hidden_features, num_layers):
        super().__init__()
        self.gru = nn.GRU(
            input_features,
            hidden_features,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x):
        return self.gru(x)[0]


class ConvBlockRes(nn.Module):
    def __init__(self, in_channels, out_channels, momentum=0.01):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, (1, 1))

    def forward(self, x):
        if not hasattr(self, "shortcut"):
            return self.conv(x) + x
        return self.conv(x) + self.shortcut(x)


class ResEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, n_blocks=1, momentum=0.01):
        super().__init__()
        self.conv = nn.ModuleList([ConvBlockRes(in_channels, out_channels, momentum)])
        for _ in range(n_blocks - 1):
            self.conv.append(ConvBlockRes(out_channels, out_channels, momentum))
        self.kernel_size = kernel_size
        if kernel_size is not None:
            self.pool = nn.AvgPool2d(kernel_size=kernel_size)

    def forward(self, x):
        for conv in self.conv:
            x = conv(x)
        if self.kernel_size is not None:
            return x, self.pool(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, in_size, n_encoders, kernel_size, n_blocks, out_channels=16, momentum=0.01):
        super().__init__()
        self.n_encoders = n_encoders
        self.bn = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.layers = nn.ModuleList()
        for _ in range(n_encoders):
            self.layers.append(
                ResEncoderBlock(in_channels, out_channels, kernel_size, n_blocks, momentum)
            )
            in_channels = out_channels
            out_channels *= 2
            in_size //= 2
        self.out_channel = out_channels

    def forward(self, x):
        concat_tensors = []
        x = self.bn(x)
        for layer in self.layers:
            t, x = layer(x)
            concat_tensors.append(t)
        return x, concat_tensors


class Intermediate(nn.Module):
    def __init__(self, in_channels, out_channels, n_inters, n_blocks, momentum=0.01):
        super().__init__()
        self.layers = nn.ModuleList([ResEncoderBlock(in_channels, out_channels, None, n_blocks, momentum)])
        for _ in range(n_inters - 1):
            self.layers.append(ResEncoderBlock(out_channels, out_channels, None, n_blocks, momentum))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ResDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, n_blocks=1, momentum=0.01):
        super().__init__()
        out_padding = (0, 1) if stride == (1, 2) else (1, 1)
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, (3, 3), stride, (1, 1), out_padding, bias=False
            ),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        self.conv2 = nn.ModuleList([ConvBlockRes(out_channels * 2, out_channels, momentum)])
        for _ in range(n_blocks - 1):
            self.conv2.append(ConvBlockRes(out_channels, out_channels, momentum))

    def forward(self, x, concat_tensor):
        x = self.conv1(x)
        x = torch.cat((x, concat_tensor), dim=1)
        for conv2 in self.conv2:
            x = conv2(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, n_decoders, stride, n_blocks, momentum=0.01):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_decoders):
            out_channels = in_channels // 2
            self.layers.append(ResDecoderBlock(in_channels, out_channels, stride, n_blocks, momentum))
            in_channels = out_channels

    def forward(self, x, concat_tensors):
        for i, layer in enumerate(self.layers):
            x = layer(x, concat_tensors[-1 - i])
        return x


class DeepUnet(nn.Module):
    def __init__(self, kernel_size, n_blocks, en_de_layers=5, inter_layers=4, in_channels=1, en_out_channels=16):
        super().__init__()
        self.encoder = Encoder(in_channels, 128, en_de_layers, kernel_size, n_blocks, en_out_channels)
        self.intermediate = Intermediate(
            self.encoder.out_channel // 2, self.encoder.out_channel, inter_layers, n_blocks
        )
        self.decoder = Decoder(self.encoder.out_channel, en_de_layers, kernel_size, n_blocks)

    def forward(self, x):
        x, concat_tensors = self.encoder(x)
        x = self.intermediate(x)
        return self.decoder(x, concat_tensors)


class E2E(nn.Module):
    def __init__(self, n_blocks, n_gru, kernel_size, en_de_layers=5, inter_layers=4, in_channels=1, en_out_channels=16):
        super().__init__()
        self.unet = DeepUnet(kernel_size, n_blocks, en_de_layers, inter_layers, in_channels, en_out_channels)
        self.cnn = nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))
        self.fc = nn.Sequential(
            BiGRU(3 * 128, 256, n_gru),
            nn.Linear(512, 360),
            nn.Dropout(0.25),
            nn.Sigmoid(),
        )

    def forward(self, mel):
        mel = mel.transpose(-1, -2).unsqueeze(1)
        x = self.cnn(self.unet(mel)).transpose(1, 2).flatten(-2)
        return self.fc(x)


class MelSpectrogram(nn.Module):
    def __init__(self, is_half, n_mel_channels, sampling_rate, win_length, hop_length, n_fft=None, mel_fmin=0, mel_fmax=None, clamp=1e-5):
        super().__init__()
        n_fft = win_length if n_fft is None else n_fft
        self.hann_window = {}
        mel_basis = torch.from_numpy(
            librosa_mel(sr=sampling_rate, n_fft=n_fft, n_mels=n_mel_channels, fmin=mel_fmin, fmax=mel_fmax, htk=True)
        ).float()
        self.register_buffer("mel_basis", mel_basis)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.clamp = clamp
        self.is_half = is_half

    def forward(self, audio, center=True):
        key = str(audio.device)
        if key not in self.hann_window:
            self.hann_window[key] = torch.hann_window(self.win_length).to(audio.device)
        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.hann_window[key],
            center=center,
            return_complex=True,
        )
        magnitude = torch.sqrt(fft.real.pow(2) + fft.imag.pow(2))
        mel_output = torch.matmul(self.mel_basis, magnitude)
        if self.is_half:
            mel_output = mel_output.half()
        return torch.log(torch.clamp(mel_output, min=self.clamp))


class RMVPEModel:
    """Loads the rmvpe.pt checkpoint and extracts an F0 contour (100 fps @ 16kHz)."""

    def __init__(self, model_path: str, device: torch.device, fp16: bool = False):
        self.device = device
        self.is_half = fp16 and device.type == "cuda"

        self.mel_extractor = MelSpectrogram(
            self.is_half, 128, 16000, 1024, 160, None, 30, 8000
        ).to(device)

        model = E2E(4, 1, (2, 2))
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        model = model.half() if self.is_half else model.float()
        self.model = model.to(device)

        cents_mapping = 20 * np.arange(360) + 1997.3794084376191
        self.cents_mapping = np.pad(cents_mapping, (4, 4))

    @torch.no_grad()
    def _mel2hidden(self, mel):
        n_frames = mel.shape[-1]
        n_pad = 32 * ((n_frames - 1) // 32 + 1) - n_frames
        if n_pad > 0:
            mel = F.pad(mel, (0, n_pad), mode="constant")
        mel = mel.half() if self.is_half else mel.float()
        hidden = self.model(mel)
        return hidden[:, :n_frames]

    def _decode(self, hidden, thred=0.03):
        center = np.argmax(hidden, axis=1)
        salience = np.pad(hidden, ((0, 0), (4, 4)))
        center += 4
        starts = center - 4
        ends = center + 5
        todo_salience = np.array(
            [salience[idx, starts[idx] : ends[idx]] for idx in range(salience.shape[0])]
        )
        todo_cents = np.array([self.cents_mapping[starts[idx] : ends[idx]] for idx in range(salience.shape[0])])
        product_sum = np.sum(todo_salience * todo_cents, 1)
        weight_sum = np.sum(todo_salience, 1)
        cents_pred = product_sum / weight_sum
        maxx = np.max(hidden, axis=1)
        cents_pred[maxx <= thred] = 0
        f0 = 10 * (2 ** (cents_pred / 1200))
        f0[f0 == 10] = 0
        return f0

    @torch.no_grad()
    def infer_from_audio(self, audio_16k: np.ndarray, thred: float = 0.03) -> np.ndarray:
        waveform = torch.from_numpy(audio_16k).float().to(self.device).unsqueeze(0)
        mel = self.mel_extractor(waveform, center=True)
        hidden = self._mel2hidden(mel).squeeze(0).cpu().numpy()
        if self.is_half:
            hidden = hidden.astype(np.float32)
        return self._decode(hidden, thred=thred)
