"""Model shapes, losses and the dataset contract."""

import numpy as np
import pytest
import torch

from src.models.dataset import TrainingSample
from src.models.vits.commons import rand_slice_segments
from src.models.vits.commons import slice_segments
from src.models.vits.discriminator import MultiPeriodDiscriminator
from src.models.vits.model import SynthesizerTrnMs768NSFsid
from src.training.losses import discriminator_loss
from src.training.losses import feature_loss
from src.training.losses import generator_loss
from src.training.losses import kl_loss


@pytest.fixture(scope="module")
def small_config():
    """A deliberately tiny model so the GAN step fits in CPU test time."""

    from src.utils.settings import AudioConfig
    from src.utils.settings import GeneratorConfig
    from src.utils.settings import ModelConfig

    config = ModelConfig()
    config.audio = AudioConfig(sample_rate=32000, hop_length=320)
    config.generator = GeneratorConfig(
        inter_channels=32,
        hidden_channels=32,
        filter_channels=64,
        n_heads=2,
        n_layers=2,
        upsample_initial_channel=32,
        resblock_kernel_sizes=[3],
        resblock_dilation_sizes=[[1, 3]],
        spk_embed_dim=4,
        gin_channels=32,
    )

    return config


def test_slice_segments_respects_start_indices():
    x = torch.arange(2 * 3 * 10, dtype=torch.float).reshape(2, 3, 10)
    ids = torch.LongTensor([0, 4])

    out = slice_segments(x, ids, 3)

    assert out.shape == (2, 3, 3)
    assert torch.equal(out[0], x[0, :, 0:3])
    assert torch.equal(out[1], x[1, :, 4:7])


def test_rand_slice_never_runs_past_the_true_length():
    x = torch.randn(8, 4, 100)
    lengths = torch.LongTensor([20] * 8)

    _, ids = rand_slice_segments(x, lengths, segment_size=10)

    assert int(ids.max()) <= 20 - 10 + 1


def test_generator_forward_shapes(small_config):
    batch, frames = 2, 40
    hop = small_config.audio.hop_length
    segment_size = 8 * hop

    net_g = SynthesizerTrnMs768NSFsid.from_config(small_config, segment_size)

    content = torch.randn(batch, frames, 768)
    content_lengths = torch.LongTensor([frames, frames])
    pitch = torch.randint(0, 255, (batch, frames))
    pitchf = torch.rand(batch, frames) * 300
    spec = torch.randn(batch, small_config.audio.filter_length // 2 + 1, frames).abs()
    spec_lengths = torch.LongTensor([frames, frames])
    sid = torch.LongTensor([0, 1])

    y_hat, ids_slice, _, z_mask, latents = net_g(
        content, content_lengths, pitch, pitchf, spec, spec_lengths, sid
    )

    assert y_hat.shape == (batch, 1, segment_size), (
        "decoder output must be segment_size samples long"
    )
    assert ids_slice.shape == (batch,)
    assert z_mask.shape == (batch, 1, frames)
    assert len(latents) == 6
    assert torch.isfinite(y_hat).all()


def test_generator_infer_length_matches_upsampling(small_config):
    frames = 30
    net_g = SynthesizerTrnMs768NSFsid.from_config(small_config, 2560).eval()

    content = torch.randn(1, frames, 768)
    lengths = torch.LongTensor([frames])
    pitch = torch.randint(1, 255, (1, frames))
    nsff0 = torch.full((1, frames), 200.0)
    sid = torch.LongTensor([0])

    audio, _, _ = net_g.infer(content, lengths, pitch, nsff0, sid)

    upsample = int(np.prod(small_config.generator.upsample_rates))

    assert audio.shape == (1, 1, frames * upsample)
    assert torch.isfinite(audio).all()
    assert audio.abs().max() <= 1.0, "tanh output must stay in [-1, 1]"


def test_discriminator_returns_matched_feature_maps():
    net_d = MultiPeriodDiscriminator(periods=[2, 3])

    y = torch.randn(2, 1, 4096)
    y_hat = torch.randn(2, 1, 4096)

    y_d_rs, y_d_gs, fmap_rs, fmap_gs = net_d(y, y_hat)

    assert len(y_d_rs) == len(y_d_gs) == 3  # one scale + two periods
    assert len(fmap_rs) == len(fmap_gs) == 3

    for real, fake in zip(fmap_rs, fmap_gs):
        assert len(real) == len(fake)


def test_discriminator_loss_is_zero_at_the_optimum():
    perfect_real = [torch.ones(2, 10)]
    perfect_fake = [torch.zeros(2, 10)]

    loss, r_losses, g_losses = discriminator_loss(perfect_real, perfect_fake)

    assert float(loss) == pytest.approx(0.0, abs=1e-6)
    assert r_losses[0] == pytest.approx(0.0, abs=1e-6)
    assert g_losses[0] == pytest.approx(0.0, abs=1e-6)


def test_generator_loss_is_zero_when_discriminator_is_fooled():
    loss, _ = generator_loss([torch.ones(2, 10)])

    assert float(loss) == pytest.approx(0.0, abs=1e-6)


def test_feature_loss_is_zero_for_identical_maps():
    fmap = [[torch.randn(2, 4, 8)]]

    assert float(feature_loss(fmap, fmap)) == pytest.approx(0.0, abs=1e-6)


def test_kl_loss_vanishes_in_expectation_when_prior_matches_posterior():
    """kl_loss is a single-sample Monte Carlo estimate.

    It only averages to zero when z_p is an actual draw from q, so a large
    batch of real samples is the only honest way to assert the optimum.
    """

    torch.manual_seed(0)
    shape = (4096, 4, 10)

    m_p = torch.zeros(shape)
    logs_p = torch.zeros(shape)
    logs_q = torch.zeros(shape)
    z_p = m_p + torch.exp(logs_q) * torch.randn(shape)
    z_mask = torch.ones(shape[0], 1, shape[2])

    loss = kl_loss(z_p, logs_q, m_p, logs_p, z_mask)

    assert float(loss) == pytest.approx(0.0, abs=0.05)


def test_kl_loss_grows_with_divergence():
    shape = (2, 4, 10)
    z_mask = torch.ones(2, 1, 10)

    zeros = torch.zeros(shape)

    near = kl_loss(zeros, zeros, zeros, zeros, z_mask)
    far = kl_loss(torch.full(shape, 3.0), zeros, zeros, zeros, z_mask)

    # A posterior sample 3 sigma from the prior mean must cost more.
    assert float(far) > float(near)


def test_training_sample_roundtrips_through_the_filelist():
    sample = TrainingSample("a.npy", "b.npy", "c.npy", "d.npy", 3, 120)

    assert TrainingSample.from_line(sample.to_line()) == sample


def test_malformed_filelist_line_is_rejected():
    with pytest.raises(ValueError, match="Malformed"):
        TrainingSample.from_line("only|three|fields")


def test_collate_pads_and_sorts_by_length():
    from src.training.dataset import VoiceConversionCollate

    def make(frames, hop=320):
        return (
            torch.randn(frames, 768),
            torch.randint(0, 255, (frames,)),
            torch.rand(frames),
            torch.randn(513, frames),
            torch.randn(1, frames * hop),
            torch.LongTensor([0]),
        )

    batch = [make(10), make(30), make(20)]

    out = VoiceConversionCollate()(batch)
    content, content_lengths, _, _, spec, spec_lengths, wav, wav_lengths, _ = out

    assert content.shape == (3, 30, 768)
    assert spec.shape == (3, 513, 30)
    # Longest first, so the padded region shrinks monotonically.
    assert list(spec_lengths) == [30, 20, 10]
    assert list(content_lengths) == [30, 20, 10]
    assert list(wav_lengths) == [30 * 320, 20 * 320, 10 * 320]
