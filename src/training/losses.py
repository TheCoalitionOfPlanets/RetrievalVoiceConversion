"""GAN + reconstruction losses for the VITS/HiFi-GAN objective."""

import torch


def feature_loss(fmap_r: list, fmap_g: list) -> torch.Tensor:
    """L1 between discriminator intermediate activations (stabilises the GAN)."""

    loss = 0.0

    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl.float().detach() - gl.float()))

    return loss * 2


def discriminator_loss(
    disc_real_outputs: list,
    disc_generated_outputs: list,
) -> tuple[torch.Tensor, list[float], list[float]]:
    """Least-squares GAN loss: push real toward 1, generated toward 0."""

    loss = 0.0
    r_losses, g_losses = [], []

    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = dr.float()
        dg = dg.float()

        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg**2)

        loss += r_loss + g_loss
        r_losses.append(float(r_loss.detach()))
        g_losses.append(float(g_loss.detach()))

    return loss, r_losses, g_losses


def generator_loss(disc_outputs: list) -> tuple[torch.Tensor, list[float]]:
    loss = 0.0
    gen_losses = []

    for dg in disc_outputs:
        dg = dg.float()
        current = torch.mean((1 - dg) ** 2)

        gen_losses.append(float(current.detach()))
        loss += current

    return loss, gen_losses


def kl_loss(
    z_p: torch.Tensor,
    logs_q: torch.Tensor,
    m_p: torch.Tensor,
    logs_p: torch.Tensor,
    z_mask: torch.Tensor,
) -> torch.Tensor:
    """KL( q(z|spec) || p(z|content,f0) ), averaged over valid frames."""

    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()

    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    kl = torch.sum(kl * z_mask)

    return kl / torch.sum(z_mask)
