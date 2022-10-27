
import torch
import torch.nn as nn


def spectral_convergence(prediction, target):
    return torch.norm(target - prediction, p='fro') / torch.norm(target, p='fro')


# !issue.. 
# magnitude contains 0.0 and log of value will be inf.
def log_stft_magnitude(prediction, target):
    prediction[prediction==0.0] = 1e-9
    target[target==0.0] = 1e-9
    log_t = torch.log(target)
    log_p = torch.log(prediction)
    return nn.functional.l1_loss(log_p, log_t, reduction='mean')


class SingleSTFTLoss(nn.Module):
    def __init__(self, n_fft, hop_length, win_length):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("win", torch.hann_window(win_length))

    def forward(self, prediction, target):
        p_stft= torch.stft(prediction, self.n_fft, self.hop_length, self.win_length, self.win, return_complex=True)
        t_stft = torch.stft(target, self.n_fft, self.hop_length, self.win_length, self.win, return_complex=True)

        # [B, T, n_fft/2+1]
        p_mag = torch.abs(p_stft).permute(0,2,1)
        t_mag = torch.abs(t_stft).permute(0,2,1)

        sc_loss = spectral_convergence(p_mag, t_mag)
        lmag_loss = log_stft_magnitude(p_mag, t_mag)

        return sc_loss, lmag_loss


class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self, n_ffts=[384, 683, 171], hop_lengths=[30, 60, 10], win_lengths=[150, 300, 60]): # default setting is for Multi-band MelGAN
        super().__init__()
        losses = []
        for (n_fft, hop_length, win_length) in zip(n_ffts, hop_lengths, win_lengths):
            losses.append(SingleSTFTLoss(n_fft, hop_length, win_length))
        self.losses= nn.ModuleList(losses)

    def forward(self, y_hat, y):
        """
        average the M single stft losses
        Args:
        both y_hat and y are audio signal: [B, T]
        """
        scs, lmags = [], []
        for loss in self.losses:
            sc_loss, lmag_loss = loss(y_hat, y)
            scs.append(sc_loss)
            lmags.append(lmag_loss)

        return (torch.stack(scs) + torch.stack(lmags)).mean()