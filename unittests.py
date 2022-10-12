import unittest
import torch
import torch.nn as nn
from modules import MultiDownSampler, ResidualStack, Discriminator, MultiBandGenerator, FullBandGenerator
from stft_loss import MultiResolutionSTFTLoss
from pqmf import PQMF


class MelGANTestCase(unittest.TestCase):
    def test_optimizer(self):

        ## halve the learning rate every 10 epochs
        initial_lr = 0.01
        target_lr = 0.0001
        estimated_target_epoch = 70

        dump_model = nn.Linear(1, 1)
        opt = torch.optim.Adam(dump_model.parameters(), lr=initial_lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=opt, lr_lambda=lambda e: max(target_lr/initial_lr, (0.5)**(e//10)))

        arr_lr = []
        for _ in range(estimated_target_epoch):
            opt.step()
            scheduler.step()
            arr_lr += scheduler.get_last_lr()

        self.assertEqual(arr_lr[-1], target_lr)


    def test_melspecs(self):
        audio_length = 30000
        from config import Config
        from audio_utils import compute_mel_spec
        config = Config()
        x = torch.randn((audio_length,)).numpy()
        y = compute_mel_spec(x, config)
        target_frame_length = (audio_length-config.win_length+4*config.hop_length)//config.hop_length +1

        self.assertEqual(y.shape[-1], target_frame_length)
        self.assertEqual(y.shape[0], config.n_mels)


    def test_pqmf_signal(self):
        audio_length = 16000
        sr = 16000
        freq = 440
        omega = torch.pi * 2 * freq / sr
        t = torch.arange(0, audio_length)
        x = torch.sin(omega*t) 
        x = x.view((1, 1, audio_length))

        pqmf = PQMF()
        # x : [B, 1, T] -> [B, 4, T/4]
        subbands = pqmf.analysis(x)
        self.assertEqual(torch.Size([1, 4, audio_length//4]), subbands.shape)
        # subbands : [B, 4, T/4] -> [B, 1, T]
        x_hat = pqmf.synthesis(subbands)
        self.assertEqual(x.shape, x_hat.shape)


    def test_melgan_discriminator(self):
        batch_size = 32
        n_channel = 1
        length = 8192

        D = Discriminator()

        x = torch.randn((batch_size, n_channel, length))
        y = D(x)
        self.assertEqual(len(y), 3)

        for i in range(3):
            self.assertEqual(torch.Size([batch_size, n_channel, length//(64*(2**i))]), y[i].shape)


    def test_melgan_generator(self):
        batch_size = 32
        n_mels = 80
        frame_length = 32
        n_subbands = 4

        upsample = 200

        x = torch.randn((batch_size, n_mels, frame_length))

        ## MultiBand MelGAN
        G = MultiBandGenerator(n_mels)
        y = G(x)
        self.assertEqual(torch.Size([batch_size, n_subbands, frame_length*upsample//n_subbands]), y.shape)
        
        ## FullBand MelGAN
        G = FullBandGenerator(n_mels)
        y = G(x)
        self.assertEqual(torch.Size([batch_size, 1, frame_length*upsample]), y.shape)




    def test_residual_stack(self):
        batch_size = 32
        num_channels = 48
        
        x = torch.randn((batch_size, num_channels, 128))
        net = ResidualStack(num_channels)
        y = net(x)
        self.assertEqual(x.shape, y.shape)
        

    def test_upsampling(self):
        for factor in range(2,10):
            input_size = 32
            conv = nn.ConvTranspose1d(1, 1, factor*2, stride=factor, padding=factor//2)
            x = torch.randn((1, input_size))
            y = conv(x)

            self.assertEqual(y.size(-1)//factor, input_size)
    

    def test_multiscale_module(self):
        mscaler = MultiDownSampler()

        x = torch.tensor([[1,2,3,4,5,6,7,8]])
        scales = mscaler(x)

        self.assertTrue(torch.equal(scales[0], x)) 
        self.assertEqual(x.size(-1)//2, scales[1].size(-1))
        self.assertEqual(x.size(-1)//4, scales[2].size(-1))


    def test_torch_stft(self):
        audio_length = 16000
        sr = 16000
        freq = 440
        omega = torch.pi * 2 * freq / sr
        t = torch.arange(0, audio_length)
        x = (torch.sin(omega*t) + torch.sin(omega*2*t) + torch.sin(omega*3*t))/3 # 440hz + 880hz + 1320hz

        n_fft=1024
        hop_length = 256
        win_length = 1024
        y = torch.stft(x, n_fft, hop_length, win_length, torch.hann_window(win_length), return_complex=True)
        # y.shape : [n_ffts/2+1, n_frames]

        self.assertTrue(y.dtype.is_complex)

        frame_length = y.size(1)
        # n_frames = int((audio_length-win_length)/hop_length + 5)
        n_frames = int(audio_length/hop_length + 1)
        self.assertEqual(frame_length, n_frames)

        magnitude = torch.abs(y)
        self.assertTrue(magnitude.dtype.is_floating_point)


    def test_multiband_stftloss(self):
        stft_loss = MultiResolutionSTFTLoss()

        y_hat = torch.randn((3,16000))
        y = y_hat

        loss = stft_loss(y_hat, y)
        self.assertEqual(loss, 0)


        


if __name__ == '__main__':
    unittest.main(verbosity=2)