
import torch
import torch.nn.functional as F
import numpy as np
from pqmf import PQMF
from utils import to_device
import time

from base_trainer import BaseTrainer


class FBMelGANTrainer(BaseTrainer):
    def __init__(self, g_model, d_model, g_optimizer_builder, d_optimizer_builder, g_scheduler_builder=None, d_scheduler_builder=None, lambda_adv=2.5, checkpoint_dir="model_cp"):
        super().__init__( g_model, d_model, g_optimizer_builder, d_optimizer_builder, g_scheduler_builder, d_scheduler_builder, lambda_adv, checkpoint_dir)

 
    def run_epoch(self, data_loader, device, train_only_G=False, desc=None):
        d_losses = []
        g_losses = []

        # measure the time
        start = time.time()
        for inputs, targets in self.tqdm(data_loader, desc=desc, leave=False):
            inputs = to_device(inputs, device)
            targets = to_device(targets, device)
            
            g_out = self.g_model(inputs)

            ##
            # train discriminator
            if not train_only_G:
                # scores : [B, C=1, 128] X 3
                fake_scores = self.d_model(g_out.detach())
                real_scores = self.d_model(targets)

                d_f_losses, d_r_losses = [],[]
                for score in fake_scores: d_f_losses.append(F.mse_loss(score, torch.zeros_like(score)))
                for score in real_scores: d_r_losses.append(F.mse_loss(score, torch.ones_like(score)))
                
                d_loss = torch.stack(d_f_losses).mean() + torch.stack(d_r_losses).mean()

                if self.d_model.training:
                    self.d_optimizer.zero_grad()
                    d_loss.backward()
                    self.d_optimizer.step()
                
                    if self.d_lr_schedule is not None:
                        self.d_lr_schedule.step()
            
            d_losses.append(d_loss.item() if not train_only_G else 0)

            # 
            # train_g = True

            ##
            # train genenator
            # if train_g:
            fake_scores = self.d_model(g_out.detach())

            # adversarial loss
            d_adv_losses = []
            for score in fake_scores: d_adv_losses.append(F.mse_loss(score, torch.ones_like(score)))
            adv_loss = torch.stack(d_adv_losses).mean()

            # stft loss
            stft_loss = self.stft_loss(g_out[:, 0, :], targets[:, 0, :])
                
            # total
            if train_only_G:
                g_loss = stft_loss
            else:
                g_loss = self.lambda_adv * adv_loss + stft_loss

            # only when training
            if self.g_model.training:
                self.g_optimizer.zero_grad()
                g_loss.backward()
                self.g_optimizer.step()

                if self.g_lr_schedule is not None:
                    self.g_lr_schedule.step()
            # g_losses.append(g_loss.item() if train_g else 0)
            g_losses.append(g_loss.item())

        end = time.time() 
        
        self.result["g loss"].append(np.mean(g_losses))
        self.result["d loss"].append(np.mean(d_losses))

        return end-start


class MBMelGANTrainer(BaseTrainer):
    def __init__(self, g_model, d_model, g_optimizer_builder, d_optimizer_builder, g_scheduler_builder=None, d_scheduler_builder=None, lambda_adv=2.5, checkpoint_dir="model_cp"):
        super().__init__( g_model, d_model, g_optimizer_builder, d_optimizer_builder, g_scheduler_builder, d_scheduler_builder, lambda_adv, checkpoint_dir)
        self.pqmf = PQMF()

    def run_epoch(self, data_loader, device, train_only_G=False, desc=None):
        d_losses = []
        g_losses = []

        self.pqmf = self.pqmf.to(device)

        # measure the time
        start = time.time()
        for inputs, targets in self.tqdm(data_loader, desc=desc, leave=False):
            inputs = to_device(inputs, device)
            targets = to_device(targets, device)
            
            g_out_sub = self.g_model(inputs)
            g_out = self.pqmf.synthesis(g_out_sub.detach())

            with torch.no_grad():
                targets_sub = self.pqmf.analysis(targets)

            ##
            # train discriminator
            if not train_only_G:
                # scores : [B, C=1, 128] X 3
                fake_scores = self.d_model(g_out.detach())
                real_scores = self.d_model(targets)

                d_f_losses, d_r_losses = [],[]
                for score in fake_scores: d_f_losses.append(F.mse_loss(score, torch.zeros_like(score)))
                for score in real_scores: d_r_losses.append(F.mse_loss(score, torch.ones_like(score)))
                
                d_loss = torch.stack(d_f_losses).mean() + torch.stack(d_r_losses).mean()

                if self.d_model.training:
                    self.d_optimizer.zero_grad()
                    d_loss.backward()
                    self.d_optimizer.step()
                
                    if self.d_lr_schedule is not None:
                        self.d_lr_schedule.step()
            
            d_losses.append(d_loss.item() if not train_only_G else 0)

            # train_g = True

            ##
            # train genenator
            # if train_g:
            fake_scores = self.d_model(g_out.detach())

            # adversarial loss
            d_adv_losses = []
            for score in fake_scores: d_adv_losses.append(F.mse_loss(score, torch.ones_like(score)))
            adv_loss = torch.stack(d_adv_losses).mean()

            # stft loss
            fb_loss = self.stft_loss(g_out[:, 0, :], targets[:, 0, :])
            
            sb_losses = []
            for i in range(4): # 4 sub bands
                sb_losses.append(self.stft_loss(g_out_sub[:, i, :], targets_sub[:, i, :]))
            sb_loss = torch.stack(sb_losses).mean()

            stft_loss = (fb_loss + sb_loss) / 2.0
                
            # total
            if train_only_G:
                g_loss = stft_loss
            else:
                g_loss = self.lambda_adv * adv_loss + stft_loss

            # only when training
            if self.g_model.training:
                self.g_optimizer.zero_grad()
                g_loss.backward()
                self.g_optimizer.step()

                if self.g_lr_schedule is not None:
                    self.g_lr_schedule.step()

            g_losses.append(g_loss.item())
            # g_losses.append(g_loss.item() if train_g else torch.zeros(1))
        end = time.time() 

        
        self.result["g loss"].append(np.mean(g_losses))
        self.result["d loss"].append(np.mean(d_losses))

        return end-start
