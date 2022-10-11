
import torch
import torch.nn.functional as F
import numpy as np
import tqdm 
import pandas as pd
import os
from pqmf import PQMF
from stft_loss import MultiResolutionSTFTLoss
from utils import to_device
import time

from tqdm.autonotebook import tqdm as nb_tqdm
from tqdm import tqdm


# two models / two optimizer 
class MBMelGANTrainer(object):
    def __init__(self, 
                 g_model, 
                 d_model,
                 g_optimizer_builder, 
                 d_optimizer_builder, 
                 g_scheduler_builder=None, 
                 d_scheduler_builder=None, 
                 lambda_adv=2.5,
                 checkpoint_dir="model_cp"):
        self.g_model = g_model
        self.d_model = d_model
        self.g_optimizer_builder = g_optimizer_builder
        self.d_optimizer_builder = d_optimizer_builder
        self.g_scheduler_builder = g_scheduler_builder
        self.d_scheduler_builder = d_scheduler_builder
        self.lambda_adv = lambda_adv
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.tqdm = tqdm
        self.reset()

        self.pqmf = PQMF()
        self.stft_loss = MultiResolutionSTFTLoss()
    
    def set_tqdm_for_notebook(self, notebook_tqdm=False):
        self.tqdm = nb_tqdm if notebook_tqdm else tqdm
    
    def _init(self):
        self.result = {}

        record_keys = ["epoch", "total time", "g loss", "d loss"]
        for item in record_keys:
            self.result[item] = []

    def reset(self):
        self.total_time = 0
        self.last_epoch = 0
        self.g_optimizer = self.g_optimizer_builder(self.g_model)
        self.d_optimizer = self.d_optimizer_builder(self.d_model)
        self.g_lr_schedule = None if self.g_scheduler_builder is None else self.g_scheduler_builder(self.g_optimizer)
        self.d_lr_schedule = None if self.d_scheduler_builder is None else self.d_scheduler_builder(self.d_optimizer)
        self.result = {}
    
    def load_data(self, full_filepath):
        self.reset()

        data = torch.load(full_filepath)
        if data.get('epoch') is not None:
            self.last_epoch = data.get('epoch') 
        if data.get('result') is not None:
            self.result = data.get('result')
        if self.result.get('total time') is not None and len(self.result['total time'])!=0:
            self.total_time = self.result['total time'][-1]
        self.g_model.load_state_dict(data.get('g_model_state_dict'))
        self.d_model.load_state_dict(data.get('d_model_state_dict'))
        self.g_optimizer.load_state_dict(data.get('g_optimizer_state_dict'))
        self.d_optimizer.load_state_dict(data.get('d_optimizer_state_dict'))
        if self.g_lr_schedule is not None:
            self.g_lr_schedule.load_state_dict(data.get('g_rl_schedule_state_dict'))
            self.g_lr_schedule.last_epoch = self.last_epoch
        if self.d_lr_schedule is not None:
            self.d_lr_schedule.load_state_dict(data.get('d_rl_schedule_state_dict'))
            self.d_lr_schedule.last_epoch = self.last_epoch

    def save_data(self, filename):
        torch.save({
            'epoch': self.last_epoch,
            'g_model_state_dict': self.g_model.state_dict(),
            'd_model_state_dict': self.d_model.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'g_rl_schedule_state_dict': None if self.g_lr_schedule is None else self.g_rl_schedule.state_dict(),
            'd_rl_schedule_state_dict': None if self.d_lr_schedule is None else self.d_rl_schedule.state_dict(),
            'result' : self.result,
            }, os.path.join(self.checkpoint_dir, filename))

    def get_result(self):
        return pd.DataFrame.from_dict(self.result)

    def train(self, data_loader, epochs=10, train_only_g_till=200000, device='cpu', reset=True, cp_filename=None, cp_interval =10, print_progress=False):
        ## initialize
        # init result
        if reset: self.reset()

        if len(self.result) == 0 or reset: self._init()

        # set device
        is_cuda = False
        if type(device) == torch.device:
            is_cuda = device.type.startswith('cuda')
        elif type(device) == str:
            is_cuda = device.startswith('cuda')
        
        if is_cuda:
            for state in self.d_optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
            for state in self.g_optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
            
        self.g_model.to(device)
        self.d_model.to(device)

        self.stft_loss = self.stft_loss.to(device)
        self.pqmf = self.pqmf.to(device)

        for epoch in self.tqdm(range(self.last_epoch + 1, self.last_epoch + 1 + epochs), desc="Epoch"):
            self.g_model = self.g_model.train()
            self.d_model = self.d_model.train()

            self.total_time += self.run_epoch(data_loader, device, train_only_G=train_only_g_till>=epoch, desc="training")
            
            self.result["epoch"].append( epoch )
            self.result["total time"].append( self.total_time )
        
            self.last_epoch = epoch

            if print_progress:
                total_secs = int(self.total_time)
                print(f"Epoch {epoch} - g loss : {self.result['g loss'][-1]}, d loss: {self.result['d loss'][-1]}, time : {total_secs//60}:{total_secs%60}")
            
            if cp_filename is not None and epoch%cp_interval== 0:
                self.save_data(cp_filename.format(epoch))
            
        return self.get_result()

    # ----------------------
    def run_epoch(self, data_loader, device, train_only_G=False, desc=None):
        d_losses = []
        g_losses = []

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
            
            d_losses.append(d_loss.item() if not train_only_G else torch.zeros(1))

            train_g = True

            ##
            # train genenator
            if train_g:
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

                    
                # total
                if train_only_G:
                    g_loss = (fb_loss + sb_loss)/2.0
                else:
                    g_loss = self.lambda_adv * adv_loss + (fb_loss + sb_loss)/2.0

                # only when training
                if self.g_model.training:
                    self.g_optimizer.zero_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                    if self.g_lr_schedule is not None:
                        self.g_lr_schedule.step()

            g_losses.append(g_loss.item() if train_g else torch.zeros(1))
        end = time.time() 
        
        self.result["g loss"].append(np.mean(g_losses))
        self.result["d loss"].append(np.mean(d_losses))

        return end-start


def train_conditional_wgan_gp(cp_file_template, G, D, latent_size, optimizer_G, optimizer_D, data_loader, epochs, device='cpu'):
    """
    Args:
    str: cp_file_template e.g "model_{}_e_{}.pt"

    Return:
    tuple: (generator loss, discriminator loss)
    """

    G.to(device)
    D.to(device)


    g_losses = []
    d_losses = []

    G.train()
    D.train()
    for epoch in tqdm(range(epochs)):
        for data in tqdm(data_loader, leave=False):
            data, class_label = data

            real = data.to(device)
            class_label = class_label.to(device)

            bsize = data.size(0)

            ### Step 1 : update D
            D.zero_grad()
            G.zero_grad()

            # real
            d_real = D(real, class_label)


            # fake
            z = torch.randn(bsize, latent_size, device=device)
            fake = G(z, class_label)
            d_fake = D(fake, class_label)

            # gradient penalty
            eps_shape = [bsize]+[1]*(len(data.shape)-1)
            eps = torch.rand(eps_shape, device=device)
            fake = eps*real + (1-eps)*fake
            output = D(fake, class_label)

            grad = torch.autograd.grad(outputs=output, inputs=fake,
                                  grad_outputs=torch.ones(output.size(), device=device),
                                  create_graph=True, retain_graph=True, only_inputs=True, allow_unused=True)[0]
            d_grad_penalty = ((grad.norm(2, dim=1) - 1) ** 2).mean()

            errd = (d_fake-d_real).mean() + d_grad_penalty.mean()*10
            errd.backward()
            optimizer_D.step()

            d_losses.append(errd.item())

            #### Step 2 : update G
            D.zero_grad()
            G.zero_grad()

            noise = torch.randn(bsize, latent_size, device=device)
            output = -D(G(noise, class_label), class_label)
            errg = output.mean()
            errg.backward()
            optimizer_G.step()

            g_losses.append(errg.item())

        # save check_points
        torch.save({
            'losses': g_losses,
            'epoch': epoch,
            'model_state_dict': G.state_dict(),
            }, cp_file_template.format('g', epoch))

        torch.save({
            'losses': d_losses,
            'epoch': epoch,
            'model_state_dict': D.state_dict(),
        }, cp_file_template.format('d', epoch))
        pass


    return g_losses, d_losses
