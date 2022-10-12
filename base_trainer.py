
import torch
import tqdm 
import pandas as pd
import os
from stft_loss import MultiResolutionSTFTLoss

from tqdm.autonotebook import tqdm as nb_tqdm
from tqdm import tqdm


# two models / two optimizer 
class BaseTrainer(object):
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
        pass
    # ----------------------
