
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import seaborn as sns

from utils import weights_init_xavier_uniform
from dataset import load_ljspeech_dataset
from modules import MBMelGANDiscriminator, MBMelGANGenerator
from trains import MBMelGANTrainer


##
# Hyperparameters from config
from config import Config
config = Config()


##
# Datasets

trainset, _ = load_ljspeech_dataset(config)

train_loader = DataLoader(trainset, batch_size=config.batch_size, pin_memory=False, shuffle=True)
# train_loader = DataLoader(trainset, batch_size=config.batch_size, pin_memory=False, shuffle=True, num_workers=config.num_workers)


##
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


##
# Initialize the model, optimizer, loss function

G = MBMelGANGenerator(config.n_mels)
D = MBMelGANDiscriminator()

weights_init_xavier_uniform(G)
weights_init_xavier_uniform(D)



## Training setting for mb-melgan from paper
# lambda =10
# multi resolution STFT loss lambda =2.5

# pre_net = MuLawEncoder(model.mu)
g_optimizer_builder = lambda model:torch.optim.Adam(G.parameters(), lr=config.g_lr, betas=config.adam_betas)
d_optimizer_builder = lambda model:torch.optim.Adam(D.parameters(), lr=config.d_lr, betas=config.adam_betas)

# 'The learning rate of all models is halved every 100K steps until 1e − 6' from paper.
g_scheduler_builder = lambda opt: torch.optim.lr_scheduler.LambdaLR(optimizer=opt, lr_lambda=lambda e: max(1e-6/config.g_lr, (0.5)**(e//100000)))
d_scheduler_builder = lambda opt: torch.optim.lr_scheduler.LambdaLR(optimizer=opt, lr_lambda=lambda e: max(1e-6/config.d_lr, (0.5)**(e//100000)))



##
# Trainer
trainer = MBMelGANTrainer(G, D,
                          g_optimizer_builder,
                          d_optimizer_builder,
                          g_scheduler_builder,
                          d_scheduler_builder,
                          lambda_adv=config.lambda_adv,
                          checkpoint_dir=config.checkpoint_dir)
trainer.set_tqdm_for_notebook(True)



##
# if train_after is not None then load data and continue the train
reset = True
if config.train_after is not None:
    trainer.load_data( config.train_after)
    reset = False


##
# Train
result = trainer.train(train_loader, train_only_g_till=config.train_generator_until, epochs=config.epochs, device=device, reset=reset, cp_filename=config.checkpoint_file_template, cp_interval=config.checkpoint_interval)




##
# plot the losses
sns.lineplot(x='epoch', y='g loss', data=result, label='Generator Loss')
sns.lineplot(x='epoch', y='d loss', data=result, label='Discriminator Loss')
plt.show()
