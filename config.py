import os
import dotenv

dotenv.load_dotenv()


DATASET_PATH = os.environ.get('DATASET_PATH')
if DATASET_PATH is None:
    DATASET_PATH = './data'
    os.makedirs(DATASET_PATH, exist_ok=True)


class Config:
    def __init__(self):

        self.num_workers = 16

        # dataset
        self.dataset_path = DATASET_PATH
        self.lj_folder_name ="LJSpeech-1.1"
        self.lj_processed_path = os.path.join(self.lj_folder_name, 'processed')
        self.val_ratio = 0.05


        # Audio pre-process
        self.sample_rate = 16000 # need to convert
        self.preprocessed_lj_datapath = os.path.join(self.dataset_path, self.lj_folder_name + '/melwav')

        self.upsample_factor = 200
        # 'Each batch randomly intercepts one second of audio' from paper
        self.sample_length = self.sample_rate # fixed time = 1sec

        # melspectrogram args
        self.hop_length = 200 # x200 for MB-MelGAN
        self.win_length = 800 # 4 x hop_length
        self.n_fft = 10244
        self.n_mels = 80
        self.fmin = 0
        self.fmax = 8000
        self.power = 1.0


        # checkpoint
        self.checkpoint_dir = "model_cp"
        self.checkpoint_file_template = "mbmelgan_e_{}.pt"
        self.checkpoint_file_template_fb = "fbmelgan_e_{}.pt"
        self.checkpoint_interval = 100

        # Hyperparamters
        self.batch_size = 128 # batch-size 48, 128 to fb and mb respectively
        self.batch_size_fb = 48 # batch-size 48, 128 to fb and mb respectively
        self.g_lr = 1e-4 # 1e-4 adam optimizer
        self.d_lr = 1e-4  # instead n_critic different learning rate x3~5
        self.adam_betas= (0.5, 0.9)
        self.lambda_adv = 2.5
        self.epochs = 500000
        self.train_after = None # full file path e.g "model_cp/mbmelgan_e_100.pt"
        # self.train_after = "model_cp/mbmelgan_e_200.pt"
        self.train_generator_until = 200000

        
