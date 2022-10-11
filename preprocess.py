import os

from config import Config
from audio_utils import preprocess_audios_and_save
from glob import glob

if __name__ == '__main__':
    # preprocess the the lj speech datasets
    config = Config()

    files = glob(os.path.join(config.dataset_path, f"{config.lj_folder_name}/wavs/*.wav"))
    target_path = os.path.join(config.dataset_path, config.lj_processed_path)

    os.makedirs(target_path, exist_ok=True)
    
    preprocess_audios_and_save(files, target_path, config)
