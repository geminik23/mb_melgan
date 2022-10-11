import os
import librosa
import numpy as np
from config import Config
from concurrent.futures import ThreadPoolExecutor, wait


def compute_mel_spec(y, config:Config):
    return librosa.feature.melspectrogram(y=y, 
                                          n_mels=config.n_mels,
                                          sr=config.sample_rate, 
                                          n_fft=config.n_fft, 
                                          hop_length=config.hop_length,
                                          win_length=config.win_length,
                                          power=config.power)


def preprocess_audio(file, target_folder, config:Config):
    # load audio and resample
    y, sr = librosa.load(file)
    # y : [1, T]
    y = librosa.resample(y, orig_sr=sr, target_sr=config.sample_rate)
    # compute the melspectrogram
    # mel_spec : [n_mels, frame_length]
    mel_spec = compute_mel_spec(y, config)
    
    fn =  os.path.basename(file).replace('.wav', '_{}.npy')
    fn_raw = fn.format('raw')
    fn_mel = fn.format('mel')

    np.save(os.path.join(target_folder, fn_raw), y)
    np.save(os.path.join(target_folder, fn_mel), mel_spec)

    return fn_raw + '\t' + fn_mel


def preprocess_audios_and_save(files, target_folder, config:Config):
    executor = ThreadPoolExecutor(max_workers=config.num_workers)
    futures = [executor.submit(preprocess_audio, f, target_folder, config) for f in files]
    results = wait(futures)
    results = [r.result() for r in results.done]

    with open(os.path.join(target_folder, 'meta.txt'), 'w+') as f:
        f.writelines(f'{l}\n' for l in results)
    




