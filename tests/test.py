from fastai.vision.all import *
import timm
from timm import *
import numpy as np
import soundfile
import librosa
import librosa.display
import torch
import torchaudio
import matplotlib as mpl
from matplotlib import pyplot as plt
from torchaudio import transforms as AT
from torchvision.transforms import functional as TF
mpl.rcParams['image.cmap'] = 'jet'


learn = load_learner('ConvNext.pkl')
categories = learn.dls.vocab

def get_spectrogram(
    path,
    n_fft=1024,
    win_len=1024,
    hop_len=512,
    power=2.0,
):
    waveform, sample_rate = torchaudio.load(path)
    resample_transform = AT.Resample(orig_freq=sample_rate, new_freq=16000)
    resampled_waveform = resample_transform(waveform)
    spec = AT.MelSpectrogram(
        n_fft=n_fft,
        win_length=win_len,
        hop_length=hop_len,
        f_min = 20,
        f_max = 8000,
        center=True,
        pad_mode="reflect",
        power=power,
    )(resampled_waveform[0].unsqueeze(0))
    spec_db = librosa.power_to_db(np.array(spec.squeeze(0)), ref=np.max, top_db=80)
    return torch.tensor(spec_db).unsqueeze(0)

def classify(audio_path):
    S_db = get_spectrogram(audio_path)
    fig = plt.figure(figsize=(500 / 100, 500 / 100), dpi=100)
    fig, axs = plt.subplots(1, 1)
    plt.axis('off')
    bytedata = (((np.array(S_db.squeeze(0)) + 80) * 255 / 80).clip(0, 255) + 0.5).astype(np.uint8)
    im = Image.fromarray(bytedata)
    resolution = np.array(im).shape
    axs.imshow(im, origin="lower", aspect="auto", extent=(0, resolution[1], 0, resolution[0]))
    dest = Path('tmp.png')
    plt.savefig(dest, bbox_inches='tight', dpi=100)
    pred, idx, prob = learn.predict(dest)
    return dict(zip(categories, map(float, prob)))
