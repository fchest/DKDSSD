import librosa
import numpy as np
from scipy import signal
import torch
import torchaudio
#import matplotlib.pyplot as plt
import soundfile as sf
# parameters
sample_rate = 16000
win_length = 1728
hop_length = 130

def extract(wav_path, feature_type):
    if feature_type == "cqt":
        return extract_cqt(wav_path)
    if feature_type == "fft":
        return extract_fft(wav_path)

def extract_cqt(wav_path):
    y, fs = librosa.load(wav_path, sr=sample_rate)
    cqt = librosa.cqt(y, fs, hop_length=hop_length, fmin=fs/(2^10), bins_per_octave=96)
    total=cqt.shape[1]
    if total<400:
        for j in range(0,400//total):
            if j==0:
                cqt=np.hstack((cqt,np.fliplr(cqt)))
            else:
                cqt=np.hstack((cqt,cqt))
    feature=cqt[:,0:400]
    #feature=feature[432:865,:]
    return np.reshape(np.array(feature),(-1,84,400))

def extract_fft(wav_path):
    p_preemphasis = 0.97
    num_freq = 1728
    

    def preemphasis(x):
        return signal.lfilter([1, -p_preemphasis], [1], x)

    def _stft(y):
        return librosa.stft(y=y, n_fft=num_freq, hop_length=hop_length, win_length=win_length, window=signal.windows.blackman)

    y, _ = librosa.load(wav_path, sr=None)
    if(len(y)>80000):
        y=y[0:80000]
    #print(y.shape)
    D = _stft(preemphasis(y))
    #print(D.shape)
    S = np.log(abs(D)+np.exp(-80))
    #feature=S
    #height=S.shape[0]
    total=S.shape[1]
    if total<600:
        for j in range(0,600//total):
            if j==0:
                S=np.hstack((S,np.fliplr(S)))
            else:
                S=np.hstack((S,S))
                if S.shape[1]>=600:
                    break
    feature=S[:,0:600]

    return np.array(feature)

def extract_FFT_enhance(data):
    
    num_freq = 320
    win_length = 320
    hop_length = 160
    
    def _stft(y):
        return librosa.stft(y=y, n_fft=num_freq, hop_length=hop_length, win_length=win_length, window="hann")
    
    while len(data)<96000:
        data = np.concatenate((data, data), axis=None)
    D = _stft(data)
    frame = D.shape[-1]
    D = D[:,0:600]    
    

       
    mag, phase = librosa.magphase(D)
    # phase = np.angle(phase)
    # print(D.shape, mag.shape, phase.shape)
    return np.array(mag), np.array(phase), np.array(data)



def extract_after_enhance(data, device):
    num_freq = 1728
    sample_rate = 16000
    win_length = 1728
    hop_length = 130
    E = 1e-30
    if(data.shape[-1]>80000):
        data=data[:,0:80000]

    noisy_d = torch.stft(
                data,
                n_fft=num_freq,
                hop_length=hop_length,
                win_length=win_length,
                window=torch.blackman_window(win_length).to(device),return_complex=True)  # [B, F, T, 2]
    
    # noisy_d = torch.view_as_complex(noisy_d)

    abs = torch.abs(noisy_d)
    abs[abs <= 1e-30] = 1e-30


    noisy_mag = torch.log(abs)
    
    feature = noisy_mag[:,:,0:600]
    feature=feature[:,:432,:]

    return feature.unsqueeze(1).to(device) # [B,1,T,F]

def main():
    #c=extract_cqt('LA_T_1000137.flac')
    #print(c.shape)
    # r=extract_FFT('/data4/dingmingming/data_human/ASVspoof_noise/ASVspoof2019_LA_train_only_noisy/LA_T_1000137_-5_1.wav')

    y, _ = librosa.load('/data4/dingmingming/data_human/ASVspoof_noise/ASVspoof2019_LA_train_only_noisy/LA_T_1000137_-5_1.wav', sr=None)
    print(y.shape)
    r=extract_FFT_enhance(y)

if __name__ == '__main__':
    main()

