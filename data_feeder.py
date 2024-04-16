import numpy as np
import librosa
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import scipy.io as scio
import soundfile as sf
from sklearn.preprocessing import scale
import feature_extract
import os, random
from torch.nn.utils.rnn import pad_sequence

torch.set_default_tensor_type(torch.FloatTensor)

def collate_fn_pad(batch):
        """
        Returns:
            [B, F, T (Longest)]
        """
        noisy_list = []
        noisy_phase_list = []
        clean_list = []
        n_frames_list = []
        clean_speech_list = []
        noisy_speech_list = []
        label_list = []

        for noisy, label, clean, noisy_phase, clean_speech, noisy_speech in batch:
            noisy_list.append(torch.as_tensor(noisy).permute(1, 0))  # [F, T] => [T, F]
            clean_list.append(torch.as_tensor(clean).permute(1, 0))  # [1, T] => [T, 1]
            noisy_phase_list.append(torch.as_tensor(noisy_phase).permute(1, 0)) # [F, T] => [T, F]
            clean_speech_list.append(clean_speech)
            noisy_speech_list.append(noisy_speech) 
            frame = 600
            n_frames_list.append(frame)
            label_list.append(label)
            
        # seq_list = [(T1, F), (T2, F), ...]
        #   item.size() must be (T, *)
        #   return (longest_T, len(seq_list), *)
        noisy_list = pad_sequence(noisy_list).permute(1, 2, 0)  # ([T1, F], [T2, F], ...) => [T, B, F] => [B, F, T]
        clean_list = pad_sequence(clean_list).permute(1, 2, 0)  # ([T1, 1], [T2, 1], ...) => [T, B, 1] => [B, 1, T]
        noisy_phase_list = pad_sequence(noisy_phase_list).permute(1, 2, 0)    # # ([T1, F], [T2, F], ...) => [T, B, F] => [B, F, T]
        noisy_speech_list = pad_sequence(noisy_speech_list).permute(1, 0)  # [T, B] -> [B, T] 以长的语音长度为T
        noisy_speech_list = torch.as_tensor(noisy_speech_list)
        clean_speech_list = pad_sequence(clean_speech_list).permute(1, 0)  # [T, B] -> [B, T] 以长的语音长度为T
        clean_speech_list = torch.as_tensor(clean_speech_list)
        n_frames_list = torch.as_tensor(n_frames_list)
        label_list = torch.as_tensor(label_list)

        return noisy_list, label_list, clean_list, n_frames_list, noisy_phase_list, clean_speech_list, noisy_speech_list


def load_label(label_file):
    labels = {}
    wav_lists = []
    #folder_list={}
    encode = {'spoof': 0, 'bonafide': 1}
    with open(label_file, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            if len(line) > 1:
                wav_id = line[1]
                wav_lists.append(wav_id)
                tmp_label = encode[line[4]]
                labels[wav_id] = tmp_label
    return labels, wav_lists


def add_noise(clean_path, noise_path, snr):

        clean, sr = sf.read(clean_path, dtype="float32")
        noise, sr = sf.read(noise_path, dtype="float32")

        if len(clean) > len(noise):
            # 极少数情况，噪声比纯净语音短。此时需要将噪声重复多次，直到可以达到纯净语音的长度
            pad_factor = (len(clean) // len(noise))  # 拓展系数为需要拓展的次数，不包括原有的
            padded_noise = noise
            for i in range(pad_factor):
                padded_noise = np.concatenate((padded_noise, noise))
            noise = padded_noise[:len(clean)]
            assert len(clean) == len(noise), f"clean = {len(clean)}, noise = {len(noise)}"
            mixture = add_noise_for_waveform(clean, noise, snr)
        elif len(clean) < len(noise):
            # random crop
            start = random.randint(0, len(noise) - len(clean) - 1)
            end = start + len(clean)
            noise = noise[start:end]
            assert len(clean) == len(noise), f"clean = {len(clean)}, noise = {len(noise)}"
            mixture = add_noise_for_waveform(clean, noise, snr)
        else:
            mixture = add_noise_for_waveform(clean, noise, snr)
        return mixture, snr

def add_noise_for_waveform(s, n, db):
    if np.sum(n ** 2) == 0:
        mix = s
    else:
        alpha = np.sqrt(
            np.sum(s ** 2) / (np.sum(n ** 2) * 10 ** (db / 10))
        )
        mix = s + alpha * n
    return mix


class ASVDataSet(Dataset):

    def __init__(self, data, label, noise_scp, wav_ids=None, transform=True, mode="train", lengths=None, feature_type="fft"):
        super(ASVDataSet, self).__init__()
        self.data = data
        self.label = label
        self.wav_ids = wav_ids
        self.transform = transform
        self.lengths = lengths
        self.mode = mode
        self.feature_type=feature_type
        
        self.noise_wav_files = []
        noise_file = open(noise_scp, 'r')
        for line in noise_file.readlines():
            line = line.strip()
            list = line.split()
            self.noise_wav_files.append(list[1])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        each_data, each_label = self.data[idx], self.label[idx]
        noise_path = random.sample(self.noise_wav_files, 1)[0]
        snr = random.uniform(0, 20)
        wavform, snr = add_noise(each_data, noise_path, snr)
        clean_data, sr = sf.read(each_data, dtype="float32")
        
        each_clean_data, clean_phase, clean_speech = feature_extract.extract_FFT_enhance(clean_data)  # (161,600) (F,T)
        each_noisy_data, noisy_phase, noisy_speech = feature_extract.extract_FFT_enhance(wavform)     # (161,600) (F,T)
        if self.transform:
            each_noisy_data=torch.as_tensor(each_noisy_data)
            each_clean_data=torch.as_tensor(each_clean_data)
            noisy_phase=torch.as_tensor(noisy_phase)
            noisy_speech = torch.as_tensor(noisy_speech)    # T
            clean_speech = torch.as_tensor(clean_speech)
            
        return each_noisy_data, each_label, each_clean_data, noisy_phase, clean_speech, noisy_speech
    


# this will load data by wav
def load_data(dataset, label_file, mode="train", feature_type="fft"):
    if mode!="eval":
        # clean_data 全路径
        clean_data, label=load_train_data(dataset, label_file, feature_type="fft")
        return clean_data, label
    else:
        data, folder_list, flag = load_eval_data(dataset, label_file, feature_type="fft")
        folder = "/data4/dingmingming/data_human/ASVspoof/ASVspoof2019data/ASVspoof2019_LA_eval/"
        return data, folder_list, flag, folder


def load_train_data(dataset, label_file, feature_type="fft"):
    #labels, wav_lists, folder_list = load_label(label_file)
    labels, wav_lists = load_label(label_file)
    final_data = []
    final_label = []

    for wav_id in tqdm(wav_lists, desc="load {} data".format(dataset)):
        label = labels[wav_id]
        
        if "T" in wav_id:
            wav_path = "/data4/dingmingming/data_human/ASVspoof/ASVspoof2019data/ASVspoof2019_LA_train/{}.wav".format(wav_id)
        if "D" in wav_id:
            wav_path = "/data4/dingmingming/data_human/ASVspoof/ASVspoof2019data/ASVspoof2019_LA_dev/{}.wav".format(wav_id)
        if os.path.exists(wav_path):
            final_data.append(wav_path)
            final_label.append(label)
        else:
            print("can not open {}".format(wav_path))
            
    return final_data, final_label

def load_eval_data(dataset, scp_file, feature_type="fft"):
    wav_lists = []
    folder_list={}
    flag = {}
    with open(scp_file, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            if len(line) > 1:
                wav_id = line[1]
                wav_lists.append(wav_id)
                folder_list[wav_id]=line[-1]
                if line[-2] == '-':
                    flag[wav_id] = 'A00'
                else:
                    flag[wav_id] = line[-2]
    return wav_lists, folder_list, flag

def main():
    labels, wav_lists=load_label("ASVspoof2019.LA.cm.train.trn.txt")
    wav_list=wav_lists[4000:4005]
    for wav_id in wav_list:
        print(wav_id)
        print(labels[wav_id])

if __name__ == '__main__':
    main()
