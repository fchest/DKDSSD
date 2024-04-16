from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_feeder import load_data, collate_fn_pad
import numpy as np
import feature_extract
import math
from torch.nn.utils.rnn import pad_sequence
from torch.autograd import Variable
from torch.nn import Parameter
from tqdm import tqdm
import soundfile as sf
from model_logit_inter1 import CRN, se_resnet34_fusion
from loss import A_softmax

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

ids = [0]

torch.set_default_tensor_type(torch.FloatTensor)


eval_protocol = "/data4/dingmingming/data_human/ASVspoof/ASVspoof2019data/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"


def feval(model_asv, model_enhance, out_fold, feature_type, noise_wav_files, device):
    model_asv.eval()
    model_enhance.eval()
    name=os.path.join(out_fold, "test_2019.txt")
    result=[]
    scp=[]
    with torch.no_grad():
        wav_list, folder_list, flag, folder=load_data("eval", eval_protocol, mode="eval", feature_type="fft")
        '''
        frames=len(data)
        data=data.to(device)
        for idx in range(frames):
            output=model(data[idx])
            output=np.mean(output,axis=0)
            result.append(output)
        '''
        for idx in tqdm(range(len(wav_list)), desc="evaluating"):
            wav_id=wav_list[idx]
            scp.append(wav_id)
            wav_path = "{}{}.wav".format(folder, wav_id)
            wavform, sr = sf.read(wav_path, dtype="float32")
            # each_noisy_data [F,T]   noisy_phase[F,T]
            each_noisy_data, noisy_phase, noisy_speech = feature_extract.extract_FFT_enhance(wavform)
            noisy_phase = torch.as_tensor(noisy_phase).to(device)
            noisy=np.reshape(each_noisy_data,(-1,1,161,600))
            noisy_speech = torch.as_tensor(noisy_speech).to(device)
            noisy=torch.as_tensor(noisy).to(device)
            # enhanced_mag[1,1,F,T]
            enhanced_mag = model_enhance(noisy)
            # [1,1,F,T] -> [F,T]
            enhanced_mag = enhanced_mag.squeeze(0).squeeze(1)
            enhanced_d = enhanced_mag * noisy_phase
            
            enhanced = torch.istft(
                enhanced_d,
                n_fft=320,
                hop_length=160,
                win_length=320,
                window=torch.hann_window(320).to(device))     # [1,T]
            T_length = enhanced.shape[1]
            noisy_speech = noisy_speech[:T_length].unsqueeze(0)
            enhanced_lowmag = feature_extract.extract_after_enhance(enhanced, device)
            noisy_lowmag = feature_extract.extract_after_enhance(noisy_speech, device)
            feature_fusion = torch.cat((enhanced_lowmag, noisy_lowmag), 1)
            asv_output = model_asv(feature_fusion)
            output=asv_output[0]

            result.append(output.cpu().numpy().ravel())
        result=np.reshape(result,(-1,2))
        print(result.shape)
        score = (result[:, 1] - result[:, 0])

        with open(name, 'w') as fh:
            for f, cm in zip(scp, score):
                fh.write('{} {} {} {}\n'.format(f, flag[f], folder_list[f], cm))
    print('Eval result saved to {}'.format(name))
        
        #np.savetxt(name,result)
            

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch LCNN ASVspoof')
    parser.add_argument("-o", "--out_fold", type=str, help="output folder", default='./models/')

    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for dev (default: 16)')
    parser.add_argument('--epochs', type=int, default=32, metavar='N',
                        help='number of epochs to train (default: 32)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--warmup', type=float, default=1000, metavar='M')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--feature_type', default='fft')
    parser.add_argument("--gpu", type=str, help="GPU index", default="0")
    parser.add_argument('--noise_scp', default='./noise_scp.scp')
    

    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'batch_size': args.batch_size, 'collate_fn': collate_fn_pad}
    if use_cuda:
        kwargs.update({'num_workers': 2,
                       'pin_memory': True,
                       'shuffle': True},
                     )

    model_asv = se_resnet34_fusion(num_classes=2).to(device)
    model_enhance = CRN().to(device)
    model_asv.load_state_dict(torch.load(os.path.join(args.out_fold, "senet.pt")))
    model_enhance.load_state_dict(torch.load(os.path.join(args.out_fold, "enhance_crn.pt")))
    
    noise_wav_files=[]
    noise_file = open(args.noise_scp, 'r')
    for line in noise_file.readlines():
        line = line.strip()
        list = line.split()
        noise_wav_files.append(list[1])
    feval(model_asv, model_enhance, args.out_fold, args.feature_type, noise_wav_files,device)

if __name__ == '__main__':
    main()
