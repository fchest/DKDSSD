from __future__ import print_function
import itertools
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data_feeder import ASVDataSet, load_data, collate_fn_pad
from torch.utils.data import DataLoader
import soundfile as sf
from feature_extract import extract_after_enhance
from model_logit_inter1 import CRN, se_resnet34, se_resnet34_fusion, ScheduledOptim
from loss import A_softmax, enhance_loss_function


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

ids = [0]
i = 0

train_protocol = "/data4/dingmingming/data_human/ASVspoof/ASVspoof2019data/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trl.txt"
dev_protocol = "/data4/dingmingming/data_human/ASVspoof/ASVspoof2019data/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"
eval_protocol = "/data4/dingmingming/data_human/ASVspoof/ASVspoof2019data/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"


def train(args, model_asv,  model_clean_teacher, model_enhance, device, train_loader, optimizer_asv, optimizer_enhance, epoch):
    model_asv.train()
    model_enhance.train()
    model_clean_teacher.train()
    # noisy_data  clean_data  noisy_phase  [B,F,T]
    for batch_idx, (noisy_data, target, clean_data, n_frames, noisy_phase, clean_speech, noisy_speech) in enumerate(train_loader):
        optimizer_asv.zero_grad()
        optimizer_enhance.zero_grad()
        
        noisy, clean = noisy_data.unsqueeze(1).to(device), clean_data.unsqueeze(1).to(device)   # [B,1,F,T]
        target, n_frames = target.to(device), n_frames.to(device)
        noisy_phase = noisy_phase.to(device)    #[B,F,T]
        noisy_speech = noisy_speech.to(device)
        clean_speech = clean_speech.to(device)
        
        enhanced_mag = model_enhance(noisy)
        enhance_loss = enhance_loss_function(enhanced_mag, clean, n_frames, device)
        enhanced_mag = enhanced_mag.squeeze(1)  # [B,1,F,T] -> [B,F,T]
        noisy = noisy.squeeze(1)
        enhanced_d = enhanced_mag * noisy_phase
        enhanced = torch.istft(
                enhanced_d,
                n_fft=320,
                hop_length=160,
                win_length=320,
                window=torch.hann_window(320).to(device))     # [B,T]
        
        T_length = enhanced.shape[1]
        noisy_speech = noisy_speech[:,:T_length]
        clean_speech = clean_speech[:,:T_length]

        enhanced_lowmag = extract_after_enhance(enhanced, device)
        noisy_lowmag = extract_after_enhance(noisy_speech, device)
        clean_lowmag = extract_after_enhance(clean_speech, device)
        feature_fusion = torch.cat((enhanced_lowmag, noisy_lowmag), 1)      # torch.Size([B, 2, 432, 600])
        asv_output_enhanced = model_asv(feature_fusion)
        asv_output_clean = model_clean_teacher(clean_lowmag)
        criterion=A_softmax()
        asv_hard_loss = criterion(asv_output_enhanced, target)
        clean_teacher_loss = criterion(asv_output_clean, target)
        T = 3
        lambda_ = 0.05
        soft_loss_KD = nn.KLDivLoss()(F.log_softmax(asv_output_enhanced[0] / T, dim=1), 
                                        F.softmax(asv_output_clean[0] / T, dim=1))
        asv_loss = (1 - lambda_) * asv_hard_loss + lambda_ * T * T * (soft_loss_KD)
        loss = enhance_loss + asv_loss + clean_teacher_loss
        loss.backward()

        optimizer_enhance.step()
        optimizer_asv.step()
        lr=optimizer_asv.update_learning_rate()
        global i
        if batch_idx % args.log_interval == 0:
            
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tasv_hard_loss: {:.6f}\tsoft_loss_KD: {:.6f}\tenhance_loss: {:.6f}'.format(
                epoch, batch_idx * len(noisy_data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), asv_hard_loss.item(), soft_loss_KD.item(), enhance_loss.item()))

            speech = enhanced[:1,:].squeeze(0).detach().cpu().numpy()
            trypath = os.path.join(args.out_fold, "try")
            sf.write( trypath + "/{}.wav".format(i), speech, 16000)
            i=i+1
            if args.dry_run:
                break


def test(model_asv, model_enhance, device, test_loader):
    model_asv.eval()
    model_enhance.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for noisy_data, target, clean_data, n_frames, noisy_phase, clean_speech, noisy_speech in test_loader:
            noisy = noisy_data.unsqueeze(1).to(device)   # [B,1,F,T]
            target, n_frames = target.to(device), n_frames.to(device)
            noisy_phase = noisy_phase.to(device)
            noisy_speech = noisy_speech.to(device)
            
            enhanced_mag = model_enhance(noisy)
            enhanced_mag = enhanced_mag.squeeze(1)  # [B,1,F,T] -> [B,F,T]
            enhanced_d = enhanced_mag * noisy_phase
            
            enhanced = torch.istft(
                    enhanced_d,
                    n_fft=320,
                    hop_length=160,
                    win_length=320,
                    window=torch.hann_window(320).to(device))     # [B,T]
            T_length = enhanced.shape[1]
            noisy_speech = noisy_speech[:,:T_length]
            enhanced_lowmag = extract_after_enhance(enhanced, device)
            noisy_lowmag = extract_after_enhance(noisy_speech, device)
            feature_fusion = torch.cat((enhanced_lowmag, noisy_lowmag), 1)
            asv_output = model_asv(feature_fusion)
            criterion=A_softmax()
            asv_hard_loss = criterion(asv_output, target)
            #loss = enhance_loss + asv_loss
            test_loss+=asv_hard_loss.item()

            result=asv_output[0]
            #result=asv_output
            pred = result.argmax(dim=1, keepdim=True) 
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss

            


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch LCNN ASVspoof')

    parser.add_argument("-o", "--out_fold", type=str, help="output folder", default='./models/')
    
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                        help='input batch size for dev (default: 16)')
    parser.add_argument('--epochs', type=int, default=32, metavar='N',
                        help='number of epochs to train (default: 9)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--lr_enhance', type=float, default=0.0006, metavar='LR',
                        help='learning rate (default: 0.0006)')
    parser.add_argument('--warmup', type=float, default=1000, metavar='M')
    #parser.add_argument('--keep_prob', type=float, default=1.0)
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',       # 100
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--feature_type', default='fft')
    parser.add_argument('--noise_scp', default='./noise_scp.scp')
    
    parser.add_argument("--gpu", type=str, help="GPU index", default="0")
    
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda:0" if use_cuda else "cpu")
    
    if not os.path.exists(args.out_fold):
        os.makedirs(args.out_fold)
    if not os.path.exists(os.path.join(args.out_fold, 'checkpoint')):
        os.makedirs(os.path.join(args.out_fold, 'checkpoint'))
    if not os.path.exists(os.path.join(args.out_fold, 'try')):
        os.makedirs(os.path.join(args.out_fold, 'try'))

    kwargs = {'batch_size': args.batch_size, 'collate_fn': collate_fn_pad}
    if use_cuda:
        kwargs.update({'num_workers': 2,
                       'pin_memory': True,
                       'shuffle': True},
                     )

    # train_data 是音频全路径
    train_data, train_label=load_data("train", train_protocol, mode="train")
    train_dataset=ASVDataSet(train_data, train_label, args.noise_scp, mode="train")
    train_dataloader=DataLoader(train_dataset, **kwargs)

    dev_data, dev_label=load_data("dev", dev_protocol, mode="dev")
    dev_dataset=ASVDataSet(dev_data, dev_label, args.noise_scp, mode="train")
    dev_dataloader=DataLoader(dev_dataset, **kwargs)


    model_asv = se_resnet34_fusion(num_classes=2).to(device)
    model_enhance = CRN().to(device)
    model_clean_teacher = se_resnet34(num_classes=2).to(device)
    param_groups = itertools.chain(model_asv.parameters(), model_clean_teacher.parameters())
    optimizer_asv = ScheduledOptim(optim.Adam(
        filter(lambda p: p.requires_grad, param_groups),
        betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, amsgrad=True),
        args.warmup)

    
    optimizer_enhance = torch.optim.Adam(
        params=model_enhance.parameters(),
        lr=args.lr_enhance,
        betas=(0.9, 0.999)
    )
    loss=10
    ploss=1


    for epoch in range(1, args.epochs + 1):

        train(args, model_asv,  model_clean_teacher, model_enhance, device, train_dataloader, optimizer_asv, optimizer_enhance, epoch)
        loss=test(model_asv, model_enhance, device, dev_dataloader)
        torch.save(model_asv.state_dict(), os.path.join(args.out_fold, 'checkpoint','senet_epoch_%d.pt' % epoch))
        torch.save(model_enhance.state_dict(), os.path.join(args.out_fold, 'checkpoint','enhance_crn_epoch_%d.pt' % epoch))

        if args.save_model:
            if loss<ploss:
                ploss=loss
                
                torch.save(model_asv.state_dict(), os.path.join(args.out_fold, 'senet.pt'))
                torch.save(model_enhance.state_dict(), os.path.join(args.out_fold, 'enhance_crn.pt'))
                print("model saved")

if __name__ == '__main__':
    main()
