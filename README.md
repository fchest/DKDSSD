# DKDSSD

# Code for paper: Dual-Branch Knowledge Distillation for Noise-Robust Synthetic Speech Detection

# Preprocess
* Please download the noisy dataset NonSpeech for training at the URL: http://web.cse.ohio-state.edu/pnl/corpus/HuNonspeech/HuCorpus.html, and update the noise_scp.scp file.
* To simulate the noisy test set, you need to add noise to the clean test set in advance. The 12 kinds of unseen noise are in the folder NOISEX-92-16K, and the seen data set is also the NonSpeech dataset.

# Run
* Using train.py to train and test the model by test.py
* The server runs instructions in the background：
`nohup sh run.sh > train.log &`

# Requirements
+ Pyhton3.8.18 \
`conda create --name <env> --file requirements.txt`

# Reference
* [JupiterEthan/CRN-causal](https://github.com/JupiterEthan/CRN-causal)

# Citation
If this repo is helpful with your research or projects, please kindly star our repo and cite our paper as follows:  
```
Waiting for publication...
```

