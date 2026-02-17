# Code for "Quantifying LLM Attention-Head Stability: Implications for Circuit Universality"

This repo contains code + configs to reproduce the main experimental results reported in the paper.  

Model weights for "l8_h8" and "l8_h8_wd" can be accessed from Hugging Face repository: karanbali/attention_head_seed_stability

https://huggingface.co/karanbali/attention_head_seed_stability

The weights can be downloaded using -> download_weights.ipynb  
The code from notebooks expects the weights to be inside a folder named "chkpts".  

# For more information related to scripts & notebooks, please read their respective 'README.md' files.

It's assumed that the environment is already installed.

We train on two corpora:  
• C4 (2B token subset): All models with 2, 4, or 8 layers are pretrained on a 2-billion token subset of C4. The dataset is
available at:  
https://huggingface.co/datasets/NeelNanda/c4-code-tokenized-2b  

• OpenWebText (9B tokens): All 12-layer models are pretrained on 9 billion tokens on OpenWebText, an open-source
replication of OpenAI’s WebText used for GPT-2. The dataset is available at:  
http://skylion007.github.io/OpenWebTextCorpus

---

#### Script 1 — scripts/train_model.py   

Paper mapping:  
    The training script used to train 2, 4, and 8 layered models on C4 dataset.  
    Additional SLURM scipt (scripts/script_train_model.sh) to use this file.

#### Script 2 — scripts/train_gpt2_shards.py   

Paper mapping:  
    The training script used to train 12 layered (GPT2-small) models on OpenWebText dataset.
    OpenWebText dataset is divided into 10 shards, which is done to adjust to varying available computational capabilities while training refits.  
    Additional SLURM scipt (scripts/script_train_gpt2_shards.sh) to use this file.
