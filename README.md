# Code for "Quantifying LLM Attention-Head Stability: Implications for Circuit Universality"

This repo contains code + configs to reproduce the main experimental results reported in the paper.  

Model weights for "l8_h8" and "l8_h8_wd" can be accessed from Hugging Face repository: karanbali/attention_head_seed_stability

https://huggingface.co/karanbali/attention_head_seed_stability

The weights can be downloaded using -> download_weights.ipynb  
The code from notebooks expects the weights to be inside a folder named "chkpts".  

# For more information related to scripts & notebooks, please read their respective 'README.md' files.

---

## 1) Quick start to install environment & requirements (recommended)

### A) Create the conda environment
From the root of this unzipped folder:

```bash
# Option 1 (recommended): full env
conda env create -n attn_head_stab -f environment.yml

# If the solver fails, try the base:
# conda env create -n attn_head_stab -f environment_base.yml

conda activate attn_head_stab

# Install 'pip' requirements
python -m pip install -r requirements_pip.txt
