# Code for "Quantifying LLM Attention-Head Stability: Implications for Circuit Universality"

This repo contains code + configs to reproduce the main experimental results reported in the paper.  

Model weights for "l8_h8" and "l8_h8_wd" can be accessed from Hugging Face repository: karanbali/attention_head_seed_stability

https://huggingface.co/karanbali/attention_head_seed_stability

The weights can be downloaded using -> download_weights.ipynb  
The code from notebooks expects the weights to be inside a folder named "chkpts".  

# For more information related to scripts & notebooks, please read their respective 'README.md' files.

## This directory "./notebooks" holds 8 notebooks, 4 prompt sets (as pickle files), and a utility python file (model_configs.py)

#### 100_prompts.pkl - Set of 100 prompts, used for almost all notebooks except "Stability_Prompts_Length.ipynb"  

#### 50_prompts.pkl - Set of 20 prompts, used for "Stability_Prompts_Length.ipynb"  

#### 40_prompts.pkl - Set of 20 prompts, used for "Stability_Prompts_Length.ipynb"  

#### 30_prompts.pkl - Set of 20 prompts, used for "Stability_Prompts_Length.ipynb"  

#### 20_prompts.pkl - Set of 20 prompts, used for "Stability_Prompts_Length.ipynb"  

#### 10_prompts.pkl - Set of 20 prompts, used for "Stability_Prompts_Length.ipynb"  

#### 5_prompts.pkl - Set of 20 prompts, used for "Stability_Prompts_Length.ipynb" 

---

#### Notebook 1 — notebooks/Stability_Attention_Head.ipynb  

Paper mapping: Use the notebook to generate the following results from the paper:  
    4.1. Head-wise and layer-wise stability   
    4.4.1. CORRELATION BETWEEN QUERY-WEIGHT NORM AND LAYER-WISE STABILITY   
    4.8. Layer-wise correlation between the stability and post-ablation change in perplexity  

#### Notebook 2 — notebooks/Stability_Cross_Layers.ipynb  

Paper mapping: Use the notebook to generate the following results from the paper:  
    4.2. Cross-layer best-match stability  

#### Notebook 3 — notebooks/Uniqueness_Attention_Head.ipynb  

Paper mapping: Use the notebook to generate the following results from the paper:  
    4.3. Within-layer uniqueness of attention heads  

#### Notebook 4 — notebooks/Stability_Prompts_Length.ipynb

Paper mapping: Use the notebook to generate the following results from the paper:  
    4.4.2. EFFECT OF PROMPT LENGTH ON STABILITY  

#### Notebook 5 — notebooks/Stability_Adam_vs_AdamW.ipynb  

Paper mapping: Use the notebook to generate the following results from the paper:  
    4.5. Stability comparison: Adam vs AdamW - Results (Fig. 6)  
    4.5. Stability comparison: Adam vs AdamW - Mechanistic check (Fig. B.6.1)  

#### Notebook 6 — notebooks/Perplexity_Adam_vs_AdamW.ipynb  

Paper mapping: Use the notebook to generate the following results from the paper:  
    4.5. Stability comparison: Adam vs AdamW - Performance Parity (§ B.7)  

#### Notebook 7 — notebooks/Stability_Residual_Stream.ipynb  

Paper mapping: Use the notebook to generate the following results from the paper:  
    4.7. Stability of residual stream  

#### Notebook 8 — notebooks/meta_SNE_avg_all_configs_attn.ipynb   

Paper mapping: Use the notebook to generate the following results from the paper:  
    4.9. Geometric observation of attention head activations using meta-SNE  
