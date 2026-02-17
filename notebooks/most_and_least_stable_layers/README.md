# Code for "Quantifying LLM Attention-Head Stability: Implications for Circuit Universality"

This repo contains code + configs to reproduce the main experimental results reported in the paper.  

Model weights for "l8_h8" and "l8_h8_wd" can be accessed from Hugging Face repository: karanbali/attention_head_seed_stability

https://huggingface.co/karanbali/attention_head_seed_stability

The weights can be downloaded using -> download_weights.ipynb  
The code from notebooks expects the weights to be inside a folder named "chkpts".  

# For more information related to scripts & notebooks, please read their respective 'README.md' files.

## In order to get results for ''4.6. Most- and least-stable layers'', first create dataframes holding the values for "Attention-Head Stability" for all architectures using -> notebooks/create_df_avg_sim.ipynb

#### Notebook 1 — notebooks/create_df_avg_sim.ipynb  

** Need to run this notebook for all architectures in order to run "Stability_Cross_Layers.ipynb"  


#### Notebook 2 — notebooks/Stability_Cross_Layers.ipynb  

Paper mapping: Use the notebook to generate the following results from the paper:  
    4.6. Most- and least-stable layers  

