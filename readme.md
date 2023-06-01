# Lingo3DMol: Generation of a Pocket-based 3D Molecule using a Language Model
Lingo3DMol is a pocket-based 3D molecule generation method that combines the ability of language model with the ability to generate 3D coordinates and geometric deep learning to produce high-quality molecules. 

## Install via conda yaml file
```
conda env create -f environment.yml
conda activate lingo3dmol
```
## Datasets
We provide sample input for sampling under `\dataset` folder.

## Model Checkpoints
Move checkpoint to the checkpoint folder.
https://stonewise-lingo3dmol-public.s3.cn-northwest-1.amazonaws.com.cn/contact.pkl
https://stonewise-lingo3dmol-public.s3.cn-northwest-1.amazonaws.com.cn/gen_mol.pkl
https://stonewise-lingo3dmol-public.s3.cn-northwest-1.amazonaws.com.cn/pretrain.pkl

## Sampling
To inference using the model, run this code:
```
cat {inference input} | python inference.py --cuda {cuda_id} --save_path {path}
```
Example:
```
cat datasets/sample_inference_list | python inference/inference.py --cuda 0 --save_path output/
```
