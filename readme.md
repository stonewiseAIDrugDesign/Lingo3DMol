# Lingo3DMol: Generation of a Pocket-based 3D Molecule using a Language Model
Lingo3DMol is a pocket-based 3D molecule generation method that combines the ability of language model with the ability to generate 3D coordinates and geometric deep learning to produce high-quality molecules. 

## System Requirements

### Hardware requirements
A standard compute with GPU with at least 5GB graphic memory.

### OS Requirements
This package is supported for macOS and Linux. The package has been tested on the following systems:
Linux: Ubuntu 16.04
macOS: Ventura (13.0)

## Install via conda yaml file
Typical install time: 40 minutes
```
conda create -n lingo3dmol python=3.8
conda activate lingo3dmol
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
conda install rdkit  -c conda-forge
conda install scipy
conda install -c conda-forge tqdm
```
## Datasets
We provide DUD-E pocket files for sampling under `\dataset` folder. Please Unzip`dude_pocket.zip`.

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
For example, to inference on DUD-E set run following code. :
```
cat datasets/dude_files | python inference/inference.py --cuda 0 --save_path output/
```

## Expected output
The output should be generated molecules in PDB format.

## License
