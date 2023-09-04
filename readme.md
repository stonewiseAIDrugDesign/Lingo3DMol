# Lingo3DMol: Generation of a Pocket-based 3D Molecule using a Language Model
Lingo3DMol is a pocket-based 3D molecule generation method that combines the ability of language model with the ability to generate 3D coordinates and geometric deep learning to produce high-quality molecules. 

## System Requirements

### Hardware requirements
A standard computer with GPU with at least 5GB graphic memory.

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
https://stonewise-lingo3dmol-public.s3.cn-northwest-1.amazonaws.com.cn/contact.pkl md5sum:6a9313726141fcf9201b9b9470dc2a7e

https://stonewise-lingo3dmol-public.s3.cn-northwest-1.amazonaws.com.cn/gen_mol.pkl md5sum:452bd401667184ae43c9818e5bdb133b

https://stonewise-lingo3dmol-public.s3.cn-northwest-1.amazonaws.com.cn/pretrain.pkl md5sum:5d63db855c1922c836a78c3c7300f930

## Sampling
To inference using the model on DUD-E set, run this code:
```
sh run.sh
```

## Expected output
The output should be generated molecules in mol format.

## License
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.