# BiLENS
The official codes for [**BiLENS : A Question Guided Framework for Bio-Medical Visual Question Answering**]

## Contents

- [Install](#install)
- [Data Download](#data-download)
- [Model Download](#model-download)
- [Evaluation](#evaluation)
- [Train](#archive)

## Install
1. Clone this repository and navigate to BiLENS folder
```bash
https://github.com/jzy-123/BiLENS.git
cd BiLENS
```
2. Install Package: Create conda environment
```Shell
conda create -n bilens python=3.10 -y
conda activate bilens
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install -r requirements.txt
```
## Data Download
| dataset         | link                                              |
|-----------------|---------------------------------------------------|
| VQA-RAD     | https://osf.io/89kps/                           |
| SLAKE | https://www.med-vqa.com/slake/                  |
| Path-VQA | https://github.com/UCSD-AI4H/PathVQA |
| BioVGQ | 


## Model Download

 Model Descriptions | 🤗 Huggingface Hub | 
| --- | ---: |
| BiLENS | [jzyang/BiLENS](https://huggingface.co/jzyang/BiLENS) |

Click [PMC-CLIP](https://github.com/WeixiongLin/PMC-CLIP) to download the weight parameters of the visual encoder and save them in the ```/models/pmcclip``` folder.

## Evaluation
