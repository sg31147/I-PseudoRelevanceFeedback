# ⚕️PSEUDO-RELEVANCE FEEDBACK ON DEEP LANGUAGE MODELS FOR MEDICAL DOCUMENT SUMMARIZATION

Official source code repository for the SIGIR 2023 paper [Automated Medical Coding on MIMIC-III and MIMIC-IV: A Critical Review and Replicability Study](https://dl.acm.org/doi/10.1145/3539618.3591918)


```bibtex
@inproceedings{edinAutomatedMedicalCoding2023,
  address = {Taipei, Taiwan},
  title = {Automated {Medical} {Coding} on {MIMIC}-{III} and {MIMIC}-{IV}: {A} {Critical} {Review} and {Replicability} {Study}},
  isbn = {978-1-4503-9408-6},
  shorttitle = {Automated {Medical} {Coding} on {MIMIC}-{III} and {MIMIC}-{IV}},
  doi = {10.1145/3539618.3591918},
  booktitle = {Proceedings of the 446th {International} {ACM} {SIGIR} {Conference} on {Research} and {Development} in {Information} {Retrieval}},
  publisher = {ACM Press},
  author = {Edin, Joakim and Junge, Alexander and Havtorn, Jakob D. and Borgholt, Lasse and Maistro, Maria and Ruotsalo, Tuukka and Maaløe, Lars},
  year = {2023}
}
```

## Update
We released a new [paper](https://arxiv.org/pdf/2406.08958) and [repository](https://github.com/JoakimEdin/explainable-medical-coding/tree/main) for explainable medical coding. The new repository offers the following:
- **Explainability**: Multiple feature attribution methods and metrics for multi-label classification. 
- **Implementation of a modified PLM-ICD**: We have fixed the problem of PLM-ICD occasionally collapsing during training.
- **Huggingface Datasets**: we implemented MIMIC-III, IV, and MDACE as HuggingFace datasets.
- **Inference code**: We provide code for inference without needing the training dataset.
The new repository no longer supports CNN, Bi-GRU, CAML, LAAT, and MultiResCNN.

## Introduction 
Automatic medical coding is the task of automatically assigning diagnosis and procedure codes based on discharge summaries from electronic health records. This repository contains the code used in the paper Automated medical coding on MIMIC-III and MIMIC-IV: A Critical Review and Replicability Study. The repository contains code for training and evaluating medical coding models and new splits for MIMIC-III and the newly released MIMIC-IV. The following models have been implemented:

| Model | Paper | Original Code |
| ----- | ----- | ------------- |
| CNN,Bi-GRU,CAML,MultiResCNN,LAAT,PLM-ICD   |[Automated Medical Coding on MIMIC-III and MIMIC-IV: A Critical Review and Replicability Study](https://arxiv.org/abs/2304.10909) | [link](https://github.com/JoakimEdin/medical-coding-reproducibility) | 
| CNN,Bi-GRU,CAML,MultiResCNN,LAAT,PLM-ICD   |[Automated Medical Coding on MIMIC-III and MIMIC-IV: A Critical Review and Replicability Study](https://arxiv.org/abs/2304.10909) | [link](https://github.com/JoakimEdin/medical-coding-reproducibility) | 
| CNN,Bi-GRU,CAML,MultiResCNN,LAAT,PLM-ICD   |[Automated Medical Coding on MIMIC-III and MIMIC-IV: A Critical Review and Replicability Study](https://arxiv.org/abs/2304.10909) | [link](https://github.com/JoakimEdin/medical-coding-reproducibility) | 
| CNN,Bi-GRU,CAML,MultiResCNN,LAAT,PLM-ICD   |[Automated Medical Coding on MIMIC-III and MIMIC-IV: A Critical Review and Replicability Study](https://arxiv.org/abs/2304.10909) | [link](https://github.com/JoakimEdin/medical-coding-reproducibility) | 
| CNN,Bi-GRU,CAML,MultiResCNN,LAAT,PLM-ICD   |[Automated Medical Coding on MIMIC-III and MIMIC-IV: A Critical Review and Replicability Study](https://arxiv.org/abs/2304.10909) | [link](https://github.com/JoakimEdin/medical-coding-reproducibility) | 
| CNN,Bi-GRU,CAML,MultiResCNN,LAAT,PLM-ICD   |[Automated Medical Coding on MIMIC-III and MIMIC-IV: A Critical Review and Replicability Study](https://arxiv.org/abs/2304.10909) | [link](https://github.com/JoakimEdin/medical-coding-reproducibility) | 

The splits are found in `files/data`. The splits are described in the paper.

## How to reproduce results
### Setup Conda environment
1. Create a conda environement `conda create -n coding python=3.10`
2. Install the packages `pip install . -e`


### Prepare MIMIC-IV
This code has been developed on MIMIC-IV and MIMIC-IV v2.2. 
1. Download MIMIC-IV and MIMIC-IV-NOTE into your preferred location `path/to/mimiciv` and `path/to/mimiciv-note`. Please note that you need to complete training to acces the data. The training is free, but takes a couple of hours.  - [mimiciv](https://physionet.org/content/mimiciv/2.2/) and [mimiciv-note](https://physionet.org/content/mimic-iv-note/2.2/)
2. Open the file `src/settings.py`
3. Change the variable `DOWNLOAD_DIRECTORY_MIMICIV` to the path of your downloaded data `./dataset/mimiciv`
4. Change the variable `DOWNLOAD_DIRECTORY_MIMICIV_NOTE` to the path of your downloaded data `./dataset/mimiciv-note`
