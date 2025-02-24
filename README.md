# ⚕️PSEUDO-RELEVANCE FEEDBACK ON DEEP LANGUAGE MODELS FOR MEDICAL DOCUMENT SUMMARIZATION

This paper is public on npjmedicine.

```bibtex
@inproceedings{edinAutomatedMedicalCoding2023,
  address = {Taipei, Taiwan},
  title = {PSEUDO-RELEVANCE FEEDBACK ON DEEP LANGUAGE MODELS FOR MEDICAL DOCUMENT SUMMARIZATION},
  isbn = {xxx-x-xxxx-xxxx-x},
  shorttitle = {PSEUDO-RELEVANCE FEEDBACK ON DEEP LANGUAGE MODELS},
  doi = {xx.xx/xxxxxxx.xxxxxxx},
  booktitle = {Medicine Journal},
  publisher = {Npj medicine},
  author = {Kitti Akkhawatthanakun1, Paisarn Muneesawang, Lalita Narupiyakul, and KonlakornWongpatikaseree},
  year = {2025}
}
```


## Introduction 
Automatic medical coding is the task of automatically assigning diagnosis and procedure codes based on discharge summaries from electronic health records. This repository contains the code used in the paper Automated medical coding on MIMIC-III and MIMIC-IV: A Critical Review and Replicability Study. The repository contains code for training and evaluating medical coding models and new splits for MIMIC-III and the newly released MIMIC-IV. The following models have been implemented:

## Associative research paper
<table>
  <tr>
    <th>Model</th>
    <th>Paper</th>
    <th>Original Code</th>
  </tr>
  <tr>
    <td rowspan="6">CNN, Bi-GRU, CAML, MultiResCNN, LAAT, PLM-ICD</td>
    <td rowspan="6">
      <a href="https://arxiv.org/abs/2304.10909">Automated Medical Coding on MIMIC-III and MIMIC-IV: A Critical Review and Replicability Study</a>
    </td>
    <td><a href="https://github.com/JoakimEdin/medical-coding-reproducibility">link</a></td>
  </tr>
  <tr>
    <td><a href="https://github.com/JoakimEdin/medical-coding-reproducibility">link</a></td>
  </tr>
  <tr>
    <td><a href="https://github.com/JoakimEdin/medical-coding-reproducibility">link</a></td>
  </tr>
  <tr>
    <td><a href="https://github.com/JoakimEdin/medical-coding-reproducibility">link</a></td>
  </tr>
  <tr>
    <td><a href="https://github.com/JoakimEdin/medical-coding-reproducibility">link</a></td>
  </tr>
  <tr>
    <td><a href="https://github.com/JoakimEdin/medical-coding-reproducibility">link</a></td>
  </tr>
</table>


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
