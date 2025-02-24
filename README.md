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
Many specialized language models have been widely adopted for predicting ICD-10 codes. However, there is still a lack of research aimed at improving the accuracy and reliability of these models. In response to this gap, we propose a novel approach to enhance prediction quality by leveraging techniques from information retrieval. Specifically, we apply the Rocchio algorithm to refine and optimize model outputs, building upon existing language models to improve their practical applicability. Furthermore, we have developed a web-based application—a dynamic playground where users can interact with and evaluate our method after following our proposed implementation steps.


![My Image Description](/files/retrieval/pesudo_relevance_feedback.png)


### Associative research paper
<table>
  <tr>
    <th>Model</th>
    <th>Paper</th>
    <th>Original Code</th>
  </tr>
  <tr>
    <td>CNN</td>
    <td rowspan="6">
      <a href="https://arxiv.org/abs/2304.10909">Automated Medical Coding on MIMIC-III and MIMIC-IV: A Critical Review and Replicability Study</a><br>
    </td>
    <td rowspan="6">
      <a href="https://github.com/JoakimEdin/medical-coding-reproducibility">Original Code</a>
    </td>
  </tr>
  <tr>
    <td>Bi-GRU</td>
  </tr>
  <tr>
    <td>CAML</td>
  </tr>
  <tr>
    <td>MultiResCNN</td>
  </tr>
  <tr>
    <td>LAAT</td>
  </tr>
  <tr>
    <td>PLM-ICD</td>
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
