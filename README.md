# ⚕️PSEUDO-RELEVANCE FEEDBACK ON DEEP LANGUAGE MODELS FOR MEDICAL DOCUMENT SUMMARIZATION

This paper is publicly available on npj Medicine.

```bibtex
@inproceedings{edinAutomatedMedicalCoding2023,
  address = {Taipei, Taiwan},
  title = {PSEUDO-RELEVANCE FEEDBACK ON DEEP LANGUAGE MODELS FOR MEDICAL DOCUMENT SUMMARIZATION},
  isbn = {xxx-x-xxxx-xxxx-x},
  shorttitle = {PSEUDO-RELEVANCE FEEDBACK ON DEEP LANGUAGE MODELS},
  doi = {xx.xx/xxxxxxx.xxxxxxx},
  booktitle = {Medicine Journal},
  publisher = {Npj medicine},
  author = {Kitti Akkhawatthanakun1, Paisarn Muneesawang, Lalita Narupiyakul, and Konlakorn Wongpatikaseree},
  year = {2025}
}
```

## Introduction

Many specialized language models have been widely adopted for predicting ICD-10 codes. However, there is still a lack of research focused on improving the accuracy and reliability of these models. In response to this gap, we propose a novel approach that incorporates information-retrieval techniques to enhance prediction quality. Specifically, our method uses the Rocchio algorithm to refine and optimize model outputs, building on existing language models to improve their practical applicability. We have also developed a web-based application—a dynamic playground—where users can interact with and evaluate our method once they have followed the implementation steps described below.


![My Image Description](/files/retrieval/pesudo_relevance_feedback.png)


### Related Research
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

## Setup and Usage 
1. Install the environment and dependencies
Make sure you have Python 3.10 installed, then install the required packages:

```bibtex
conda create -n pseudo_relevance python=3.10
conda activate pseudo_relevance
```

Then, install the required packages and set up the project:

```bibtex
pip install -e .
```

2. Download the MIMIC-IV dataset from PhysioNet. You will need to request access to the MIMIC-IV data, which requires following the credentialing process on PhysioNet. Note that it typically takes 2–3 days to receive approval. They will review your intended use to ensure it is not for commercial purposes or for direct use in large language models (LLMs), as both are prohibited. After receiving approval, you can proceed with downloading the dataset.

  2.1 [Clinical notes](https://physionet.org/content/mimic-iv-note/2.2/) (MIMIC-IV Note)

      Place the file at: ./dataset/mimiciv/note/*
   
  2.2 Reference tables (MIMIC-IV hosp files):

          d_icd_diagnoses.csv.gz (855.8 KB)

          d_icd_procedures.csv.gz (575.4 KB)

          diagnoses_icd.csv.gz (32.0 MB)

          procedures_icd.csv.gz (7.4 MB)

      Place them at: ./dataset/mimiciv/hosp/*
   


  2.3 download model from my public site. extract this file and follow this directory.
  
      Place them at:            - ./experiment/*
  
  2.4 you can run my notebooks steo 1-5 follow each step to get model,...etc output. or skip by download my output that i prepare for you by download from my public site. you will got
      Place them at:            - ./file/*

   
bonus  if you want to see webapplicaion that i usage a little small code you can run notebooks step 6 here are final output




## Docker Alternative
If you are comfortable with Docker, you can use the provided Docker Compose file to pull and run our pre-built image with a single command:
```
docker compose up
```
![My Image Description](/files/retrieval/webapp.png)


