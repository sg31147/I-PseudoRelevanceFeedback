# ⚕️PSEUDO-RELEVANCE FEEDBACK ON DEEP LANGUAGE MODELS FOR MEDICAL DOCUMENT SUMMARIZATION

This paper is publicly available on npj Medicine.

```bibtex
@inproceedings{PSEUDO-RELEVANCE FEEDBACK,
  address = {Nakhon Pathom, Thailand},
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

**Introduction**

Many specialized language models have been widely adopted for predicting ICD-10 codes. However, there is still a lack of research focused on improving the accuracy and reliability of these models. In response to this gap, we propose a novel approach that incorporates information-retrieval techniques to enhance prediction quality. Specifically, our method uses the Rocchio algorithm to refine and optimize model outputs, building on existing language models to improve their practical applicability. We have also developed a web-based application—a dynamic playground—where users can interact with and evaluate our method once they have followed the implementation steps described below.


![My Image Description](/files/retrieval/method3.png)


**Related Research**

<table style="margin: auto; border-collapse: collapse;">
  <tr>
    <th style="text-align: center; vertical-align: middle;">Model</th>
    <th style="text-align: center; vertical-align: middle;">Paper</th>
    <th style="text-align: center; vertical-align: middle;">Original Code</th>
  </tr>
  <tr>
    <td style="text-align: center; vertical-align: middle;">CNN</td>
    <td rowspan="6" style="text-align: center; vertical-align: middle;">
      <a href="https://arxiv.org/abs/2304.10909">
        Automated Medical Coding on MIMIC-III and MIMIC-IV: 
        A Critical Review and Replicability Study
      </a>
    </td>
    <td rowspan="6" style="text-align: center; vertical-align: middle;">
      <a href="https://github.com/JoakimEdin/medical-coding-reproducibility">
        Original Code
      </a>
    </td>
  </tr>
  <tr>
    <td style="text-align: center; vertical-align: middle;">Bi-GRU</td>
  </tr>
  <tr>
    <td style="text-align: center; vertical-align: middle;">CAML</td>
  </tr>
  <tr>
    <td style="text-align: center; vertical-align: middle;">MultiResCNN</td>
  </tr>
  <tr>
    <td style="text-align: center; vertical-align: middle;">LAAT</td>
  </tr>
  <tr>
    <td style="text-align: center; vertical-align: middle;">PLM-ICD</td>
  </tr>
</table>


**Setup and Usage**

1. Install the environment and dependencies

Make sure you have Python 3.10 installed, then install the required packages:

```bibtex
python -m venv pseudo_relevance_feedback
cd ./pseudo_relevance_feedback
source ./bin/activate
```

Then, install the required packages and set up the project:

```bibtex
git clone https://github.com/sg31147/pseudo_relevance_feedback.git
pip install -e .
```


2. Download the MIMIC-IV dataset from PhysioNet. You will need to request access to the MIMIC-IV data, which requires following the credentialing process on PhysioNet. Note that it typically takes 2–3 days to receive approval. They will review your intended use to ensure it is not for commercial purposes or for direct use in large language models (LLMs), as both are prohibited. After receiving approval, you can proceed with downloading the dataset.

3. [Clinical notes](https://physionet.org/content/mimic-iv-note/2.2/) (MIMIC-IV Note)

    	Place the file at: ./dataset/mimiciv/note/*

4. [Reference tables](https://physionet.org/content/mimiciv/2.2/) (MIMIC-IV hosp files):

    - d_icd_diagnoses.csv.gz (855.8 KB)
    - d_icd_procedures.csv.gz (575.4 KB)
    - diagnoses_icd.csv.gz (32.0 MB)
    - procedures_icd.csv.gz (7.4 MB)

			Place them at: ./dataset/mimiciv/hosp/*
  
  
5. Instructions:

    Option A: Run the Notebooks
  
      Run notebooks steps 1–5 for data preprocessing, model training, and retrieval.
    
    Option B: Use Pre-Built Output
  
      Skip the above steps by downloading the prepared output from my public site.
  
      Place the files in:
      - ./file/*
      - ./experiment/*

**Web Application Playground**

To see the web application in action, run notebook step 6 to view the final output.

![My Image Description](/files/retrieval/result5.png)

**Docker Alternative**



If you’re comfortable with Docker, you can use the included Docker Compose file to pull and run the pre-built image:

```
docker compose up
```

Enjoy exploring this cutting-edge approach to medical document summarization—may it illuminate new pathways for more efficient, patient-centered care.
