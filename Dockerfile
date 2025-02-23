FROM python:3.10-slim
#FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# Install only the required Python packages explicitly
RUN pip install --upgrade pip


COPY ./README.md /app
COPY ./setup.py /app
COPY ./app /app/app
COPY ./requirements.txt /app
RUN pip install --no-cache-dir -e .


COPY ./configs /app/configs

COPY ./experiments/jfywyfod/best_model.pt /app/experiments/jfywyfod/best_model.pt
COPY ./experiments/jfywyfod/retrieve.feather /app/experiments/jfywyfod/retrieve.feather
COPY ./experiments/jfywyfod/target2index.json /app/experiments/jfywyfod/target2index.json

COPY ./dataset/mimiciv/hosp /app/dataset/mimiciv/hosp

COPY ./files/hfs/RoBERTa-base-PM-M3-Voc-hf /app/files/hfs/RoBERTa-base-PM-M3-Voc-hf
COPY ./files/data/mimiciv_icd10 /app/files/data/mimiciv_icd10
COPY ./prepare_data /app/prepare_data
COPY ./src/ /app/src



# RUN pip install --upgrade accelerate
# RUN pip uninstall -y transformers accelerate
# RUN pip install  transformers accelerate
WORKDIR /app/app
CMD ["sh", "-c", "python3 ./backend.py & streamlit run ./frontend.py"]