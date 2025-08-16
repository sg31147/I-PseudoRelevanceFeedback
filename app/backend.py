import os
os.chdir('../')

import logging
import math
import os
from pathlib import Path
import json 
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra

from fastapi import FastAPI
from pydantic import BaseModel,Field
from uvicorn import Config, Server
from typing import List
import pandas as pd
import gc
from src.utils.fun_retrieval import pseudo_relevance_feedback

from src.settings import (
    DOWNLOAD_DIRECTORY_MIMICIV
)


from prepare_data.utils import (
    TextPreprocessor,
    preprocess_documents,
    load_gz_file_into_df,
    ID_COLUMN, SUBJECT_ID_COLUMN, TARGET_COLUMN, TEXT_COLUMN
    
)
from src.data.data_pipeline import data_predict_pipeline
from src.factories import (
    get_callbacks,
    get_dataloaders,
    get_datasets,
    get_lookups,
    get_lr_scheduler,
    get_metric_collections,
    get_model,
    get_optimizer,
    get_text_encoder,
    get_transform,
)
from src.trainer.trainer import Trainer
from src.utils.seed import set_seed
from src.settings import best_runs

LOGGER = logging.getLogger(name='test')
LOGGER.setLevel(logging.INFO)



def deterministic() -> None:
    """Run experiment deterministically. There will still be some randomness in the backward pass of the model."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    import torch

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


initialize(config_path="../configs")
#caml multi_res_conv vanillaconv vanillagru laat plm_icd
cfg = compose(config_name="config",
              overrides=["experiment=mimiciv_icd10/plm_icd.yaml",
                         "callbacks=no_wandb",
                         "load_model=null","trainer.epochs=0"]
                        )


cfg.load_model = best_runs[cfg.model.name]


if cfg.deterministic:
    deterministic()
else:
    import torch

    if torch.cuda.is_available():
        print("GPU is available")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU is not available")
set_seed(cfg.seed)


# Check if CUDA_VISIBLE_DEVICES is set
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    if cfg.gpu != -1 and cfg.gpu is not None and cfg.gpu != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = (
            ",".join([str(gpu) for gpu in cfg.gpu])
            if isinstance(cfg.gpu, list)
            else str(cfg.gpu)
        )

    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.set_num_threads(os.cpu_count())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# mapping Definition
download_dir = Path(DOWNLOAD_DIRECTORY_MIMICIV)

d_icd_procedures = load_gz_file_into_df(
    download_dir / "hosp/d_icd_procedures.csv.gz", dtype={"icd_code": str}
)
d_icd_procedures=d_icd_procedures[d_icd_procedures['icd_version']==10]
d_icd_diagnoses = load_gz_file_into_df(
    download_dir / "hosp/d_icd_diagnoses.csv.gz", dtype={"icd_code": str}
)
d_icd_diagnoses=d_icd_diagnoses[d_icd_diagnoses['icd_version']==10]
d_icd_diagnoses = d_icd_diagnoses[['icd_code', 'long_title']].set_index('icd_code')['long_title'].to_dict()
d_icd_procedures = d_icd_procedures[['icd_code', 'long_title']].set_index('icd_code')['long_title'].to_dict()
d_icd={**d_icd_diagnoses,**d_icd_procedures}



train_targets = pd.read_feather(best_runs[cfg.model.name] + '/train_targets.feather')
train_targets = torch.tensor(train_targets.values, dtype=torch.float32, device=device)
print('load train_targets finish')

# val_targets = pd.read_feather(best_runs[cfg.model.name] + '/val_targets.feather')
# val_targets = torch.tensor(val_targets.values, dtype=torch.float32, device=device)
# print('load val_targets finish')


# test_targets = pd.read_feather(best_runs[cfg.model.name] + '/test_targets.feather')
# test_targets = torch.tensor(test_targets.values, dtype=torch.float32, device=device)
# print('load test_targets finish')


# Merge all into retrieve
retrieve = torch.cat([train_targets,
                        #val_targets,
                        #test_targets
                    ], dim=0)
del train_targets
#del val_targets
print('merge finish')

# Specify the path to your JSON file
file_path = best_runs[cfg.model.name] +'/target2index.json'
with open(file_path, 'r') as file:
    data = json.load(file)
token2index = {token: index for index, token in data.items()}





# Define a Pydantic model for the data structure
class TestDataModel(BaseModel):
    id: List[int]
    text: List[str]
    target: List[List[str]]
    split: str
    Task:str
    iteration: int = Field(2, ge=0, le=5, description="iteration must be between 0 and 5") 
    TopKSelection: int = Field(10, ge=1, le=15, description="TopKSelection must be between 1 and 15")
    CosSim_Thresh: float = Field(10, ge=0, le=1, description="CosSim_Thresh must be between 0 and 1")
    alpha: float = Field(1, ge=0, le=1, description="alpha must be between 0 and 1")
    beta: float = Field(0.1, ge=0, le=1, description="beta must be between 0 and 1")
    gamma: float = Field(0.1, ge=0, le=1, description="gamma must be between 0 and 1")
    Precisionk: int = Field(8, ge=1, le=15, description="precision@k must be between 1 and 15")

# Apply nest_asyncio to allow FastAPI to run within Jupyter Notebook

app = FastAPI()

@app.get("/mapping/")
async def mapping():
    return {
        "status": "load mapping successfully",
        "data": {
            values: d_icd[values.replace(".", "")] 
                           for values in token2index.values() 
        }
    }


@app.post("/predict/")
async def predict(Testdata: TestDataModel):
    
    data = Testdata.model_dump(by_alias=True)
    data = pd.DataFrame(data).rename(columns={'id': ID_COLUMN})
    
    #preprocesss
    preprocessor = TextPreprocessor(
                lower=True,
                remove_special_characters_mullenbach=True,
                remove_special_characters=False,
                remove_digits=True,
                remove_accents=False,
                remove_brackets=False,
                convert_danish_characters=False,
            )

    data=preprocess_documents(df=data, preprocessor=preprocessor)
    

    
    # Here, the validated data can be processed as needed
    data = data_predict_pipeline(config=cfg.data,data=data)

    text_encoder = get_text_encoder(
        config=cfg.text_encoder, data_dir=cfg.data.dir, texts=data.get_train_documents
    )
    label_transform = get_transform(
        config=cfg.label_transform,
        targets=data.all_targets,
        load_transform_path=cfg.load_model,
    )
    text_transform = get_transform(
        config=cfg.text_transform,
        texts=data.get_train_documents,
        text_encoder=text_encoder,
        load_transform_path=cfg.load_model,
    )
    
    data.truncate_text(cfg.data.max_length)
    data.transform_text(text_transform.batch_transform)
    lookups = get_lookups(
        config=cfg.lookup,
        data=data,
        label_transform=label_transform,
        text_transform=text_transform,
    )
   
    model = get_model(
        config=cfg.model, data_info=lookups.data_info, text_encoder=text_encoder
    ).to(device)
    
    datasets = get_datasets(
        config=cfg.dataset,
        data=data,
        text_transform=text_transform,
        label_transform=label_transform,
        lookups=lookups,
    )
    
    dataloaders = get_dataloaders(config=cfg.dataloader, datasets_dict=datasets)
    
    optimizer = get_optimizer(config=cfg.optimizer, model=model)
    accumulate_grad_batches = int(
        max(cfg.dataloader.batch_size / cfg.dataloader.max_batch_size, 1)
    )
    
    num_training_steps = (
        math.ceil(len(dataloaders["train"]) / accumulate_grad_batches)
        * cfg.trainer.epochs
    )
    
    lr_scheduler = get_lr_scheduler(
        config=cfg.lr_scheduler,
        optimizer=optimizer,
        num_training_steps=num_training_steps,
    )
    metric_collections = get_metric_collections(
        config=cfg.metrics,
        number_of_classes=lookups.data_info["num_classes"],
        code_system2code_indices=lookups.code_system2code_indices,
        split2code_indices=lookups.split2code_indices,
    )
    callbacks = get_callbacks(config=cfg.callbacks)


    pred = Trainer(
        config=cfg,
        data=data,
        model=model,
        optimizer=optimizer,
        dataloaders=dataloaders,
        metric_collections=metric_collections,
        callbacks=callbacks,
        lr_scheduler=lr_scheduler,
        lookups=lookups,
        accumulate_grad_batches=accumulate_grad_batches,
        experiment_path = Path(cfg.load_model) 
    ).to(device)

    if cfg.load_model:
        pred.experiment_path = Path(cfg.load_model)
    predict=pred.fit(predict=True)
    
    ids, logits, targets = predict["ids"], predict["logits"], predict["targets"]

    Task=Testdata.Task
    alpha=Testdata.alpha
    beta=Testdata.beta
    gamma=Testdata.gamma
    TopKSelection=Testdata.TopKSelection
    CosSim_Thresh=Testdata.CosSim_Thresh
    Precisionk = Testdata.Precisionk
    for i in range(1,Testdata.iteration):

        logits=pseudo_relevance_feedback(retrieve, logits, TopKSelection=TopKSelection, CosSim_Thresh=CosSim_Thresh, 
                                                alpha=alpha, beta=beta, gamma=gamma, 
                                                chunk_size_b=10000)


  

    predict = {id_: {"logits": logit, "targets": target} for id_, logit, target in zip(ids, logits, targets)}

    if Task == "Ranking":
        result = {
            ids.item(): {
                "id": ids.item(),
                "result": {
                    token2index[idx]: f'{d_icd[token2index[idx].replace(".", "")]} ({prob})'

                    for idx, prob in zip(
                        logits["logits"].topk(Precisionk).indices.tolist(),
                        logits["logits"].topk(Precisionk).values.tolist()
                    )
                },
                #"target": [token2index[idx] for idx in (data_entry["target"] == 1).nonzero(as_tuple=True)[0].tolist()],
                "match_percentage": round(
                sum(1 for idx in logits["logits"].topk(Precisionk).indices.tolist() if token2index[idx] in [
                    token2index[idx] for idx in (logits["targets"] == 1).nonzero(as_tuple=True)[0].tolist()
                ]) / Precisionk * 100, 2),
            }
            for ids, logits in predict.items()
        }
    else:
        result = {
            ids.item(): {
                "id": ids.item(),
                "result": {
                    token2index[idx]: f'{d_icd[token2index[idx].replace(".", "")]} ({prob})'
                    for idx, prob in zip(
                        (logits["logits"] > pred.best_db).nonzero(as_tuple=True)[0].tolist(),
                        logits["logits"][logits["logits"] > pred.best_db].tolist()
                    )
                },
                #"target": [token2index[idx] for idx in (data_entry["target"] == 1).nonzero(as_tuple=True)[0].tolist()],
                "match_percentage": round(
                sum(1 for idx in (logits["logits"] > pred.best_db).nonzero(as_tuple=True)[0].tolist() if token2index[idx] in [
                    token2index[idx] for idx in (logits["targets"] == 1).nonzero(as_tuple=True)[0].tolist()
                ]) / len((logits["logits"] > pred.best_db).nonzero(as_tuple=True)[0].tolist()) * 100, 2),

            }
            for ids, logits in predict.items()
        }
    
    del predict,pred,metric_collections,lr_scheduler,num_training_steps,accumulate_grad_batches,optimizer,dataloaders,datasets,callbacks,data,text_encoder,label_transform,text_transform,lookups,model
    gc.collect()
    return {"status": "Predict successfully", "data": result}

# Run the app
config = Config(app=app, host="0.0.0.0", port=8081)
server = Server(config)
server.run()