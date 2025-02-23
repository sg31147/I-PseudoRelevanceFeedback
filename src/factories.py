from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from rich.pretty import pprint
from torch.utils.data import DataLoader

import src.data.batch_sampler as batch_samplers
import src.data.datasets as datasets
import src.data.transform as transform
import src.metrics as metrics
import src.models as models
import src.text_encoders as text_encoders
import src.trainer.callbacks as callbacks
from src.data.datatypes import Data, Lookups
from src.lookups import load_lookups
from src.text_encoders.base_text_encoder import BaseTextEncoder


def get_lookups(
    config: OmegaConf,
    data: Data,
    label_transform: transform.Transform,
    text_transform: transform.Transform,
) -> Lookups:
    return load_lookups(
        config=config,
        data=data,
        label_transform=label_transform,
        text_transform=text_transform,
    )


def get_model(
    config: OmegaConf, data_info: dict, text_encoder: Optional[Any] = None
) -> models.BaseModel:
    model_class = getattr(models, config.name)
    return model_class(text_encoder=text_encoder, **data_info, **config.configs)


def get_optimizer(config: OmegaConf, model: models.BaseModel) -> torch.optim.Optimizer:
    optimizer_class = getattr(torch.optim, config.name)
    return optimizer_class(model.parameters(), **config.configs)


def get_lr_scheduler(
    config: OmegaConf,
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    if config.name is None:
        return None
    if hasattr(torch.optim.lr_scheduler, config.name):
        lr_scheduler_class = getattr(torch.optim.lr_scheduler, config.name)
        return lr_scheduler_class(optimizer, **config.configs)
    from transformers import get_scheduler

    return get_scheduler(
        name=config.name,
        optimizer=optimizer,
        num_training_steps=num_training_steps,
        **config.configs,
    )


def get_text_encoder(
    config: OmegaConf, data_dir: str, texts: list[str]
) -> text_encoders.BaseTextEncoder:
    
    #จะกำหนดว่าใช้ pretrain หรือ ไม่ใช้
    if not hasattr(config, "name"):
        return None

    path = Path(data_dir) / config.file_name

    text_encoder_class = getattr(text_encoders, config.name)
    #มี embed แล้วไม่ fit ใหม่
    if path.exists() and config.load_model:
        print('embed already exist')
        return text_encoder_class.load(path)
    
    text_encoder = text_encoder_class(config=config.configs)
    text_encoder.fit(texts)
    text_encoder.save(path)
    return text_encoder


def get_transform(
    config: OmegaConf,
    targets: Optional[set[str]] = None,
    texts: Optional[list[str]] = None,
    text_encoder: Optional[BaseTextEncoder] = None,
    load_transform_path: Optional[str] = None,
) -> transform.Transform:
    """Get transform class

    Args:
        config (OmegaConf): Config for the transform
        targets (Optional[set[str]], optional): Groundtruth targets. Defaults to None.
        texts (Optional[list[str]], optional): Text. If text is not provided, the transform will be fitted on the targets. Defaults to None.
        text_encoder (Optional[BaseTextEncoder], optional): Text encoder. If not none, the tokenmap will be loaded from the text encoder. Defaults to None.
        load_transform_path (Optional[str], optional): Path of where the transform is saved. Will be loaded if not none. Defaults to None.

    Raises:
        ValueError: Error if either texts, targets, text_encoder or load_transform_path are not provided.

    Returns:
        transform.Transform: _description_
    """

    #encoder type
    transform_class = getattr(transform, config.name)(**config.configs)

    if load_transform_path:
   
        transform_class.load(load_transform_path)
        print("loaded transform")
    #part นี้คือแปล  text เป็น vector จริง 
    
    elif text_encoder:
        
        transform_class.set_tokenmap(
            token2index=text_encoder.token2index, index2token=text_encoder.index2token
        )

    elif texts:#กรณี pretrain 
        
        transform_class.fit(texts)
 
    elif targets:
   
        transform_class.fit(targets)
    else:
        raise ValueError(
            "Provide set of labels, a text encoder or texts of tokens to perform fit transformation"
        )
    #ถึงตรงนรี้ยังไม่แปลง แค่ set ก่อน run
    return transform_class


def get_datasets(
    config: OmegaConf,
    data: Data,
    text_transform: transform.Transform,
    label_transform: transform.Transform,
    lookups: Lookups,
) -> dict[str, datasets.BaseDataset]:

    dataset_class = getattr(datasets, config.name)
   
    datasets_dict = {}

    #เริ่มเกิด tqdm ตรงนี้
    
    train_data = data.train
    val_data = data.val
    test_data = data.test
    del data.df
   
    #มีการ init dataset ตรงนี้อยู๋ มี method อื่นๆ ที่ยังคงถูกใช้เพื่อ process ข้อมูล ตอน Loader 
    datasets_dict["train"] = dataset_class(
        train_data,
        split_name="train",
        text_transform=text_transform,
        label_transform=label_transform,
        lookups=lookups,
        **config.configs,
    )

    datasets_dict["val"] = dataset_class(
        val_data,
        split_name="val",
        text_transform=text_transform,
        label_transform=label_transform,
        lookups=lookups,
        **config.configs,
    )
  
    datasets_dict["test"] = dataset_class(
        test_data,
        split_name="test",
        text_transform=text_transform,
        label_transform=label_transform,
        lookups=lookups,
        **config.configs,
    )
    
    return datasets_dict

#dict จะรวม แต่ตอนแบ่ง batch จะแยกกัน["train","test","val"]
def get_dataloaders(
    config: OmegaConf, datasets_dict: dict[str, datasets.BaseDataset]
) -> dict[str, DataLoader]:
    dataloaders = {}
    train_batch_size = min(config.batch_size, config.max_batch_size)
    pprint(f"Train batch size: {train_batch_size}")
    #แบ่งว่าจะเข้า train ไหม
    if config.batch_sampler.name:
        #เริ่มพูดถึง batch เพื่อ optimize ให้เหมาะการเอาลง cpu gpu จะ drop last ไหม 
        batch_sampler_class = getattr(batch_samplers, config.batch_sampler.name)
        batch_sampler = batch_sampler_class(
            dataset=datasets_dict["train"],
            batch_size=train_batch_size,
            drop_last=config.drop_last,
            **config.batch_sampler.configs,
        )

        dataloaders["train"] = DataLoader(
            datasets_dict["train"],
            batch_sampler=batch_sampler,
            collate_fn=datasets_dict["train"].collate_fn,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )

    else:
        dataloaders["train"] = DataLoader(
            datasets_dict["train"],
            shuffle=True,
            batch_size=train_batch_size,
            drop_last=config.drop_last,
            collate_fn=datasets_dict["train"].collate_fn,#padding
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )
    
    # test ไม่ shuffle แต่ใส่ maxbatchsize
    dataloaders["train_val"] = DataLoader(
        datasets_dict["train"],
        batch_size=config.max_batch_size,
        shuffle=False,
        collate_fn=datasets_dict["val"].collate_fn,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    dataloaders["val"] = DataLoader(
        datasets_dict["val"],
        batch_size=config.max_batch_size,
        shuffle=False,
        collate_fn=datasets_dict["val"].collate_fn,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    dataloaders["test"] = DataLoader(
        datasets_dict["test"],
        batch_size=config.max_batch_size,
        shuffle=False,
        collate_fn=datasets_dict["test"].collate_fn,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    return dataloaders


def get_metric_collection(
    config: OmegaConf,
    number_of_classes: int,
    code_system_code_indices: Optional[torch.Tensor] = None,# มี system เดียวนะ tensor([   4,    6,   16,  ..., 7929, 7934, 7935])  | tensor([3963,    5, 3964,  ..., 7937, 7939, 6164])
    split_code_indices: Optional[torch.Tensor] = None, #split_code_indices diag only ที่เป็น target เพราะก่อนหน้า วน loop ออกมา
    code_system_name: Optional[str] = None,#icd10_diag  #icd10_proc  มีแต่ชื่อ
) -> metrics.MetricCollection:
    metric_list = []
    for metric in config:
        metric_class = getattr(metrics, metric.name)
        metric_list.append( #สิ่งสำคัญคือ เห็นมันเอาตแหน เข้าไปทุกคลาสเลยแหะ num_class
            metric_class(number_of_classes=number_of_classes, **metric.configs) #ตรงนี้เป็นตัวกำหนดประเภท metric ไปแล้ว มันถูกตั้งค่าก่อ่น train แล้วเก็บเเป็น list ไปหมดแล้ว 
        )
       
    #กลไกแยกกลุ่มไส้ของ 3  กลุ่มไม่รวม train เอาข้างในมา intersect 
    if code_system_code_indices is not None and split_code_indices is not None:
        # Get overlapping indices
        code_indices = torch.tensor(
            np.intersect1d(code_system_code_indices, split_code_indices)
        )# สุดท้ายก็คงได้แค่ ตัว split_code_indices เพราะมันเล็กว่าไง

    #ไม่น่าเข้าเงื่อนไขนี้นะ
    elif code_system_code_indices is not None:
        code_indices = code_system_code_indices.clone()
    #แยกกลุ่ม all 4 กลุ่ม 
    elif split_code_indices is not None:
        code_indices = split_code_indices.clone() 
    else:
        code_indices = None
    #คืนประเภท metric,codeแต่ละกลุ่ม,ชื่อกลุ่ม
    return metrics.MetricCollection(
        metrics=metric_list,
        code_indices=code_indices, #เฉพาะกลุ่มหมายถึงกลุ่มที่ย่อยถึงขนาดตัว diag แล้ว proc แล้วนะ ไส้นั่นหละ
        code_system_name=code_system_name, # NONE ถ้าไม่ได้สนใจกลุ่มไส้ของ 3  กลุ่มไม่รวม train
    )


def get_metric_collections(
    config: OmegaConf,
    number_of_classes: int,
    split_names: list[str] = ["train", "train_val", "val", "test"],
    splits_with_multiple_code_systems: set[str] = {"train_val", "val", "test"},
    code_system2code_indices: dict[str, torch.Tensor] = None, #รวม 2 code system diag+proc = number_of_classes
    split2code_indices: dict[str, torch.Tensor] = None, #แยก train train_val val test
) -> dict[str, dict[str, metrics.MetricCollection]]:
    metric_collections = defaultdict(dict)
    for split_name in split_names:
        if split2code_indices is not None:
            split_code_indices = split2code_indices.get(split_name) # มันคือรหัส icd ที่มีเฉพาะกลุ่มมัน
        else:
            split_code_indices = None
        #เก็บ metric แต่ละกลุ่ม "train", "train_val", "val", "test"  m[x][y] ยังไม่พูดถึงเลข
            #code_system2code_indices => diag proc
            #split2code_indices =>  train train_val val test
        metric_collections[split_name]["all"] = get_metric_collection(
            config=config,
            number_of_classes=number_of_classes, #class ทุก class
            split_code_indices=split_code_indices, #กลุ่มใครกลุ่มมันแล้วตรงนี้
        )
        #train กระโดดข้าม loop ไปเลย ตรงนี้ไม่แยก diag + proc
        if split_name not in splits_with_multiple_code_systems: #"train_val", "val", "test" ฉะนัน train ไม่มีไส้เอาไว้แยก diag proc
            continue

        if code_system2code_indices is None:
            continue
        
        #icd10_diag tensor([   4,    6,   16,  ..., 7929, 7934, 7935]) #icd10_proc tensor([3963,    5, 3964,  ..., 7937, 7939, 6164])
        for (
            code_system_name,
            code_system_code_indices,
        ) in code_system2code_indices.items(): #แต่ละ proc และ diag 
            metric_collections[split_name][code_system_name] = get_metric_collection(
                config=config,
                number_of_classes=number_of_classes,
                code_system_code_indices=code_system_code_indices, #ย่อยแล้ว เช่น diag only ทั้งหมดมีอะไรใน
                split_code_indices=split_code_indices, #เฉพาะกลุ่ม ที่เป็น train เท่านั้น code_sys จะ < split เสมอ ถ้าเกิดว่าเราซอยตัว frequency ออกไป
                code_system_name=code_system_name,#ชื่อกลุ่ม
            )

    return metric_collections


def get_callbacks(config: OmegaConf) -> list[callbacks.BaseCallback]:
    callbacks_list = []
    for callback in config:
        callback_class = getattr(callbacks, callback.name) #class เดียว แต่ metric มันเกือบ 40 ตัวเลยนะ 
        callbacks_list.append(callback_class(config=callback.configs))
    return callbacks_list
