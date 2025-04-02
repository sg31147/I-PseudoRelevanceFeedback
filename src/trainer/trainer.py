import gc
from collections import defaultdict
from pathlib import Path
from tempfile import mkdtemp
from typing import Optional

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from rich.pretty import pprint
from tqdm import tqdm
from torch.utils.data import DataLoader
import os 
from src.data.datatypes import Data, Lookups
from src.metrics import  MetricCollection
from src.models import BaseModel
from src.settings import ID_COLUMN, TARGET_COLUMN
from src.trainer.callbacks import BaseCallback
from src.utils.decision_boundary import f1_score_db_tuning

class Trainer:
    def __init__( #สังเกต python ใส่กลับมาเป็น class หมดแล้วนะ 
        self,
        config: OmegaConf,
        data: Data,
        model: BaseModel,
        optimizer: torch.optim.Optimizer,
        dataloaders: dict[str, DataLoader],
        metric_collections: dict[str, dict[str, MetricCollection]],
        callbacks: BaseCallback,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        lookups: Optional[Lookups] = None,
        accumulate_grad_batches: int = 1,
        experiment_path = None
    ) -> None:
        self.config = config
        self.data = data
        self.model = model
        self.optimizer = optimizer
        self.dataloaders = dataloaders
        self.callbacks = callbacks
        self.device = "cpu"
        self.metric_collections = metric_collections #เกิดการสร้างแบบแยกกลุ่มไปหมดแล้วหละ #กลไก top best k อยู่ตรงนี้
        self.lr_scheduler = lr_scheduler
        self.lookups = lookups
        self.accumulate_grad_batches = accumulate_grad_batches
        pprint(f"Accumulating gradients over {self.accumulate_grad_batches} batch(es).")
        self.validate_on_training_data = config.trainer.validate_on_training_data
        self.print_metrics = config.trainer.print_metrics
        self.epochs = config.trainer.epochs
        self.epoch = 0
        self.use_amp = config.trainer.use_amp
        self.threshold_tuning = config.trainer.threshold_tuning
        self.gradient_scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        #ไฟล์นี้บรรจุแค่ configfile เอง มันมีการถูกบันทึกเหมือนกลไก wandb
        self.experiment_path = experiment_path or Path(mkdtemp())
        pprint(self.experiment_path)
        self.current_val_results = None
        self.stop_training = False
        self.best_db = 0.5
        self.on_initialisation_end()

    def fit(self,predict:bool=None) -> None:
        """Train and validate the model."""
        try:
            self.on_fit_begin()
            #กรณี eval จะออก loop เลย
            for _ in range(self.epoch, self.epochs):
                #ตรงนี้ดัก early stopping
                if self.stop_training: #ทำงานตรง on_epoch_end เห็นว่าควรให้ค่า false หลังจากเจอ early stopping
                    break
                self.on_epoch_begin() #ไป reset metriccollection แต่ละตัว แต่ความสามารถของ metric แต่ละอัน เก๋บ data ทิ้งไว้นะ
                self.train_one_epoch(self.epoch)#บรรทัดนี้ train ที่ไม่แยก diag+proc ปกติจะแยกหมด
                if self.validate_on_training_data:
                    self.train_val(self.epoch, "train_val") #diag+proc แยกหมด
                self.val(self.epoch, "val") #บรรทัดนี้ไม่สนใจ eval ใช้ train นำ
                self.on_epoch_end()#เก็บ best model และกำหนด early stopping 
                #metric print ใครปริ้นมัน
                #พวก callback wandb savemodel ลงไฟล์เก็บใครเก้บมัน earlystopping ไว้จบ train เลย
                #ไม่มีใครได้ tune เลย
                self.epoch += 1
            self.on_fit_end() #load model ดีสุดใส่ไว้จากไฟล์เลย 

            #ตรงนี้จะ val ทั้ง val ทั้ง test พร้อมกันเลย  มันคือหยุด train ละ แต่ตอนจะทายจะยึดจาก model ที่ดีที่สุด
            if predict:  
                return self.val(self.epoch, "test", evaluating_best_model=False,predict=predict)
            else:
                self.val(self.epoch, "val", evaluating_best_model=True) # feather val
                self.val(self.epoch, "test", evaluating_best_model=True)  # feather_test
                #ไม่มีใครได้ปรับ threshold นะ อาจจะอยู่ใน part analysis เลย
                self.save_final_model()  #เป็นไปได้ที่ มันไม่ท่ากับ fit เนื่องจากมัน
        except KeyboardInterrupt:
            pprint("Training interrupted by user. Stopping training")
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
        self.on_end()#มีแค่ wan

    def train_one_epoch(self, epoch: int) -> None:
        """Train the model for one epoch.

        Args:
            epoch (int): The current epoch.
        """
        self.model.train()
        self.on_train_begin()
        num_batches = len(self.dataloaders["train"]) 
        for batch_idx, batch in enumerate(
            tqdm(self.dataloaders["train"], desc=f"Epoch: {epoch} | Training")
        ):#แบ่ง batch
            batch = batch.to(self.device)
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=self.use_amp
            ):
                output = self.model.training_step(batch)
              
                loss = output["loss"] / self.accumulate_grad_batches
                #backpropagate
            self.gradient_scaler.scale(loss).backward() #สะสมตัว loss

            #กรณีสะสมครบ batch accumurate แล้ว #นึกถึง cfg.dataloader.batch กับ .maxbatchsize ปกติจะปรับทีละ batch เลย ในที่นี้ปรับทั้ง optimizer กับ lr
            if ((batch_idx + 1) % self.accumulate_grad_batches == 0) or (
                batch_idx + 1 == num_batches  
            ):
                #ปรับค่า weight  Optimizer 
                self.gradient_scaler.step(self.optimizer) 
                self.gradient_scaler.update()#ตรงนี้เอาไว้ปรับจริง optimizer
                #ปรับ learning rate
                if self.lr_scheduler is not None:
                    if not isinstance(
                        self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                    ):
                        self.lr_scheduler.step()
                self.optimizer.zero_grad()  # โดย default สะสมค่าเลย ล้างค่า gradient เพื่อไม่ให้สะสม
            self.update_metrics(output, "train") # alert metric #อย่าลืมว่าเราทำทีละ batch จริงๆมันจะวนครบทั้ง epoch
        self.on_train_end(epoch)#call back

    def train_val(self, epoch, split_name: str = "train_val") -> None:
        """Validate on the training data. This is useful for testing for overfitting. Due to memory constraints, we donøt save the outputs.
        
        Args:
            epoch (_type_): _description_
            split_name (str, optional): _description_. Defaults to "train_val".
        """
        self.model.eval()
        self.on_val_begin()
        with torch.no_grad(): #nograd แต่ใช้ model จากที่ train ไว้ ซึ่งกว่าเอา threshold ดีสุดแล้ว
            for batch in tqdm(
                self.dataloaders[split_name], #train_val
                desc=f"Epoch: {epoch} | Validating on training data",
            ):
                with torch.autocast(
                    device_type="cuda", dtype=torch.float16, enabled=self.use_amp
                ):
                    output = self.model.validation_step(batch.to(self.device)) # model ที่ train ไว้แล้ว ตัวมันเองทายตัวมันเอง มันคือ val ใน train 
                    #ตรงนนี้คือ validate ทีละ 1 batch return {"logits": logits, "loss": loss, "targets": targets}
                self.update_metrics(output, split_name) # updmetric ยังอยู่ในทีละ batch
            # ตรงนี้ครบทุก batch ละ 
            self.on_val_end(split_name, epoch) # จบเลย เข้า print metric + wandb

    def val(
        self, epoch, split_name: str = "val", evaluating_best_model: bool = False,predict:bool=False
    ) -> None:

        self.model.eval()
        self.on_val_begin()
        logits = []
        targets = []
        logits_cpu = []
        targets_cpu = []
        ids = []

        
        with torch.no_grad(): #ไม่มีการยุ่งกับ gradient ละ 
            for idx, batch in enumerate(
                tqdm( #แบ่ง batch ตรงนี้จะน้อยเพราะไม่ใช่ train ละ
                    self.dataloaders[split_name],
                    desc=f"Epoch: {epoch} | Validating on {split_name}",
                )
            ):
               
                with torch.autocast(
                    device_type="cuda", dtype=torch.float16, enabled=self.use_amp
                ):
                  
                    output = self.model.validation_step(batch.to(self.device)) #model ที่ค้างไว้ตอน train


                if not predict:
                    self.update_metrics(output, split_name) #set metric ก่อน
                        
                #เพิ่มบน gpu ล้วนๆ
                logits.append(output["logits"])
                targets.append(output["targets"])
                ids.append(batch.ids)
                if idx % 1000 == 0: #ขนกลับไปทีละ 1000
               
                    # move to cpu to save gpu memory
                    logits_cpu.append(torch.cat(logits, dim=0).cpu())
                    targets_cpu.append(torch.cat(targets, dim=0).cpu()) 
                    #ล้าง ram gpu
                    logits = []
                    targets = []
             
            #สำหรับตัวเหลือ
            if logits:  # Ensures logits is not empty before concatenating
                logits_cpu.append(torch.cat(logits, dim=0).cpu())
            if targets:  # Ensures targets is not empty before concatenating
                targets_cpu.append(torch.cat(targets, dim=0).cpu())
     
        logits = torch.cat(logits_cpu, dim=0)
        targets = torch.cat(targets_cpu, dim=0)
        ids = torch.cat(ids, dim=0)

        #ตรงนี้ครบทุกตัวแล้ว 
        if predict:
            return {"ids": ids,"logits":logits,"targets":targets}
         
        
        
        self.on_val_end(split_name, epoch, logits, targets, ids, evaluating_best_model) #จบแล้ว 

    def update_metrics(self, outputs: dict[str, torch.Tensor], split_name: str) -> None:
        #ยังจำได้ว่าตัว metric จะเก็บประเภทของ train test val ไว้  [splitename][['all']['diag','proc']]
        #{"logits": logits, "loss": loss, "targets": targets}
     
        # split_name = train 1 train_val 3 val 3 test 3 #target_name = all / diag / proc กรณี train จะมีแค้่ all อันเดยีว
        for target_name in self.metric_collections[split_name].keys():
            #self.metric_collections[split_name] ตรงนี้คือให้เรียก meticcollection ของกลุ่มมันใช้งานเลย ฉะนั้น update ก็รู้เลยว่าเอา กลุ่มไหนแล้ว เพราะตอนสร้าง
            #มีแยกกลุ่ม collection metric เด๋วเราค่อยเอา metroc แต่ละตัว ไปวนๆ เอา score ออกมา
            self.metric_collections[split_name][target_name].update(outputs)


    def calculate_metrics(
        self,
        split_name: str,
        logits: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        evaluating_best_model: bool = False,#ทำงานตอนจบ epoch ครบแล้ว
    ) -> dict[str, dict[str, torch.Tensor]]:
        results_dict = defaultdict(dict)
        if split_name == "val":
            for target_name in self.metric_collections[split_name].keys():
                results_dict[split_name][target_name] = self.metric_collections[
                    split_name
                ][target_name].compute()
        else: #train,test เข้าตรงนี้ 
           
            for target_name in self.metric_collections[split_name].keys():
  
                results_dict[split_name][target_name] = self.metric_collections[
                    split_name
                ][target_name].compute(logits, targets) #ตรงนี้สั่งให้คำนวน compute
        # val เข้า ดู tune ด้วยไหม 

        if self.threshold_tuning and split_name == "val":   #กรณ val จะมีตัว tuning อยู่นะ
      
            best_result, best_db = f1_score_db_tuning(logits, targets) #เลือก f1 กับ threshold ที่ดีที่สุดเลย
     
            results_dict[split_name]["all"] |= {"f1_micro_tuned": best_result} # F1 ภาพรวมไป
  
            if evaluating_best_model:
                #เจอดีกว่าก็แจ้ง bestmodel
                pprint(f"Best threshold: {best_db}")
                pprint(f"Best result: {best_result}")
                for target_name in self.metric_collections["test"]: # test เข้าตรงนี้ โดยใช้ threshold จาก val ที่ดีที่สุด
                    self.metric_collections["test"][target_name].set_threshold(best_db) #เอาเข้า self set อย่างเดียวนะ 
            self.best_db = best_db  #เอาเข้า self 
        return results_dict #คืน dict

    def reset_metric(self, split_name: str) -> None:
        for target_name in self.metric_collections[split_name].keys():
            self.metric_collections[split_name][target_name].reset_metrics()

    def reset_metrics(self) -> None:
        for split_name in self.metric_collections.keys():
            for target_name in self.metric_collections[split_name].keys():
                self.metric_collections[split_name][target_name].reset_metrics()

    def on_initialisation_end(self) -> None:
        for callback in self.callbacks:
            #เรียก first callback 
            callback.on_initialisation_end(self)

    def on_fit_begin(self) -> None:
        for callback in self.callbacks:
            callback.on_fit_begin(self)

    def on_fit_end(self) -> None:
        
        for callback in self.callbacks:
            callback.on_fit_end(self) #มัน เอา model ที่เก็บไว้ซึงดีที่สุด load กลับมาใส่รอเลย

    def on_train_begin(self) -> None:
        for callback in self.callbacks:
            callback.on_train_begin()

    def on_train_end(self, epoch: int) -> None:
        results_dict = self.calculate_metrics(split_name="train") #train เข้า  
        results_dict["lr"] = self.optimizer.param_groups[0]["lr"]
        self.log_dict(results_dict, epoch)#แจ้ง log metric
        for callback in self.callbacks:
            callback.on_train_end()#แจ้งว่า end pass ไปเฉยๆ

    def on_val_begin(self) -> None:
        for callback in self.callbacks:
            callback.on_val_begin()

    def on_val_end(
        self,
        split_name: str,
        epoch: int,
        logits: torch.Tensor = None,
        targets: torch.Tensor = None,
        ids: torch.Tensor = None,
        evaluating_best_model: bool = False
    ) -> None:
        
        results_dict = self.calculate_metrics(
            split_name=split_name,
            logits=logits,
            targets=targets,
            evaluating_best_model=evaluating_best_model,
        )
        self.current_val_results = results_dict
        self.log_dict(results_dict, epoch)# แจ้ง log ตรงนี้
        for callback in self.callbacks:#เข้า logic call back pass เฉยๆ
            callback.on_val_end()
 
        if evaluating_best_model:# save feather ตอนท้ายเท่านั้นนะ ไม่เกี่ยวกับตอน train
            self.save_predictions(
                split_name=split_name, logits=logits, targets=targets, ids=ids
            )

    def save_predictions( #เก็บ id output ไว้เลยว่า ทำนายอะไร
        self,
        split_name: str = "test",
        logits: torch.Tensor = None,
        targets: torch.Tensor = None,
        ids: torch.Tensor = None,
    ):
        from time import time
      
        tic = time()
        pprint("Saving predictions")
        label_transform = self.dataloaders[split_name].dataset.label_transform
        code_names = label_transform.get_classes()
        logits = logits.numpy()
        pprint("Building dataframe")
        df = pd.DataFrame(logits, columns=code_names)
        pprint("Adding targets")
        df[TARGET_COLUMN] = list(map(label_transform.inverse_transform, targets)) #เก็บ target ที่ inverse transform ให้แล้ว
        pprint("Adding ids")
        df[ID_COLUMN] = ids.numpy() #มี tag id ให้ด้วยนะตอน save หลังทำนาย
        pprint("Saving dataframe")
        df.to_feather(self.experiment_path / f"predictions_{split_name}.feather")
        pprint("Saved predictions in {:.2f} seconds".format(time() - tic))

    def on_epoch_begin(self) -> None:
        self.reset_metrics()#เริม่ใหม่ reset metric ใหม่หมด
        for callback in self.callbacks:
            callback.on_epoch_begin(self)

    def on_epoch_end(self) -> None:
        if self.lr_scheduler is not None:
            if isinstance(
                self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                self.lr_scheduler.step(
                    self.current_val_results["val"]["all"]["f1_micro"]
                )

        for callback in self.callbacks:
            callback.on_epoch_end(self) #แจ้ง callback หลังจบหลูป

    def on_batch_begin(self) -> None:
        for callback in self.callbacks:
            callback.on_batch_begin()

    def on_batch_end(self) -> None:
        for callback in self.callbacks:
            callback.on_batch_end()

    def log_dict(
        self, nested_dict: dict[str, dict[str, torch.Tensor]], epoch: int
    ) -> None:
        if self.print_metrics:
            self.print(nested_dict) #print dict ตรงนี้ ไม่ปรื้นก็ได้นะ 
        for callback in self.callbacks:#ตรงนี้ มี แต่ wandb ทำงาน อื่นๆ pass
            callback.log_dict(nested_dict, epoch) #จบทีกลุ่มใครกลุ่มมันr

    def on_end(self) -> None:
        for callback in self.callbacks:
            callback.on_end()

    def print(self, nested_dict: dict[str, dict[str, torch.Tensor]]) -> None:
        for split_name in nested_dict.keys():
            pprint(nested_dict[split_name])

    def to(self, device: str) -> "Trainer":
        self.model.to(device)
        for split_name in self.metric_collections.keys():
            for target_name in self.metric_collections[split_name].keys():
                self.metric_collections[split_name][target_name].to(device)
        self.device = device
        return self

    def save_checkpoint(self, file_name: str) -> None:#เก็บ model optimize scale epoch threshold 
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.gradient_scaler.state_dict(),
            "epoch": self.epoch,
            "db": self.best_db,
        }
        torch.save(checkpoint, self.experiment_path / file_name)
        pprint("Saved checkpoint to {}".format(self.experiment_path / file_name))

    def load_checkpoint(self, file_name: str) -> None:

        checkpoint = torch.load(self.experiment_path / file_name,map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.gradient_scaler.load_state_dict(checkpoint["scaler"])
        self.epoch = checkpoint["epoch"]
        self.best_db = checkpoint["db"]
        print(self.best_db)
        

        pprint("Loaded checkpoint from {}".format(self.experiment_path / file_name))

    def save_transforms(self) -> None:
        """Save text tokenizer and label encoder"""

        #target2index json file 
        #label2index json file 
        self.dataloaders["train"].dataset.text_transform.save(self.experiment_path)
        self.dataloaders["train"].dataset.label_transform.save(self.experiment_path)

    def save_retrieval(self) -> None:
        
        
       # Define the dataloader keys and corresponding file names
        keys_and_files = {
            'train': 'train_targets.feather',
            'val': 'val_targets.feather',
            'test': 'test_targets.feather'
        }

        # Loop through the dataloader keys and file names
        for key, file_name in keys_and_files.items():
            # Collect all targets from the dataloader into a single tensor
            all_targets = torch.cat([batch.targets for batch in self.dataloaders[key]])

            # Save directly as a .feather file
            pd.DataFrame(all_targets.numpy()).to_feather(self.experiment_path / file_name)
            pprint("Saved retrieval to {}".format(self.experiment_path / file_name))
        del all_targets
       

    def load_retrieval(self,file_name):
        # Load the .feather file into a Pandas DataFrame
        retrieve = pd.read_feather(self.experiment_path / file_name)
        
        # Convert the DataFrame back to a PyTorch tensor
        retrieve = torch.tensor(retrieve.values).to(self.device)

        pprint("loaded retrieval to {}".format(self.experiment_path / file_name))
        return retrieve
     
    def save_final_model(self) -> None:
        self.save_checkpoint("final_model.pt")
        self.save_transforms() #เก็บ text_transform label_transform
        self.save_retrieval()
          
        OmegaConf.save(self.config, self.experiment_path / "config.yaml")