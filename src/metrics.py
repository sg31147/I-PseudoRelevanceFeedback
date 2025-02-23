from copy import deepcopy
from typing import Optional

import numpy as np
import torch
from torchmetrics.classification import MultilabelAUROC

from src.utils import detach


class Metric: #ไว้ inheritance
    base_tags = set()
    _str_value_fmt = "<.3"
    higher_is_better = True
    batch_update = True #อย่าลืมว่าเราทำทีละ batch
    filter_codes = True

    def __init__(
        self,
        name: str,
        tags: set,
        number_of_classes: int,
        threshold: Optional[float] = None,
    ):
        
        self.name = name
        self.tags = self.base_tags if tags is None else (tags | self.base_tags)
        self.number_of_classes = number_of_classes
        self.device = "cpu"
        self.threshold = threshold
        self.reset()

    def update(self, batch: dict):
        """Update the metric from a batch"""
        raise NotImplementedError()

    def set_target_boolean_indices(self, target_boolean_indices: list[bool]):
        self.target_boolean_indices = target_boolean_indices

    def compute(self):
        """Compute the metric value"""
        raise NotImplementedError()

    def reset(self):
        """Reset the metric"""
        raise NotImplementedError()

    def to(self, device: str):
        self.device = device
        if self.threshold is not None:
            self.threshold = torch.tensor(self.threshold).clone().to(device)
        self.reset()
        return self

    def copy(self):
        return deepcopy(self)

    def set_number_of_classes(self, number_of_classes: int):
        self.number_of_classes = number_of_classes

#ถูกสร้างตั้งแต่  metric_collections   ยังไม่ configsetting train นะ 
class MetricCollection:#แยกกลุ่มมาหมด
    def __init__(
        self,
        metrics: list[Metric],
        code_indices: Optional[torch.Tensor] = None, #code เฉพาะกลุ่ม
        code_system_name: Optional[str] = None#ชื่อกลุ่มแ code system 
    ):
        
        #tensor([   1,    2,   11,  ..., 7935, 7936, 7940]) #พูดถึงทั้งหมดไม่ซ้ำหละ
        self.metrics = metrics
        self.code_system_name = code_system_name
        if code_indices is not None:
            # Get overlapping indices
            self.code_indices = code_indices.clone()
            self.set_number_of_classes(len(code_indices))
        else:
            self.code_indices = None
        self.reset()

    def set_number_of_classes(self, number_of_classes_split: int):
        """Sets the number of classes for metrics with the filter_codes attribute to the number of classes in the split.
        Args:
            number_of_classes_split (int): Number of classes in the split
        """
        for metric in self.metrics:
            
            if metric.filter_codes:
                metric.set_number_of_classes(number_of_classes_split)

    def to(self, device: str):


        self.metrics = [metric.to(device) for metric in self.metrics] #กลับเข้า cpu
        if self.code_indices is not None:
            self.code_indices = self.code_indices.to(device)
        return self

    def filter_batch(self, batch: dict) -> dict:
        if self.code_indices is None:
            return batch
        #กรองรหัสที่เราสนใจตามตำแหน่งเลย  code_indices มันคือตำแหน่งของข้อมูลในแต่ละชุดแล้ว code2index
        filter_batch = {}
        targets, logits = batch["targets"], batch["logits"]
        #ทาบ filter เอาเฉพาะ 

        #สนใจเฉพาะ index ของกลุ่ม เท่านั้น ส่วนอื่นๆ 
        filtered_targets = torch.index_select(targets, -1, self.code_indices)
        filtered_logits = torch.index_select(logits, -1, self.code_indices)
        # Elements in the batch with targets
        idx_targets = torch.sum(filtered_targets, dim=-1) > 0 
        # Remove all elements wihtout targets [T,F,T,F,F,F,T,F]
        filter_batch["targets"] = filtered_targets[idx_targets] 
        filter_batch["logits"] = filtered_logits[idx_targets]
        return filter_batch

    def filter_tensor(
        self, tensor: torch.Tensor, code_indices: torch.Tensor
    ) -> list[torch.Tensor]:
        if code_indices is None:
            return tensor
        return torch.index_select(tensor, -1, code_indices)

    def is_best(
        self,
        prev_best: Optional[torch.Tensor],
        current: torch.Tensor,
        higher_is_better: bool,
    ) -> bool:
        if higher_is_better:
            return prev_best is None or current > prev_best
        else:
            return prev_best is None or current < prev_best

    def update_best_metrics(self, metric_dict: dict[str, torch.Tensor]):
        for metric in self.metrics:
            if metric.name not in metric_dict:
                continue

            if self.is_best(
                self.best_metrics[metric.name],
                metric_dict[metric.name],
                metric.higher_is_better,
            ):
                self.best_metrics[metric.name] = metric_dict[metric.name]
                    #output
    def update(self, batch: dict):
       
        #ปกติจะไม่ filter
        for metric in self.metrics: 
            if metric.batch_update and not metric.filter_codes:
                metric.update(batch)
        #ทุกๆอันต้องโดน filter ทิ้งไวหมด
        filtered_batch = self.filter_batch(batch) # คืน [] [0] [0] [] [] [] [] [] มา

        #ตรงนี้หละจะเลือก metric.batch_update
        for metric in self.metrics:
            if metric.batch_update and metric.filter_codes:
                metric.update(filtered_batch)

    def compute(
        self,
        logits: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        

        metric_dict = {
            metric.name: metric.compute()
            for metric in self.metrics
            if metric.batch_update
        }
        #จะหลุดมา auc
        if logits is not None and targets is not None:
            # Compute the metrics for the whole dataset
            if self.code_indices is not None:

                logits_filtered = self.filter_tensor(logits, self.code_indices.cpu())
                targets_filtered = self.filter_tensor(targets, self.code_indices.cpu())
            #เพิ่ม AUC กรณี val ตรงนี้
           
            for metric in self.metrics:
                #ข้าม metric ตัวอื่นๆ 
                if metric.batch_update:
                    continue
                #AUC
                if metric.filter_codes and self.code_indices is not None:
                    metric_dict[metric.name] = metric.compute(
                        logits=logits_filtered, targets=targets_filtered
                    )
                  
                else:#metric.batch_update =false กลุ่มที่ไม่อยากให้ filter ปกติ โดน filter หมด 
                    metric_dict[metric.name] = metric.compute(
                        logits=logits, targets=targets
                    )
    
            metric_dict.update(
                {
                    metric.name: metric.compute(logits=logits, targets=targets)
                    for metric in self.metrics
                    if not metric.batch_update
                }
            )
   
        #อัพเดท metric
        self.update_best_metrics(metric_dict)
        return metric_dict

    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset()

    def reset(self):
        self.reset_metrics()
        self.best_metrics = {metric.name: None for metric in self.metrics}

    def get_best_metric(self, metric_name: str) -> dict[str, torch.Tensor]:
        return self.best_metrics[metric_name]

    def copy(self):
        return deepcopy(self)

    def set_threshold(self, threshold: float):
        for metric in self.metrics:
            if hasattr(metric, "threshold"):
                metric.threshold = threshold #ตั้งค้่าไว้
 

""" ------------Classification Metrics-------------"""


class ExactMatchRatio(Metric): #เปะทุก element 
    _str_value_fmt = "6.4"  # 6.4321

    def __init__(
        self,
        threshold: float = 0.5,
        name: str = "exact_match_ratio",
        tags: set[str] = None,
        number_of_classes: int = 0,
        filter_codes: bool = True,
    ):
        if not filter_codes:
            name = f"{name}_mullenbach"
        super().__init__(
            name=name,
            tags=tags,
            number_of_classes=number_of_classes,
            threshold=threshold,
        )
        self.filter_codes = filter_codes

    def update(self, batch: dict):
        logits, targets = detach(batch["logits"]), detach(batch["targets"]) #detach nodgraident ย้อนหลัง
        predictions = (logits > self.threshold).long()
        
        self._num_exact_matches += torch.all(#มันมาเป็น batch นะ เลยนับรวม batch  ไป 
            torch.eq(predictions, targets), dim=-1
        ).sum()
        self._num_examples += targets.size(0)

    def compute(self) -> torch.Tensor:
        return self._num_exact_matches / self._num_examples

    def reset(self):
        self._num_exact_matches = torch.tensor(0).to(self.device)
        self._num_examples = 0


class Recall(Metric): #หาเจอไหม
    _str_value_fmt = "6.4"  # 6.4321

    def __init__(
        self,
        number_of_classes: int,
        threshold: float = 0.5,
        average: str = "micro",
        name: str = "recall",
        tags: set[str] = None,
        filter_codes: bool = True,
    ):
        if average:
            name = f"{name}_{average}"
        if not filter_codes:
            name = f"{name}_mullenbach"
        super().__init__(
            name=name,
            tags=tags,
            number_of_classes=number_of_classes,
            threshold=threshold,
        )
        self._average = average
        self.filter_codes = filter_codes

    def update(self, batch: dict):
        logits, targets = detach(batch["logits"]), detach(batch["targets"])
        predictions = (logits > self.threshold).long()
        self._tp += torch.sum(predictions * targets, dim=0)
        self._fn += torch.sum((1 - predictions) * targets, dim=0)

    def compute(self) -> torch.Tensor:
        if self._average == "micro":
            return (self._tp.sum() / (self._tp.sum() + self._fn.sum() + 1e-10)).cpu()
        if self._average == "macro":
            return torch.mean(self._tp / (self._tp + self._fn + 1e-10)).cpu()
        if self._average is None or self._average == "none":
            return (self._tp / (self._tp + self._fn + 1e-10)).cpu()
        raise ValueError(f"Invalid average: {self._average}")

    def reset(self):
        self._tp = torch.zeros((self.number_of_classes)).to(self.device)
        self._fn = torch.zeros((self.number_of_classes)).to(self.device)


class Precision(Metric): #แม่นไหม
    _str_value_fmt = "6.4"  # 6.4321

    def __init__(
        self,
        number_of_classes: int,
        threshold: float = 0.5,
        average: str = "micro",
        name: str = "precision",
        tags: set[str] = None,
        filter_codes: bool = True,
    ):
        if average:
            name = f"{name}_{average}"
        if not filter_codes:
            name = f"{name}_mullenbach"
        super().__init__(
            name=name,
            tags=tags,
            number_of_classes=number_of_classes,
            threshold=threshold,
        )
        self._average = average
        self.filter_codes = filter_codes

    def update(self, batch: dict):
        logits, targets = detach(batch["logits"]), detach(batch["targets"])
        predictions = (logits > self.threshold).long()
        self._tp += torch.sum(predictions * targets, dim=0)
        self._fp += torch.sum((predictions) * (1 - targets), dim=0)

    def compute(self):
        if self._average == "micro": #คิดแยกไปทีละกลุ่ม
            return (self._tp.sum() / (self._tp.sum() + self._fp.sum() + 1e-10)).cpu()
        if self._average == "macro": #เหมาเข่งเฉลี่ยทีเดยว
            return torch.mean(self._tp / (self._tp + self._fp + 1e-10)).cpu()
        if self._average is None or self._average == "none":
            return (self._tp / (self._tp + self._fp + 1e-10)).cpu()
        raise ValueError(f"Invalid average: {self._average}")

    def reset(self):
        self._tp = torch.zeros((self.number_of_classes)).to(self.device)
        self._fp = torch.zeros((self.number_of_classes)).to(self.device)


class FPR(Metric): #False positive rate มองหาคลาสเจอแบบผิดๆแค่ไหน
    _str_value_fmt = "6.4"  # 6.4321
    higher_is_better = False

    def __init__(
        self,
        number_of_classes: int,
        threshold: float = 0.5,
        average: str = "micro",
        name: str = "fpr",
        tags: set[str] = None,
        filter_codes: bool = True,
    ):
        if average:
            name = f"{name}_{average}"
        if not filter_codes:
            name = f"{name}_mullenbach"
        super().__init__(
            name=name,
            tags=tags,
            number_of_classes=number_of_classes,
            threshold=threshold,
        )
        self._average = average
        self.filter_codes = filter_codes

    def update(self, batch: dict):
        logits, targets = detach(batch["logits"]), detach(batch["targets"])
        predictions = (logits > self.threshold).long()
        self._fp += torch.sum(predictions * (1 - targets), dim=0)
        self._tn += torch.sum((1 - predictions) * (1 - targets), dim=0)

    def compute(self) -> torch.Tensor:
        if self._average == "micro":
            return (self._fp.sum() / (self._fp.sum() + self._tn.sum() + 1e-10)).cpu()
        if self._average == "macro":
            return torch.mean(self._fp / (self._fp + self._tn + 1e-10)).cpu()
        if self._average is None or self._average == "none":
            return (self._fp / (self._fp + self._tn + 1e-10)).cpu()
        raise ValueError(f"Invalid average: {self._average}")

    def reset(self):
        self._fp = torch.zeros((self.number_of_classes)).to(self.device)
        self._tn = torch.zeros((self.number_of_classes)).to(self.device)




class AUC(Metric):
    _str_value_fmt = "6.4"  # 6.4321
    batch_update = False

    def __init__(
        self,
        average: str = "micro",
        name: str = "auc",
        tags: set[str] = None,
        number_of_classes: Optional[int] = None,
        filter_codes: bool = True,
    ):
        """Area under the ROC curve. All classes that have no positive examples are ignored as implemented by Mullenbach et al. Please note that all the logits and targets are stored in the GPU memory if they have not already been moved to the CPU.

        Args:
            logits (torch.Tensor): logits from a machine learning model. [batch_size, num_classes]
            name (str, optional): name of the metric. Defaults to "auc".
            tags (set[str], optional): metrics tages. Defaults to None.
            log_to_console (bool, optional): whether to print this metric. Defaults to True.
            number_of_classes (int, optional): number of classes. Defaults to None.
            filter_codes (bool, optional): whether to filter out codes that have no positive examples. Defaults to True.
        """
        if average:
            name = f"{name}_{average}"
        if not filter_codes:
            name = f"{name}_mullenbach"
        super().__init__(name=name, tags=tags, number_of_classes=number_of_classes)
        self._average = average
        self.filter_codes = filter_codes

    
    def compute(self, logits: torch.Tensor, targets: torch.Tensor) -> np.float32:
        logits = detach(logits)
        targets = detach(targets).long()
         # Number of labels
        num_labels = logits.shape[1]

        # Compute ROC curve for each label
        # roc = MultilabelROC(num_labels=num_labels)
        # fpr, tpr, thresholds = roc(logits, targets)
        
        if self._average == "micro":
            auroc_micro = MultilabelAUROC(num_labels=num_labels, average='micro')
            value = auroc_micro(logits, targets)
            
        elif self._average == "macro":
            auroc_macro = MultilabelAUROC(num_labels=num_labels, average='macro')
            value = auroc_macro(logits, targets)
 
        return value.cpu()
    

    def update(self, batch: dict):
        raise NotImplementedError("AUC is not batch updateable.")


    def reset(self):
        pass

    
class F1Score(Metric):
    _str_value_fmt = "6.4"  # 6.4321

    def __init__(
        self,
        number_of_classes: int,
        threshold: float = 0.5,
        average: str = "micro",
        name: str = "f1",
        tags: set[str] = None,
        filter_codes: bool = True,
    ):
        if average:
            name = f"{name}_{average}"
        if not filter_codes:
            name = f"{name}_mullenbach"
        super().__init__(
            name=name,
            tags=tags,
            number_of_classes=number_of_classes,
            threshold=threshold,
        )
        self._average = average
        self.filter_codes = filter_codes

    def update(self, batch: dict):
   
        logits, targets = detach(batch["logits"]), detach(batch["targets"])
        predictions = (logits > self.threshold).long() #เกณฑ์คัดเลือก
        
        self._tp += torch.sum((predictions) * (targets), dim=0)
        self._fp += torch.sum(predictions * (1 - targets), dim=0)
        self._fn += torch.sum((1 - predictions) * targets, dim=0)

    def compute(self):
        if self._average == "micro":
            return (
                self._tp.sum()
                / (self._tp.sum() + 0.5 * (self._fp.sum() + self._fn.sum()) + 1e-10)
            ).cpu()
        if self._average == "macro":
            return torch.mean(
                self._tp / (self._tp + 0.5 * (self._fp + self._fn) + 1e-10)
            ).cpu()
        if self._average is None or self._average == "none":
            return (self._tp / (self._tp + 0.5 * (self._fp + self._fn) + 1e-10)).cpu()
        raise ValueError(f"Invalid average: {self._average}")

    def reset(self):
        self._tp = torch.zeros((self.number_of_classes)).to(self.device)
        self._fp = torch.zeros((self.number_of_classes)).to(self.device)
        self._fn = torch.zeros((self.number_of_classes)).to(self.device)


""" ------------Information Retrieval Metrics-------------"""





class Precision_K(Metric): #lสนใจ k กลุ่ม
    _str_value_fmt = "6.4"  # 6.4321

    def __init__(
        self,
        k: int = 10,
        name: str = "precision",
        tags: set[str] = None,
        number_of_classes: Optional[int] = None,
        filter_codes: bool = True,
    ):
        name = f"{name}@{k}"
        if not filter_codes:
            name = f"{name}_mullenbach"
        super().__init__(name=name, tags=tags, number_of_classes=number_of_classes)
        self._k = k
        self.filter_codes = filter_codes

    def update(self, batch: dict):
        logits, targets = detach(batch["logits"]), detach(batch["targets"])
        top_k = torch.topk(logits, dim=1, k=self._k)

        targets_k = targets.gather(1, top_k.indices)
        logits_k = torch.ones(targets_k.shape, device=targets_k.device)

        tp = torch.sum(logits_k * targets_k, dim=1)
        fp = torch.sum((logits_k) * (1 - targets_k), dim=1)
        self._num_examples += logits.size(0)
        self._precision_sum += torch.sum(tp / (tp + fp + 1e-10))

    def compute(self) -> torch.Tensor:
        return self._precision_sum.cpu() / self._num_examples

    def reset(self):
        self._num_examples = 0
        self._precision_sum = torch.tensor(0.0).to(self.device)


class MeanAveragePrecision(Metric): #เฉลี่ยแบบ accum ไป
    _str_value_fmt = "6.4"  # 6.4321

    def __init__(
        self,
        name: str = "map",
        tags: set[str] = None,
        number_of_classes: Optional[int] = None,
        filter_codes: bool = True,
    ):
        if not filter_codes:
            name = f"{name}_mullenbach"
        super().__init__(name=name, tags=tags, number_of_classes=number_of_classes)
        self.filter_codes = filter_codes

    def update(self, batch: dict):
        logits, targets = detach(batch["logits"]), detach(batch["targets"])
        _, indices = torch.sort(logits, dim=1, descending=True) #มากไปหน้อย
        sorted_targets = targets.gather(1, indices)
        sorted_targets_cum = torch.cumsum(sorted_targets, dim=1) # [1 2 3] => [1 3 6]
        batch_size = logits.size(0) #size
        denom = torch.arange(1, targets.shape[1] + 1, device=targets.device).repeat(
            batch_size, 1 # [1 2 3 4] [1 2 3 4]... n
        )
        prec_at_k = sorted_targets_cum / denom # [1/1 2/3 3/6]
        average_precision_batch = torch.sum(
            prec_at_k * sorted_targets, dim=1
        ) / torch.sum(sorted_targets, dim=1)
        self._average_precision_sum += torch.sum(average_precision_batch)
        self._num_examples += batch_size # batch ละ n

    def compute(self) -> torch.Tensor:
        return self._average_precision_sum.cpu() / self._num_examples

    def reset(self):
        self._num_examples = 0
        self._average_precision_sum = torch.tensor(0.0).to(self.device)


class Recall_K(Metric): #สนใจเฉพาะ k ที่ถูกกำหนดไว้เท่านั้น ตัวอื่นๆ ไม่สนใจเท่าไร
    _str_value_fmt = "6.4"  # 6.4321

    def __init__(
        self,
        k: int = 10,
        name: str = "recall",
        tags: set[str] = None,
        number_of_classes: Optional[int] = None,
        filter_codes: bool = True,
    ):
        name = f"{name}@{k}"
        if not filter_codes:
            name = f"{name}_mullenbach"
        super().__init__(name=name, tags=tags, number_of_classes=number_of_classes)
        self._k = k
        self.filter_codes = filter_codes

    def update(self, batch: dict):
        logits, targets = detach(batch["logits"]), detach(batch["targets"])
        top_k = torch.topk(logits, dim=1, k=self._k) #มีตัว k มาข้อเกี่ยวคือการเลือกแค่ top k พอ
    
        targets_k = targets.gather(1, top_k.indices)

        logits_k = torch.ones(targets_k.shape, device=targets_k.device)
        
        tp = torch.sum(logits_k * targets_k, dim=1) #คณเพื่อให้ได้ tp
        total_number_of_relevant_targets = torch.sum(targets, dim=1) #นับเฉพาะจำนวนที่ข้องเกี่ยว k เท่านั้น ฉะนั้นการคำนวน loss target ย่อมตัด target ที่ไม่สนใจออก

        self._num_examples += logits.size(0) #เก็บว่า batch ละจำนวนเท่าใด
        self._recall_sum += torch.sum(tp / (total_number_of_relevant_targets + 1e-10)) # เพิ่มราย batch เลย

    def compute(self) -> torch.Tensor:
        return self._recall_sum.cpu() / self._num_examples

    def reset(self):
        self._num_examples = 0 # นับใหม่หละ
        self._recall_sum = torch.tensor(0.0).to(self.device) # คำนวณใหม้


""" ------------Running Mean Metrics-------------"""


class RunningMeanMetric(Metric): # อติดตามค่าเฉลี่ยแบบสะสม (running mean) 
    _str_value_fmt = "<.3"

    def __init__(
        self,
        name: str,
        tags: set[str],
        number_of_classes: Optional[int] = None,
    ):
        """Create a running mean metric.

        Args:
            name (str): Name of the metric
            tags (Set[str]): Tags to use for grouping with other metrics.
            number_of_classes (Optional[int], optional): Number of classes. Defaults to None.
        """
        super().__init__(name=name, tags=tags, number_of_classes=number_of_classes)

    def update(self, batch: dict):
        raise NotImplementedError

    def update_value(
        self,
        values: torch.Tensor,
        reduce_by: Optional[torch.Tensor] = None,
        weight_by: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            values (torch.Tensor): Values of the metric
            reduce_by (Optional[torch.Tensor], optional): A single or per example divisor of the values. Defaults to batch size.
            weight_by (Optional[torch.Tensor], optional): A single or per example weights for the running mean. Defaults to `reduce_by`.
        """
        values = detach(values)
        reduce_by = detach(reduce_by)
 
        numel = values.numel() if isinstance(values, torch.Tensor) else 1
        value = values.sum().tolist() if isinstance(values, torch.Tensor) else values

        reduce_by = (
            reduce_by.sum().tolist()
            if isinstance(reduce_by, torch.Tensor)
            else (reduce_by or numel)
        )
    
        weight_by = (
            weight_by.sum().tolist()
            if isinstance(weight_by, torch.Tensor)
            else (weight_by or reduce_by)
        )
   
        values = value / reduce_by
        # self เก็บ weight เดิมไว้
        d = self.weight_by + weight_by # d ใหม่
        w1 = self.weight_by / d #w1ใหม่
        w2 = weight_by / d  #w2 ใหม่

        self._values = (
            self._values * w1 + values * w2
        )  # Reduce between batches (over entire epoch)

        self.weight_by = d

    def compute(self) -> torch.Tensor:
        return self._values

    def reset(self):
        self._values = torch.tensor(0.0).to(self.device)
        self.weight_by = torch.tensor(0.0).to(self.device)


class LossMetric(RunningMeanMetric):
    base_tags = {"losses"}
    higher_is_better = False

    def __init__(
        self,
        name: str = "loss",
        tags: set[str] = None,
        number_of_classes: Optional[int] = None,
        filter_codes: bool = False,
    ):
        super().__init__(
            name=name,
            tags=tags,
            number_of_classes=number_of_classes,
        )
        self.filter_codes = filter_codes

    def update(self, batch: dict[str, torch.Tensor]):
        loss = detach(batch["loss"]).cpu()
        self.update_value(loss, reduce_by=loss.numel(), weight_by=loss.numel())



class PrecisionAtRecall(Metric): 
    _str_value_fmt = "6.4"  # 6.4321

    def __init__(
        self,
        name: str = "precision@recall",
        tags: set[str] = None,
        number_of_classes: Optional[int] = None,
        filter_codes: bool = True,
    ):
        if not filter_codes:
            name = f"{name}_mullenbach"
        super().__init__(name=name, tags=tags, number_of_classes=number_of_classes)
        self.filter_codes = filter_codes

    def update(self, batch: dict):
        logits, targets = detach(batch["logits"]), detach(batch["targets"])
        num_targets = targets.sum(dim=1, dtype=torch.int64) # รวมกันในคำตอบแถวเดียว ของแต่ละ n 
        _, indices = torch.sort(logits, dim=1, descending=True) #จะกำหนดว่าใน logit เรียง มากไปน้อยเป็นตำแหน่งลำดับที่ ก็คือความน่าจะเป็นเยอะถูก select
        sorted_targets = targets.gather(1, indices) #เอาเฉพาะ target โดยอ้างอิงจาก Logit 
        sorted_targets_cum = torch.cumsum(sorted_targets, dim=1) # รวมแบบ cumu
        self._precision_sum += torch.sum(
            sorted_targets_cum.gather(1, num_targets.unsqueeze(1) - 1).squeeze()
            / num_targets
        )#ตำแหน่งไกล่โดนเฉลี่ย+ / เยอะ
        self._num_examples += logits.size(0)

    def compute(self) -> torch.Tensor:
        return self._precision_sum.cpu() / self._num_examples

    def reset(self):
        self._num_examples = 0
        self._precision_sum = torch.tensor(0.0).to(self.device)