import os
from dataclasses import dataclass
from functools import reduce
from typing import Any, Callable, Optional

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch
from rich.progress import track
from tqdm import tqdm

from src.settings import ID_COLUMN, TARGET_COLUMN, TEXT_COLUMN

import multiprocessing as mp
import psutil

@dataclass
class Lookups:
    data_info: dict[str, Any]
    code2description: Optional[dict[str, str]] = None
    code_system2code_indices: Optional[dict[str, torch.Tensor]] = None
    split2code_indices: Optional[dict[str, torch.Tensor]] = None


@dataclass
class Data:
    """Dataclass containing the dataset and the code occurrences of each code system."""

    df: pa.Table
    code_system2code_counts: dict[str, dict[str, int]]

    


    @property
    def train(self) -> list[pa.RecordBatch]:
        """Get the training data.

        Returns:
            list[pa.RecordBatch]: List of record batches.
        """
        batches = (
            self.df.filter(pc.field("split") == "train")
            .sort_by("num_words")
            .to_batches(max_chunksize=1)
        )
        return self.from_batches_to_list(batches,"train")

    @property
    def val(self) -> list[pa.RecordBatch]:
        """Get the validation data.

        Returns:
            list[pa.RecordBatch]: List of record batches.
        """
        batches = (
            self.df.filter(pc.field("split") == "val")
            .sort_by("num_words")
            .to_batches(max_chunksize=1)
        )

        return self.from_batches_to_list(batches, "val")

    @property
    def test(self) -> list[pa.RecordBatch]:
        """Get the test data.

        Returns:
            list[pa.RecordBatch]: List of record batches.
        """
        batches = (
            self.df.filter(pc.field("split") == "test")
            .sort_by("num_words")
            .to_batches(max_chunksize=1)
        )
        
        return self.from_batches_to_list(batches,"test")

    def from_batches_to_list(
        self,
        batches: list[pa.RecordBatch],
        split: str
    ) -> list[tuple[torch.Tensor, np.array, str, int, torch.Tensor]]:
        """Convert a list of record batches to a list of tuples

        Args:
            batches (list[pa.RecordBatch]): List of record batches.

        Returns:
            list[tuple[torch.Tensor, np.array, str, int, torch.Tensor]]: List of tuples.
        """
        examples = []
        import pdb

        for batch in tqdm(batches, desc=f"Creating examples {split}"):
            token_ids = torch.from_numpy(
                batch.column("token_ids")
                .flatten()
                .to_numpy(zero_copy_only=False, writable=True)
            )
            
            targets = (
                batch.column(TARGET_COLUMN)
                .flatten()
                .to_numpy(zero_copy_only=False, writable=True)
            )
            
            id = batch.column(ID_COLUMN)[0].as_py()
            num_tokens = len(token_ids)
            attention_mask = torch.ones_like(token_ids)
            examples.append((token_ids, targets, id, num_tokens, attention_mask))
        return examples

    @property
    def get_documents(self) -> list[str]:
        """Get all the documents in the dataset.

        Returns:
            list[str]: List of documents.
        """
        return self.df.column(TEXT_COLUMN).to_pylist()

    @property
    def all_target_counts(self) -> dict[str, int]:
        """Get the number of occurrences of each code in the dataset.

        Returns:
            dict[str, int]: Dictionary with the number of occurrences of each code.
        """
        return reduce(lambda x, y: {**x, **y}, self.code_system2code_counts.values())

    @property
    def get_train_documents(self) -> list[str]:
        """Get the training documents."""
        return (
            self.df.filter(pc.field("split") == "train").column(TEXT_COLUMN).to_pylist()
        )

    def split_targets(self, name: str) -> set[str]:
        """Get the targets of a split."""
        return set(
            self.df.filter(pc.field("split") == name)
            .column(TARGET_COLUMN)
            .combine_chunks()
            .flatten()
            .unique()
            .to_pylist()
        )

    def split_size(self, name: str) -> int:
        """Get the size of a split."""
        return len(self.df.filter(pc.field("split") == name))

    def num_split_targets(self, name: str) -> int:
        """Get the number of targets of a split."""
        return len(self.split_targets(name))

    @property
    def all_targets(self) -> set[str]:
        """Get all the targets in the dataset.

        Returns:
            set[str]: Set of all targets.
        """
        all_codes = set()
        for codesystem in self.code_system2code_counts.values():
            all_codes |= set(codesystem.keys())
        return all_codes

    @property
    def info(self) -> dict[str, int]:
        """Get information about the dataset.

        Returns:
            dict[str, int]: Dictionary with information about the dataset.
        """
        return {
            "num_classes": len(self.all_targets),
            "num_examples": len(self.df),
            "num_train_tokens": sum(
                self.df.filter(pc.field("split") == "train")
                .column("num_words")
                .to_pylist()
            ),
            "average_tokens_per_example": sum(self.df.column("num_words").to_pylist())
            / len(self.df),
            "num_train_examples": self.split_size("train"),
            "num_val_examples": self.split_size("val"),
            "num_test_examples": self.split_size("test"),
            "num_train_classes": self.num_split_targets("train"),
            "num_val_classes": self.num_split_targets("val"),
            "num_test_classes": self.num_split_targets("test"),
            "average_classes_per_example": sum(
                [
                    sum(codesystem.values())
                    for codesystem in self.code_system2code_counts.values()
                ]
            )
            / len(self.df),
        }
    #จำกัดความยาว
    def truncate_text(self, max_length: int) -> None:
        """Truncate text to a maximum length.

        Args:
            max_length (int): Maximum length of text.
        """
        if max_length is None:
            return

        text = self.df.column(TEXT_COLUMN)
        text_split = pc.utf8_split_whitespace(text)  # pylint: disable=no-member
        text_split_df = text_split.to_pandas().apply(lambda x: x[:max_length])
        text_split_truncate = pa.array(text_split_df.values)
        text_truncate = pc.binary_join(  # pylint: disable=no-member
            text_split_truncate, " "
        )

        # Change column in table
        new_table_no_text = self.df.drop([TEXT_COLUMN])
        new_table = new_table_no_text.append_column(
            pa.field(TEXT_COLUMN, pa.string()), text_truncate
        )
        del self.df
        self.df = new_table

    def transform_text(self, batch_transform: Callable[[list[str]], str]) -> None:
        """Transform the text using a batch transform function.

        Args:
            batch_transform (Callable[[list[str]], str]): Batch transform function.
        """
        max_chunksize=len(self.df)/(mp.cpu_count()-2) if len(self.df)> mp.cpu_count()-2 else len(self.df)
        token_ids_list = []
        with mp.Pool(mp.cpu_count()-2) as pool:
            results = []

            for batch in tqdm(
                self.df.to_batches(max_chunksize=max_chunksize), desc="Transforming text..."
            ):
                texts = batch.column(TEXT_COLUMN).to_pylist()

                # args should be passed as a tuple
                results.append(pool.apply_async(batch_transform, args=(texts,)))
            pool.close()
            pool.join()
            # Collect the results from the workers
            for result in tqdm(results, desc="Collecting results..."):
                token_ids_list += result.get()
                
            
      
        # Convert to list of pyarrays
        token_ids = pa.array(token_ids_list, type=pa.list_(pa.int64()))
        del token_ids_list
        # Append to table
        new_table_no_text = self.df.drop([TEXT_COLUMN])
        del self.df
        new_table = new_table_no_text.append_column(
            pa.field("token_ids", pa.list_(pa.int64())), [token_ids]
        )
        self.df = new_table
        os.environ["TOKENIZERS_PARALLELISM"] = "False"

@dataclass
class Batch:
    """Batch class. Used to store a batch of data."""

    data: torch.Tensor
    targets: torch.Tensor
    ids: torch.Tensor
    code_descriptions: Optional[torch.Tensor] = None
    num_tokens: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None

    def to(self, device: Any) -> "Batch":
        """Move the batch to a device.

        Args:
            device (Any): Device to move the batch to.

        Returns:
            self: Moved batch.
        """
        self.data = self.data.to(device, non_blocking=True)
        self.targets = self.targets.to(device, non_blocking=True)
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.to(device, non_blocking=True)
        return self

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.data = self.data.pin_memory()
        self.targets = self.targets.pin_memory()
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.pin_memory()
        return self
