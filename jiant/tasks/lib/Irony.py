import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Union
from jiant.tasks.core import (
    BaseExample,
    BaseTokenizedExample,
    BaseDataRow,
    BatchMixin,
    GlueMixin,
    Task,
    TaskTypes,
)
from jiant.tasks.lib.templates.shared import single_sentence_featurize, labels_to_bimap
from jiant.utils.python.io import read_jsonl
import pandas as pd


@dataclass
class Example(BaseExample):
    guid: str
    text: str
    label: str

    def tokenize(self, tokenizer):
        return TokenizedExample(
            guid=self.guid,
            text=tokenizer.tokenize(self.text),
            label_id=Ironytask.LABEL_TO_ID[self.label],
        )

@dataclass
class TokenizedExample(BaseTokenizedExample):
    guid: str
    text: List
    label_id: int

    def featurize(self, tokenizer, feat_spec):
        return single_sentence_featurize(
            guid=self.guid,
            input_tokens=self.text,
            label_id=self.label_id,
            tokenizer=tokenizer,
            feat_spec=feat_spec,
            data_row_class=DataRow,
        )

@dataclass
class DataRow(BaseDataRow):
    guid: str
    input_ids: np.ndarray
    input_mask: np.ndarray
    segment_ids: np.ndarray
    label_id: int
    tokens: list

@dataclass
class Batch(BatchMixin):
    input_ids: torch.LongTensor
    input_mask: torch.LongTensor
    segment_ids: torch.LongTensor
    label_id: torch.LongTensor
    tokens: list

class Ironytask(Task):
    Example = Example
    TokenizedExample = Example
    DataRow = DataRow
    Batch = Batch

    TASK_TYPE = TaskTypes.CLASSIFICATION
    LABELS = [0, 1]
    LABEL_TO_ID, ID_TO_LABEL = labels_to_bimap(LABELS)

    def get_train_examples(self):
        return self._create_examples(path=self.train_path, set_type="train")

    def get_val_examples(self):
        return self._create_examples(path=self.val_path, set_type="val")

    def get_test_examples(self):
        return self._create_examples(path=self.test_path, set_type="test")

    @classmethod
    def _create_examples(cls, path, set_type):
        examples = []
        df = pd.read_csv(path, names=["sentence", "label"], sep="\t", header=0)
        for i, row in df.iterrows():
            examples.append(
                Example(
                    guid="%s-%s" % (set_type, i),
                    text=row.sentence,
                    label=row.label if set_type != "test" else cls.LABELS[-1],
                )
            )
        return examples