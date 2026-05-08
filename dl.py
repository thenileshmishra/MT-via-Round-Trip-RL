import os
import torch
from torch.utils.data import DataLoader, Sampler
from torchdata.datapipes.map import MapDataPipe
from pytorch_lightning import LightningDataModule
import warnings
from datasets import load_dataset

warnings.filterwarnings("ignore", ".*does not have many workers.*")


def _load_local_split(path):
    """Load a local CSV/TSV/JSON(L) file as a HF Dataset split.

    Expected columns: sentence_<source_lang> and sentence_<target_lang>.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        ds = load_dataset("csv", data_files=path)
    elif ext == ".tsv":
        ds = load_dataset("csv", data_files=path, delimiter="\t")
    elif ext in (".json", ".jsonl"):
        ds = load_dataset("json", data_files=path)
    else:
        raise ValueError(f"Unsupported file extension '{ext}' for {path}")
    return ds["train"]


class TranslationDataModule(LightningDataModule):
    def __init__(
        self,
        tokenizer,
        illegal_token_mask,
        data_path,
        dataset_config_name,
        source_lang,
        target_lang,
        sort_by_length: bool = True,
        sort_direction: str = "asc",
        train_batch_size: int = 1,
        train_file: str = None,
        valid_file: str = None,
        test_file: str = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore="tokenizer")
        self.tokenizer = tokenizer
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self._train_sampler = None
        self._train_batch_size = max(int(train_batch_size), 1)

    def setup(self, stage=None):
        src_col = "sentence_" + self.hparams.source_lang
        tgt_col = "sentence_" + self.hparams.target_lang

        train_file = self.hparams.train_file
        valid_file = self.hparams.valid_file
        test_file = self.hparams.test_file
        use_local = any(f is not None for f in (train_file, valid_file, test_file))

        if use_local:
            if train_file is None or valid_file is None:
                raise ValueError(
                    "When using local files, at least train_file and valid_file must be provided."
                )
            train_split = _load_local_split(train_file)
            val_split = _load_local_split(valid_file)
            test_split = _load_local_split(test_file) if test_file is not None else None
        else:
            prompts = load_dataset(
                self.hparams.data_path,
                self.hparams.dataset_config_name,
                trust_remote_code=True,
            )

            def _resolve_split(split_names):
                for split_name in split_names:
                    if split_name in prompts:
                        return prompts[split_name]
                return None

            train_split = _resolve_split(("train", "training"))
            val_split = _resolve_split(("valid", "validation", "val"))
            test_split = _resolve_split(("test", "test_final"))

        if train_split is None:
            raise ValueError("Training split not found in dataset.")
        if val_split is None:
            raise ValueError("Validation split not found in dataset.")

        self.train_data = TranslationDataPipe(train_split, self.tokenizer, src_col, tgt_col)
        self.val_data = TranslationDataPipe(val_split, self.tokenizer, src_col, tgt_col)
        self.test_data = (
            TranslationDataPipe(test_split, self.tokenizer, src_col, tgt_col)
            if test_split is not None
            else None
        )

    def train_dataloader(self):
        if self._train_sampler is not None:
            return DataLoader(
                self.train_data,
                sampler=self._train_sampler,
                batch_size=self._train_batch_size,
                num_workers=0,
                collate_fn=self._collate_batch,
            )
        return DataLoader(
            self.train_data,
            shuffle=True,
            batch_size=self._train_batch_size,
            num_workers=0,
            collate_fn=self._collate_batch,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=48,
            num_workers=0,
            collate_fn=self._collate_batch,
        )

    def test_dataloader(self):
        if self.test_data is None:
            return None
        return DataLoader(
            self.test_data,
            batch_size=48,
            num_workers=0,
            collate_fn=self._collate_batch,
        )

    def _collate_batch(self, batch):
        encoder_texts, targets, sources, sample_ids = zip(*batch)
        batch_encoding = self.tokenizer(
            list(encoder_texts),
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return batch_encoding, list(targets), list(sources), list(sample_ids)


class TranslationDataPipe(MapDataPipe):
    def __init__(self, prompts, tokenizer, src_col, tgt_col) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.src_col = src_col
        self.tgt_col = tgt_col

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index):
        src_text = self.prompts[index][self.src_col]
        tgt_text = self.prompts[index][self.tgt_col]
        return src_text, tgt_text, src_text, index
