# coding: utf-8
"""
Data module
"""
from torchtext_compat import Dataset, Field, RawField, Iterator, BucketIterator, Example
from typing import List, Tuple
import pickle
import gzip
import torch
import joblib
import ipdb


def load_dataset_file(filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = joblib.load(f)
        # ipdb.set_trace()
        return loaded_object


class SignTranslationDataset(Dataset):
    """Defines a dataset for machine translation."""

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.sgn), len(ex.txt))

    def __init__(
        self,
        path: str,
        fields: Tuple[RawField, RawField, Field, Field, Field],
        **kwargs
    ):
        """Create a SignTranslationDataset given paths and fields.

        Arguments:
            path: Common prefix of paths to the data files for both languages.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        if not isinstance(fields[0], (tuple, list)):
            fields = [
                ("sequence", fields[0]),
                ("signer", fields[1]),
                ("sgn", fields[2]),
                ("gls", fields[3]),
                ("txt", fields[4]),
            ]

        if not isinstance(path, list):
            path = [path]

        samples = {}
        for annotation_file in path:
            tmp = load_dataset_file(annotation_file)
            for s in tmp:
                seq_id = s["name"]
                if seq_id in samples:
                    assert samples[seq_id]["name"] == s["name"]
                    assert samples[seq_id]["signer"] == s["signer"]
                    assert samples[seq_id]["gloss"] == s["gloss"]
                    assert samples[seq_id]["text"] == s["text"]
                    samples[seq_id]["sign"] = torch.cat(
                        [samples[seq_id]["sign"], s["sign"]], axis=1
                    )
                else:
                    samples[seq_id] = {
                        "name": s["name"],
                        "signer": s["signer"],
                        "gloss": s["gloss"],
                        "text": s["text"],
                        "sign": s["sign"],
                    }

        max_sent_length = kwargs.pop("max_sent_length", None)
        examples = []
        for s in samples:
            sample = samples[s]

            gloss_len = len(sample["gloss"].strip().split())
            text_len = len(sample["text"].strip().split())
            sign_len = sample["sign"].shape[0]

            if max_sent_length is not None:
                if sign_len > max_sent_length or text_len > max_sent_length:
                    continue  # skip this sample

            examples.append(
                Example.fromlist(
                    [
                        sample["name"],
                        sample["signer"],
                        sample["sign"] + 1e-8,
                        sample["gloss"].strip(),
                        sample["text"].strip(),
                    ],
                    fields,
                )
            )

        super().__init__(examples, fields, **kwargs)
