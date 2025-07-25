# coding: utf-8
"""
Data module
"""
import os
import sys
import random

import torch
from torchtext_compat import Dataset, Field, RawField, Iterator, BucketIterator
from dataset import SignTranslationDataset
from vocabulary import (
    build_vocab,
    Vocabulary,
    UNK_TOKEN,
    EOS_TOKEN,
    BOS_TOKEN,
    PAD_TOKEN,
)


def load_data(data_cfg: dict) -> (Dataset, Dataset, Dataset, Vocabulary, Vocabulary):
    """
    Load train, dev and optionally test data as specified in configuration.
    Vocabularies are created from the training set with a limit of `voc_limit`
    tokens and a minimum token frequency of `voc_min_freq`
    (specified in the configuration dictionary).

    The training data is filtered to include sentences up to `max_sent_length`
    on source and target side.

    If you set ``random_train_subset``, a random selection of this size is used
    from the training set instead of the full training set.

    If you set ``random_dev_subset``, a random selection of this size is used
    from the dev development instead of the full development set.

    :param data_cfg: configuration dictionary for data
        ("data" part of configuration file)
    :return:
        - train_data: training dataset
        - dev_data: development dataset
        - test_data: test dataset if given, otherwise None
        - gls_vocab: gloss vocabulary extracted from training data
        - txt_vocab: spoken text vocabulary extracted from training data
    """

    data_path = data_cfg.get("data_path", "./data")

    if isinstance(data_cfg["train"], list):
        train_paths = [os.path.join(data_path, x) for x in data_cfg["train"]]
        dev_paths = [os.path.join(data_path, x) for x in data_cfg["dev"]]
        test_paths = [os.path.join(data_path, x) for x in data_cfg["test"]]
        ground_dev_paths = os.path.join(data_path, data_cfg["ground_dev"])
        ground_test_paths = os.path.join(data_path, data_cfg["ground_test"])
        pad_feature_size = sum(data_cfg["feature_size"])

    else:
        train_paths = os.path.join(data_path, data_cfg["train"])
        dev_paths = os.path.join(data_path, data_cfg["dev"])
        test_paths = os.path.join(data_path, data_cfg["test"])
        ground_dev_paths = os.path.join(data_path, data_cfg["ground_dev"])
        ground_test_paths = os.path.join(data_path, data_cfg["ground_test"])
        pad_feature_size = data_cfg["feature_size"]

    level = data_cfg["level"]
    txt_lowercase = data_cfg["txt_lowercase"]
    max_sent_length = data_cfg["max_sent_length"]

    def tokenize_text(text):
        # Skip if already a list of strings
        if isinstance(text, list) and all(isinstance(t, str) for t in text):
            return text
        if level == "char":
            return list(text)
        else:
            return text.strip().split()




    def tokenize_features(features):
        FEATURE_DIM = 150

        if features is None:
            raise ValueError("[tokenize_features] Received None")

        if isinstance(features, torch.Tensor):
            if features.dim() == 2 and features.shape[1] == FEATURE_DIM:
                return [frame for frame in features]  # return as list of frames
            if features.dim() == 1:
                if features.shape[0] % FEATURE_DIM != 0:
                    raise ValueError("Can't reshape to [T, FEATURE_DIM]")
                return [frame for frame in features.view(-1, FEATURE_DIM)]
            raise ValueError("Unrecognized tensor shape")

        if isinstance(features, list):
            if all(isinstance(f, list) and len(f) == FEATURE_DIM for f in features):
                return [torch.tensor(f, dtype=torch.float32) for f in features]
            if all(isinstance(f, torch.Tensor) and f.shape == (FEATURE_DIM,) for f in features):
                return features
            if all(isinstance(f, (int, float)) for f in features):
                if len(features) % FEATURE_DIM != 0:
                    raise ValueError("Cannot reshape flat list")
                return [frame for frame in torch.tensor(features, dtype=torch.float32).view(-1, FEATURE_DIM)]

        raise ValueError(f"[tokenize_features] Invalid input: {type(features)}, {getattr(features, 'shape', 'N/A')}")









    # NOTE (Cihan): The something was necessary to match the function signature.
    def stack_features(batch_features, _):
        stacked = [torch.stack(feat, dim=0) for feat in batch_features]
        lengths = torch.tensor([x.shape[0] for x in stacked], dtype=torch.long)
        padded = torch.nn.utils.rnn.pad_sequence(stacked, batch_first=True)
        return padded, lengths
    
    
    def stack_text_sequences(batch_seqs, _):
        #print("[DEBUG] Raw batch_seqs example:", batch_seqs[0], type(batch_seqs[0]))

        # Convert strings to list of token ids if needed
        if isinstance(batch_seqs[0], str):
            raise ValueError("❌ [stack_text_sequences] Got a string instead of a list of token ids.")

        # This assumes batch_seqs is a list of lists of ints (token ids)
        batch_seqs = [torch.tensor(seq, dtype=torch.long) for seq in batch_seqs]

        lengths = torch.tensor([len(seq) for seq in batch_seqs], dtype=torch.long)
        padded = torch.nn.utils.rnn.pad_sequence(batch_seqs, batch_first=True, padding_value=0)

        return padded, lengths







    sequence_field = RawField()
    signer_field = RawField()

    sgn_field = Field(
                        use_vocab=False,
                        init_token=None,
                        dtype=torch.float32,
                        tokenize=tokenize_features,
                        #preprocessing=tokenize_features,
                        postprocessing=stack_features,
                        batch_first=True,
                        include_lengths=True,
                        pad_token=torch.zeros((pad_feature_size,)),
                )

    gls_field = Field(
        use_vocab=True,
        pad_token=PAD_TOKEN,
        tokenize=tokenize_text,
        postprocessing=stack_text_sequences,
        batch_first=True,
        lower=False,
        include_lengths=True,
    )

    txt_field = Field(
        use_vocab=True,
        init_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        pad_token=PAD_TOKEN,
        tokenize=tokenize_text,
        postprocessing=stack_text_sequences,
        unk_token=UNK_TOKEN,
        batch_first=True,
        lower=txt_lowercase,
        include_lengths=True,
    )

    train_data = SignTranslationDataset(
                                        path=train_paths,
                                        fields=(sequence_field, signer_field, sgn_field, gls_field, txt_field),
                                        max_sent_length=max_sent_length  # pass this as a keyword argument
                 )


    gls_max_size = data_cfg.get("gls_voc_limit", sys.maxsize)
    gls_min_freq = data_cfg.get("gls_voc_min_freq", 1)
    txt_max_size = data_cfg.get("txt_voc_limit", sys.maxsize)
    txt_min_freq = data_cfg.get("txt_voc_min_freq", 1)

    gls_vocab_file = data_cfg.get("gls_vocab", None)
    txt_vocab_file = data_cfg.get("txt_vocab", None)

    #print(">>> First example txt raw:", train_data[0].txt)

    gls_vocab = build_vocab(
        field="gls",
        min_freq=gls_min_freq,
        max_size=gls_max_size,
        dataset=train_data,
        vocab_file=gls_vocab_file,
    )
    txt_vocab = build_vocab(
        field="txt",
        min_freq=txt_min_freq,
        max_size=txt_max_size,
        dataset=train_data,
        vocab_file=txt_vocab_file,
    )
    random_train_subset = data_cfg.get("random_train_subset", -1)
    if random_train_subset > -1:
        # select this many training examples randomly and discard the rest
        keep_ratio = random_train_subset / len(train_data)
        keep, _ = train_data.split(
            split_ratio=[keep_ratio, 1 - keep_ratio], random_state=random.getstate()
        )
        train_data = keep

    dev_data = SignTranslationDataset(
        path=dev_paths,
        fields=(sequence_field, signer_field, sgn_field, gls_field, txt_field),
    )
    random_dev_subset = data_cfg.get("random_dev_subset", -1)
    if random_dev_subset > -1:
        # select this many development examples randomly and discard the rest
        keep_ratio = random_dev_subset / len(dev_data)
        keep, _ = dev_data.split(
            split_ratio=[keep_ratio, 1 - keep_ratio], random_state=random.getstate()
        )
        dev_data = keep

    ground_dev_data = SignTranslationDataset(
        path=ground_dev_paths,
        fields=(sequence_field, signer_field, sgn_field, gls_field, txt_field),
    )

    # check if target exists
    test_data = SignTranslationDataset(
        path=test_paths,
        fields=(sequence_field, signer_field, sgn_field, gls_field, txt_field),
    )

    ground_test_data = SignTranslationDataset(
        path=ground_test_paths,
        fields=(sequence_field, signer_field, sgn_field, gls_field, txt_field),
    )

    gls_field.vocab = gls_vocab
    txt_field.vocab = txt_vocab
    return train_data, dev_data, test_data, gls_vocab, txt_vocab, ground_dev_data, ground_test_data


# TODO (Cihan): I don't like this use of globals.
#  Need to find a more elegant solution for this it at some point.
# pylint: disable=global-at-module-level
global max_sgn_in_batch, max_gls_in_batch, max_txt_in_batch


# pylint: disable=unused-argument,global-variable-undefined
def token_batch_size_fn(new, count, sofar):
    """Compute batch size based on number of tokens (+padding)"""
    global max_sgn_in_batch, max_gls_in_batch, max_txt_in_batch
    if count == 1:
        max_sgn_in_batch = 0
        max_gls_in_batch = 0
        max_txt_in_batch = 0
    max_sgn_in_batch = max(max_sgn_in_batch, len(new.sgn))
    max_gls_in_batch = max(max_gls_in_batch, len(new.gls))
    max_txt_in_batch = max(max_txt_in_batch, len(new.txt) + 2)
    sgn_elements = count * max_sgn_in_batch
    gls_elements = count * max_gls_in_batch
    txt_elements = count * max_txt_in_batch
    return max(sgn_elements, gls_elements, txt_elements)


def make_data_iter(
    dataset: Dataset,
    batch_size: int,
    batch_type: str = "sentence",
    train: bool = False,
    shuffle: bool = False,
) -> Iterator:
    """
    Returns a torchtext iterator for a torchtext dataset.

    :param dataset: torchtext dataset containing sgn and optionally txt
    :param batch_size: size of the batches the iterator prepares
    :param batch_type: measure batch size by sentence count or by token count
    :param train: whether it's training time, when turned off,
        bucketing, sorting within batches and shuffling is disabled
    :param shuffle: whether to shuffle the data before each epoch
        (no effect if set to True for testing)
    :return: torchtext iterator
    """

    batch_size_fn = token_batch_size_fn if batch_type == "token" else None

    if train:
        # optionally shuffle and sort during training
        data_iter = BucketIterator(
            repeat=False,
            sort=False,
            dataset=dataset,
            batch_size=batch_size,
            batch_size_fn=batch_size_fn,
            train=True,
            sort_within_batch=True,
            sort_key=lambda x: len(x.sgn),
            shuffle=shuffle,
        )
    else:
        # don't sort/shuffle for validation/inference
        data_iter = BucketIterator(
            repeat=False,
            dataset=dataset,
            batch_size=batch_size,
            batch_size_fn=batch_size_fn,
            train=False,
            sort=False,
        )

    return data_iter
