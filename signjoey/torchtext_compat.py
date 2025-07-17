# torchtext_compat.py

import torch
from typing import List, Tuple, Callable
import collections
import random
from batch import Batch

def tokenize_text(text):
        # Skip if already a list of strings
        if isinstance(text, list) and all(isinstance(t, str) for t in text):
            return text
        if level == "char":
            return list(text)
        else:
            return text.strip().split()

FEATURE_DIM = 150
FEATURE_DIM = 150

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




    
class RawField:
    def __init__(self):
        pass

    def preprocess(self, x):
        return x


class Field:
    def __init__(
        self,
        use_vocab=True,
        init_token=None,
        eos_token=None,
        pad_token="<pad>",
        unk_token="<unk>",
        tokenize=None,
        preprocessing=None,
        postprocessing=None,
        batch_first=True,
        include_lengths=False,
        lower=False,
        dtype=torch.float32,
    ):
        self.use_vocab = use_vocab
        self.init_token = init_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.tokenize = tokenize or (lambda x: x)
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing
        self.batch_first = batch_first
        self.include_lengths = include_lengths
        self.lower = lower
        self.dtype = dtype
        self.vocab = None

    def preprocess(self, x):
        """
        Apply tokenization and preprocessing depending on the field type.
        - For non-text fields (use_vocab=False), apply tokenize/preprocessing per sample.
        - For text fields (use_vocab=True), apply batch-level tokenization/preprocessing.
        """
        if self.tokenize:
            x = self.tokenize(x)
        if self.preprocessing:
            x = self.preprocessing(x)
        return x
        
            




class Example:
    @classmethod
    def fromlist(cls, data: List, fields: List[Tuple[str, object]]):
        ex = cls()
        for (name, field), val in zip(fields, data):
            if field is not None:
                setattr(ex, name, field.preprocess(val))
            else:
                setattr(ex, name, val)
        return ex


class Dataset:
    def __init__(self, examples: List[Example], fields: List[Tuple[str, object]]):
        self.examples = examples
        self.fields = collections.OrderedDict(fields)

    def __len__(self):
        return len(self.examples)

    def split(self, split_ratio=0.7, random_state=None):
        import random
        if random_state:
            random.setstate(random_state)
        random.shuffle(self.examples)
        n = len(self.examples)
        n_split = int(n * split_ratio)
        return (
            Dataset(self.examples[:n_split], list(self.fields.items())),
            Dataset(self.examples[n_split:], list(self.fields.items())),
        )
        
    #def __getitem__(self, index):
        #print(f">>> __getitem__ called for index: {index}")
     #   return self.examples[index]
    
    #debug func
    def __getitem__(self, idx):
        item = self.examples[idx]
        item.idx = idx  # tag the sample with its index
        return item



from collections import namedtuple

class TTBatch:
    def __init__(self, data, fields):
        # Each field (e.g., 'sequence', 'signer', 'sgn', etc.)
        # will be converted into an attribute on this batch.
        #for x in data:
            #print(f"[DEBUG] Building batch — Example idx: {getattr(x, 'idx', 'N/A')}")

        for name, field in fields.items():
            if field is None:
                setattr(self, name, [getattr(x, name) for x in data])
            else:
                individual = [field.preprocess(getattr(x, name)) for x in data]

                # 🩹 Fix: Convert tokens to indices if using vocab
                if hasattr(field, "use_vocab") and field.use_vocab:
                    if isinstance(individual[0], list):  # batch of token lists
                        numericalized = [
                            [field.vocab.stoi.get(tok, field.vocab.stoi.get(field.unk_token, 0)) for tok in seq]
                            for seq in individual
                            ]
                    else:  # batch of single tokens
                        numericalized = [
                            field.vocab.stoi.get(tok, field.vocab.stoi.get(field.unk_token, 0))
                            for tok in individual
                        ]
                else:
                    numericalized = individual

                # 🔄 Run postprocessing (e.g., padding, lengths)
                if hasattr(field, "postprocessing") and field.postprocessing:
                    processed = field.postprocessing(numericalized, None)
                    setattr(self, name, processed)
                else:
                    setattr(self, name, numericalized)

               


                
                
class BucketIterator:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        sort_key: Callable = None,
        sort: bool = False,
        sort_within_batch: bool = False,
        batch_size_fn: Callable = None,
        shuffle: bool = False,
        train: bool = False,
        repeat: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.sort_key = sort_key
        self.shuffle = shuffle
        self.train = train
        self.repeat = repeat
        self.batch_size_fn = batch_size_fn
        self.sort_within_batch = sort_within_batch
        self.sort = sort

    
    def __iter__(self):
        data = self.dataset.examples

        if self.shuffle:
            random.shuffle(data)

        if self.sort and self.sort_key:
            data = sorted(data, key=self.sort_key)

        if self.batch_size_fn is None:
            for i in range(0, len(data), self.batch_size):
                minibatch = data[i : i + self.batch_size]
                yield TTBatch(minibatch, self.dataset.fields)  # ✅ wrap in Batch
        else:
            batch, size = [], 0
            for ex in data:
                batch.append(ex)
                size = self.batch_size_fn(ex, len(batch), size)
                if size >= self.batch_size:
                    yield TTBatch(batch, self.dataset.fields)  # ✅ wrap in Batch
                batch, size = [], 0
            if batch:
                yield TTBatch(batch, self.dataset.fields)  # ✅ final batch



Iterator = BucketIterator  # legacy alias

def interleave_keys(len1, len2):
    return int(''.join(f"{a}{b}" for a, b in zip(str(len1).zfill(4), str(len2).zfill(4))))
