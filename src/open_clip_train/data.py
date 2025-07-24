import ast
import json
import logging
import math
import os
import random
import sys
import braceexpand
from dataclasses import dataclass
from multiprocessing import Value
import time
from typing import Optional, Iterator

import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
import webdataset as wds
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info, Sampler, BatchSampler
from torch.utils.data.distributed import DistributedSampler
from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, is_train, base_folder, sep="\t", tokenizer=None):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        logging.debug('Done loading data.')

        self.tokenize = tokenizer
        self.base_dir = base_folder
        '''if is_train:
            self.base_dir = os.path.join(base_folder, "images")
        else:
            self.base_dir = os.path.join(base_folder, "val_images")'''

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        #images = self.transforms(Image.open(str(self.images[idx])))
        images = self.transforms(Image.open(os.path.join(self.base_dir, str(self.images[idx]))))
        texts = self.tokenize([str(self.captions[idx])])[0]
        return images, texts

class MultiCLIPCsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, is_train, base_folder, subfolders, sep="\t", tokenizer=None):
        logging.info(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        self.relative_paths = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        self.tokenize = tokenizer
        self.base_dir = base_folder
        self.subfolders = subfolders
        self.m = len(subfolders)
        logging.info(f"Loaded {len(self.captions)} unique captions with {self.m} views each. Each epoch will thus yield {len(self.captions) * self.m} samples.")

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        #images = self.transforms(Image.open(str(self.images[idx])))
        images = self.transforms(Image.open(os.path.join(self.base_dir, str(self.images[idx]))))
        texts = self.tokenize([str(self.captions[idx])])[0]
        return images, texts
    
    def __getitem__(self, idx):
        rel_path = self.relative_paths[idx]
        caption = self.captions[idx]
        
        images = []
        texts = []

        for sub in self.subfolders:
            full_path = os.path.join(self.base_dir, sub, rel_path)
            image = self.transforms(Image.open(full_path))
            images.append(image)
            texts.append(self.tokenize([str(caption)])[0])

        return images, texts  # length m each
    
def multi_positive_collate_fn(batch):
    """
    Collate function for multi-positive loss CLIP training.
    Args:
        batch: List of size n_captions_per_batch.
               Each element is a tuple: (List[image_tensor], List[text_tensor]) of length m.
    Returns:
        Tuple of:
          - images: Tensor of shape (n * m, C, H, W)
          - texts: Tensor of shape (n * m, seq_len)
    """
    images = []
    texts = []

    for img_list, txt_list in batch:
        images.extend(img_list)  # list of m image tensors
        texts.extend(txt_list)   # list of m tokenized caption tensors

    # Stack into batch tensors
    images = torch.stack(images, dim=0)
    texts = torch.stack(texts, dim=0)

    return images, texts

class MultiImageCsvDataset(Dataset):
    """
        Dataset for CSV files with captions and relative image paths, but with multiple image variations per caption.
        Each image is stored in a subfolder, so for each caption there are m images stored in
        subfolders[0]/df[image_key][idx], 
        subfolders[1]/df[image_key][idx], etc.
    """
    def __init__(self, input_filename, transforms, img_key, caption_key, is_train, 
                 base_folder, subfolders, sep="\t", tokenizer=None):
        """
        Args:
            input_filename: Path to CSV file with n caption-image pairs
            transforms: Image transforms to apply
            img_key: Column name for image paths in CSV
            caption_key: Column name for captions in CSV  
            is_train: Training flag (for compatibility)
            base_folder: Base directory containing subfolders
            subfolders: List of subfolder names, each containing image variations
            sep: CSV separator
            tokenizer: Text tokenizer
        """
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)
        
        # Original n caption-image pairs
        self.original_images = df[img_key].tolist()
        self.original_captions = df[caption_key].tolist()
        self.transforms = transforms
        self.tokenize = tokenizer
        self.base_dir = base_folder
        self.subfolders = subfolders
        
        # Create expanded dataset: n*m items total
        self.images = []
        self.captions = []
        self.caption_ids = []  # To track which original caption this belongs to
        
        print(f'Loading {len(self.original_captions)} captions with {len(subfolders)} variations each.')
        for caption_idx, (original_image, caption) in enumerate(zip(self.original_images, self.original_captions)):
            for subfolder in subfolders:
                # Full path for this image variation
                image_path = os.path.join(self.base_dir, subfolder, str(original_image))
                self.images.append(image_path)
                self.captions.append(caption)
                self.caption_ids.append(caption_idx)  # Track original caption index
        print(f'Loaded {len(self.images)} total image-caption pairs.')

        self.n = len(self.original_captions)  # Number of unique captions
        self.m = len(subfolders)  # Number of variations per caption
        
        logging.debug(f'Done loading data. {self.n} unique captions with {self.m} variations each = {len(self)} total items.')
    
    def __len__(self):
        # Return n*m (total number of image-caption pairs)
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load and transform image
        images = self.transforms(Image.open(self.images[idx]))
        # Tokenize caption
        texts = self.tokenize([str(self.captions[idx])])[0]
        return images, texts
    
    def get_caption_id(self, idx):
        """Get the original caption ID for a given dataset index"""
        return self.caption_ids[idx]

class MultiImageDistributedBatchSampler(BatchSampler):
    """
    BatchSampler for n captions x m variations, in distributed training.
    Guarantees:
      - No two samples in a batch share the same caption.
      - All n*m indices are seen each epoch (modulo drop_last).
      - Work is evenly sharded across `num_replicas`.
    """
    def __init__(self,
                 dataset,
                 batch_size: int,
                 num_replicas: int = None,
                 rank: int = None,
                 shuffle: bool = True,
                 drop_last: bool = False,
                 seed: int = 0):
        """
        Args:
          dataset: must have attributes `.n` (# captions) and `.m` (# per‑caption variations)
          batch_size: batch size PER RANK
          num_replicas, rank: from torch.distributed (or None for single‑GPU)
          shuffle: whether to reshuffle each epoch
          drop_last: if True, drop the final partial batch on each rank
          seed: base seed
        """
        if num_replicas is None:
            # fallback to 1‑GPU
            num_replicas, rank = 1, 0
        super().__init__(sampler=None, batch_size=batch_size, drop_last=drop_last)

        self.n = dataset.n
        self.m = dataset.m
        self.batch_size = batch_size

        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        self.total_size = self.n * self.m

    def __iter__(self):
        logging.debug(f'BatchSampler epoch {self.epoch}')
        # 1) Shuffle captions and their variations
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        caption_order = torch.randperm(self.n, generator=g).tolist() if self.shuffle else list(range(self.n))
        queues = {
            ci: (torch.randperm(self.m, generator=g).tolist() if self.shuffle else list(range(self.m)))
            for ci in range(self.n)
        }

        # 2) Build the full interleaved list
        all_indices = []
        while any(queues.values()):
            for ci in caption_order:
                if queues[ci]:
                    var = queues[ci].pop(0)
                    all_indices.append(ci * self.m + var)

        # 3) Pad to multiple of num_replicas
        #    so that we can evenly slice
        pad_len = (self.num_replicas - len(all_indices) % self.num_replicas) % self.num_replicas
        if pad_len:
            # simplest: repeat from the start
            all_indices += all_indices[:pad_len]

        # 4) Slice out this rank’s samples
        rank_indices = all_indices[self.rank : len(all_indices) : self.num_replicas]

        # 5) Chunk into batches
        #    optionally pad the last batch if drop_last=False
        batches = []
        for i in range(0, len(rank_indices), self.batch_size):
            batch = rank_indices[i : i + self.batch_size]
            if len(batch) == self.batch_size:
                batches.append(batch)
            elif not self.drop_last:
                # pad up to batch_size
                pad = self.batch_size - len(batch)
                batch += rank_indices[:pad]
                batches.append(batch)

        self.epoch += 1

        for batch in batches:
            yield batch

    def __len__(self):
        # how many batches this rank will see
        # total_rank_samples = ceil(self.total_size / num_replicas)
        total_rank = math.ceil(self.total_size / self.num_replicas)
        if self.drop_last:
            return total_rank // self.batch_size
        else:
            return math.ceil(total_rank / self.batch_size)

    def set_epoch(self, epoch: int) -> None:
        """Call at start of each epoch to reshuffle differently."""
        self.epoch = epoch

class DistillationCsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, is_train, base_folder, sep="\t", tokenizer=None):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        logging.debug('Done loading data.')

        self.tokenize = tokenizer

        self.is_train = is_train
        if is_train:
            self.dist_image_features = torch.load(os.path.join(base_folder, "image_features.pt"))
            self.dist_text_features = torch.load(os.path.join(base_folder, "text_features.pt"))
            logging.debug(f'Loaded distillation features, {self.dist_image_features.shape}, {self.dist_text_features.shape}') 
            self.base_dir = os.path.join(base_folder, "images")
        else:
            self.dist_image_features = [None] * len(self.images)
            self.dist_text_features = [None] * len(self.images)
            self.base_dir = os.path.join(base_folder, "val_images")           

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(os.path.join(self.base_dir, str(self.images[idx]))))
        texts = self.tokenize([str(self.captions[idx])])[0]
        if self.is_train:
            dist_image_features = self.dist_image_features[idx]
            dist_text_features = self.dist_text_features[idx]
            return images, texts, dist_image_features, dist_text_features
        return images, texts

class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def expand_urls(urls, weights=None):
    if weights is None:
        expanded_urls = wds.shardlists.expand_urls(urls)
        return expanded_urls, None
    if isinstance(urls, str):
        urllist = urls.split("::")
        weights = weights.split('::')
        assert len(weights) == len(urllist),\
            f"Expected the number of data components ({len(urllist)}) and weights({len(weights)}) to match."
        weights = [float(weight) for weight in weights]
        all_urls, all_weights = [], []
        for url, weight in zip(urllist, weights):
            expanded_url = list(braceexpand.braceexpand(url))
            expanded_weights = [weight for _ in expanded_url]
            all_urls.extend(expanded_url)
            all_weights.extend(expanded_weights)
        return all_urls, all_weights
    else:
        all_urls = list(urls)
        return all_urls, weights


def get_dataset_size(shards):
    shards_list, _ = expand_urls(shards)
    dir_path = os.path.dirname(shards_list[0])
    sizes_filename = os.path.join(dir_path, 'sizes.json')
    len_filename = os.path.join(dir_path, '__len__')
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, 'r'))
        total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list])
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, 'r').read())
    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # CC3M (train): 2905954
        # CC12M: 10968539
        # LAION-400M: 407332084
        # LAION-2B (english): 2170337258
    num_shards = len(shards_list)
    return total_size, num_shards


def get_imagenet(args, preprocess_fns, split):
    assert split in ["train", "val", "v2"]
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns

    if split == "v2":
        from imagenetv2_pytorch import ImageNetV2Dataset
        dataset = ImageNetV2Dataset(location=args.imagenet_v2, transform=preprocess_val)
    else:
        if is_train:
            data_path = args.imagenet_train
            preprocess_fn = preprocess_train
        else:
            data_path = args.imagenet_val
            preprocess_fn = preprocess_val
        assert data_path

        dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)

    if is_train:
        idxs = np.zeros(len(dataset.targets))
        target_array = np.array(dataset.targets)
        k = 50
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:k] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype('int')
        sampler = SubsetRandomSampler(np.where(idxs)[0])
    else:
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
    )

    return DataInfo(dataloader=dataloader, sampler=sampler)


def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches


def filter_no_caption_or_no_image(sample):
    has_caption = ('txt' in sample)
    has_image = ('png' in sample or 'jpg' in sample or 'jpeg' in sample or 'webp' in sample)
    return has_caption and has_image


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


class detshuffle2(wds.PipelineStage):
    def __init__(
            self,
            bufsize=1000,
            initial=100,
            seed=0,
            epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
            seed = pytorch_worker_seed(epoch)
        else:
            # This seed to be deterministic AND the same across all nodes/workers in each epoch
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        weights=None,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
        epoch=-1,
    ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls, weights = expand_urls(urls, weights)
        self.urls = urls
        self.weights = weights
        if self.weights is not None:
            assert len(self.urls) == len(self.weights),\
                f"Number of urls {len(self.urls)} and weights {len(self.weights)} should match."
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        if self.deterministic:
            # reset seed w/ epoch if deterministic
            if self.worker_seed is None:
                # pytorch worker seed should be deterministic due to being init by arg.seed + rank + worker id
                seed = pytorch_worker_seed(epoch)
            else:
                seed = self.worker_seed() + epoch
            self.rng.seed(seed)
        for _ in range(self.nshards):
            if self.weights is None:
                yield dict(url=self.rng.choice(self.urls))
            else:
                yield dict(url=self.rng.choices(self.urls, weights=self.weights, k=1)[0])


def get_wds_dataset(args, preprocess_img, is_train, epoch=0, floor=False, tokenizer=None):
    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None
    resampled = getattr(args, 'dataset_resampled', False) and is_train

    num_shards = None
    if is_train:
        if args.train_num_samples is not None:
            num_samples = args.train_num_samples
        else:
            num_samples, num_shards = get_dataset_size(input_shards)
            if not num_samples:
                raise RuntimeError(
                    'Currently, the number of dataset samples must be specified for the training dataset. '
                    'Please specify it via `--train-num-samples` if no dataset length info is present.')
    else:
        # Eval will just exhaust the iterator if the size is not specified.
        num_samples = args.val_num_samples or 0 

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc

    if is_train and args.train_data_upsampling_factors is not None:
        assert resampled, "--train_data_upsampling_factors is only supported when sampling with replacement (with --dataset-resampled)."
    
    if resampled:
        pipeline = [ResampledShards2(
            input_shards,
            weights=args.train_data_upsampling_factors,
            deterministic=True,
            epoch=shared_epoch,
        )]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    # at this point we have an iterator over all the shards
    if is_train:
        if not resampled:
            pipeline.extend([
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ])
        pipeline.extend([
            # at this point, we have an iterator over the shards assigned to each worker at each node
            tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ])
    pipeline.extend([
        wds.select(filter_no_caption_or_no_image),
        wds.decode("pilrgb", handler=log_and_continue),
        wds.rename(image="jpg;png;jpeg;webp", text="txt"),
        wds.map_dict(image=preprocess_img, text=lambda text: tokenizer(text)[0]),
        wds.to_tuple("image", "text"),
        wds.batched(args.batch_size, partial=not is_train)
    ])

    dataset = wds.DataPipeline(*pipeline)

    if is_train:
        if not resampled:
            num_shards = num_shards or len(expand_urls(input_shards)[0])
            assert num_shards >= args.workers * args.world_size, 'number of shards must be >= total workers'
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )

    # FIXME not clear which approach is better, with_epoch before vs after dataloader?
    # hoping to resolve via https://github.com/webdataset/webdataset/issues/169
    # if is_train:
    #     # roll over and repeat a few samples to get same number of full batches on each node
    #     global_batch_size = args.batch_size * args.world_size
    #     num_batches = math.ceil(num_samples / global_batch_size)
    #     num_workers = max(1, args.workers)
    #     num_batches = math.ceil(num_batches / num_workers) * num_workers
    #     num_samples = num_batches * global_batch_size
    #     dataloader = dataloader.with_epoch(num_batches)
    # else:
    #     # last batches are partial, eval is done on single (master) node
    #     num_batches = math.ceil(num_samples / args.batch_size)

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def get_csv_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    input_filename = args.train_data if is_train else args.val_data
    base_folder = args.base_folder if is_train else args.val_base_folder
    assert input_filename
    dataset = CsvDataset(
        input_filename,
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        is_train=is_train,
        base_folder=base_folder,
        sep=args.csv_separator,
        tokenizer=tokenizer
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

def get_multi_clip_csv_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    assert is_train, "Multi-CLIP dataset is only supported for training. The validation should be done with a regular CSV dataset."
    input_filename = args.train_data
    base_folder = args.base_folder
    subfolders = args.image_subfolders

    dataset = MultiCLIPCsvDataset(
        input_filename=input_filename,
        transforms=preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        is_train=is_train,
        base_folder=base_folder,
        subfolders=subfolders,
        sep=args.csv_separator,
        tokenizer=tokenizer
    )

    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,  # n = number of captions per batch
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
        collate_fn=multi_positive_collate_fn  # <- custom collate function
    )

    dataloader.num_samples = num_samples * len(subfolders)  # actual #samples per epoch
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

def get_multi_image_csv_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    assert is_train, "Multi-image dataset is only supported for training. The validation should be done with a regular CSV dataset."
    input_filename = args.train_data
    base_folder = args.base_folder
    image_subfolders = args.image_subfolders
    assert input_filename
    dataset = MultiImageCsvDataset(
        input_filename,
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        is_train=is_train,
        base_folder=base_folder,
        subfolders=image_subfolders,  # list of subfolders with image variations
        sep=args.csv_separator,
        tokenizer=tokenizer
    )
    num_samples = len(dataset)
    sampler = MultiImageDistributedBatchSampler(
        dataset,
        batch_size=args.batch_size,
        num_replicas=torch.distributed.get_world_size(),
        rank=torch.distributed.get_rank(),
        shuffle=is_train,
        drop_last=is_train,
        seed=args.seed
    ) if args.distributed and is_train else None
    shuffle = is_train and sampler is None
    dataloader = DataLoader(
        dataset,
        num_workers=args.workers,
        pin_memory=True,
        batch_sampler=sampler
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

def get_distilled_csv_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = DistillationCsvDataset(
        input_filename,
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        is_train=is_train,
        base_folder=args.base_folder,
        sep=args.csv_separator,
        tokenizer=tokenizer
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


class SyntheticDataset(Dataset):

    def __init__(
            self,
            transform=None,
            image_size=(224, 224),
            caption="Dummy caption",
            dataset_size=100,
            tokenizer=None,
    ):
        self.transform = transform
        self.image_size = image_size
        self.caption = caption
        self.image = Image.new('RGB', image_size)
        self.dataset_size = dataset_size

        self.preprocess_txt = lambda text: tokenizer(text)[0]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if self.transform is not None:
            image = self.transform(self.image)
        return image, self.preprocess_txt(self.caption)


def get_synthetic_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    image_size = preprocess_fn.transforms[0].size
    dataset = SyntheticDataset(
        transform=preprocess_fn, image_size=image_size, dataset_size=args.train_num_samples, tokenizer=tokenizer)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_dataset_fn(data_path, dataset_type):
    if dataset_type == "distillation-csv":
        return get_distilled_csv_dataset
    elif dataset_type == "webdataset":
        return get_wds_dataset
    elif dataset_type == "csv":
        return get_csv_dataset
    elif dataset_type == "synthetic":
        return get_synthetic_dataset
    elif dataset_type == "multi-image-csv":
        if "val" in data_path:
            return get_csv_dataset
        return get_multi_image_csv_dataset
    elif dataset_type == "multi-clip-csv":
        if "val" in data_path:
            return get_csv_dataset
        return get_multi_clip_csv_dataset
    elif dataset_type == "auto":
        ext = data_path.split('.')[-1]
        if ext in ['csv', 'tsv']:
            return get_csv_dataset
        elif ext in ['tar']:
            return get_wds_dataset
        else:
            raise ValueError(
                f"Tried to figure out dataset type, but failed for extension {ext}.")
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    

def get_data(args, preprocess_fns, epoch=0, tokenizer=None):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.train_data or args.dataset_type == "synthetic":
        data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
            args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer)

    if args.val_data:
        data["val"] = get_dataset_fn(args.val_data, args.dataset_type)(
            args, preprocess_val, is_train=False, tokenizer=tokenizer)

    if args.imagenet_val is not None:
        data["imagenet-val"] = get_imagenet(args, preprocess_fns, "val")

    if args.imagenet_v2 is not None:
        data["imagenet-v2"] = get_imagenet(args, preprocess_fns, "v2")

    return data

# ---- TESTING GPT generated ----
'''def create_mock_dataset(n_captions=100, m_variations=4, image_size=(32, 32)):
    tmp_dir = tempfile.mkdtemp()
    image_names = [f"img_{i}.jpg" for i in range(n_captions)]
    captions = [f"caption {i}" for i in range(n_captions)]
    subfolders = [f"var_{j}" for j in range(m_variations)]

    # Create subfolders and dummy images
    for sub in subfolders:
        os.makedirs(os.path.join(tmp_dir, sub))
        for name in image_names:
            img_path = os.path.join(tmp_dir, sub, name)
            Image.new("RGB", image_size).save(img_path)

    # Create CSV
    df = pd.DataFrame({
        "image": image_names,
        "caption": captions
    })
    csv_path = os.path.join(tmp_dir, "mock.csv")
    df.to_csv(csv_path, sep="\t", index=False)

    return tmp_dir, csv_path, subfolders, image_names, captions

def dummy_tokenizer(texts):
    return [torch.tensor([ord(c) for c in t[:5]]) for t in texts]

def test_multi_image_csv_dataset():
    base_dir, csv_path, subfolders, image_names, captions = create_mock_dataset(n_captions=4, m_variations=3)

    transform = transforms.ToTensor()
    dataset = MultiImageCsvDataset(
        input_filename=csv_path,
        transforms=transform,
        img_key="image",
        caption_key="caption",
        is_train=True,
        base_folder=base_dir,
        subfolders=subfolders,
        sep="\t",
        tokenizer=dummy_tokenizer
    )

    # 1. Dataset length should be n * m
    assert len(dataset) == len(captions) * len(subfolders)

    # 2. Each caption should appear m times
    from collections import Counter
    caption_ids = [dataset.get_caption_id(i) for i in range(len(dataset))]
    counts = Counter(caption_ids)
    assert all(v == len(subfolders) for v in counts.values())

    # 3. All image paths must be valid
    for path in dataset.images:
        assert os.path.isfile(path), f"Missing image: {path}"

    # 4. Check __getitem__
    img, tokenized_caption = dataset[0]
    assert isinstance(img, torch.Tensor) and img.ndim == 3
    assert isinstance(tokenized_caption, torch.Tensor)

    shutil.rmtree(base_dir)
    print("✅ test_multi_image_csv_dataset passed.")

def test_multi_image_batch_sampler():
    base_dir, csv_path, subfolders, image_names, captions = create_mock_dataset(n_captions=5, m_variations=2)

    transform = transforms.ToTensor()
    dataset = MultiImageCsvDataset(
        input_filename=csv_path,
        transforms=transform,
        img_key="image",
        caption_key="caption",
        is_train=True,
        base_folder=base_dir,
        subfolders=subfolders,
        sep="\t",
        tokenizer=dummy_tokenizer
    )

    batch_size = 2
    sampler = MultiImageDistributedBatchSampler(
        dataset=dataset,
        batch_size=batch_size,
        num_replicas=1,
        rank=0,
        shuffle=False,
        drop_last=False
    )

    all_indices = []
    for batch in sampler:
        # 5. Ensure unique caption IDs within each batch
        caption_ids = [dataset.get_caption_id(i) for i in batch]
        assert len(set(caption_ids)) == len(caption_ids), f"Duplicate captions in batch: {caption_ids}"
        all_indices.extend(batch)

    # 6. Ensure all indices were covered
    assert sorted(all_indices) == sorted(list(range(len(dataset))))

    print("✅ test_multi_image_batch_sampler passed.")
    shutil.rmtree(base_dir)

if __name__ == "__main__":
    import os
    import tempfile
    import shutil
    import pandas as pd
    import torch
    from PIL import Image
    from torchvision import transforms
    from torch.utils.data import DataLoader
    from types import SimpleNamespace
    test_multi_image_csv_dataset()
    test_multi_image_batch_sampler()'''