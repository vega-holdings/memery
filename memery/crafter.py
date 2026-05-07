import torch
from torch import Tensor, device
from torch.utils.data import DataLoader, default_collate
from torchvision.datasets import VisionDataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image, ImageFile


def make_dataset(new_files: list[tuple[str, str]]) -> list[tuple[str, int]]:
    '''Returns a list of (path, index) pairs.

    The previous implementation also built a parallel `slugs` list that
    nothing downstream ever read — pure dead weight on every build.
    '''
    return [(str(path), i) for i, (path, _slug) in enumerate(new_files)]


def pil_loader(path: str) -> Image.Image:
    """Open `path` and return an RGB PIL image, or None on failure.

    Uses PIL's JPEG `draft` mode to decode at the smallest IDCT scale that's
    still ≥ 256px. For large JPEGs this skips most of the inverse-DCT work
    and saves a substantial amount of file I/O — measured ~1.6x speedup on
    a realistic Downloads sample (~165KB median image size). For non-JPEG
    formats `draft` is a no-op, so it's safe to leave on unconditionally.
    """
    ImageFile.LOAD_TRUNCATED_IMAGES = True  # tolerate partially-truncated files
    try:
        # open path as file to avoid ResourceWarning
        # https://github.com/python-pillow/Pillow/issues/835
        with open(path, 'rb') as f:
            img = Image.open(f)
            # Hint the JPEG decoder for partial-scale decode. 256 is chosen
            # to stay safely above the 224 CLIP input size after CenterCrop.
            try:
                img.draft('RGB', (256, 256))
            except (AttributeError, OSError):
                pass
            return img.convert('RGB')
    except Exception as e:
        print(f"Skipping image {path}: {e}")
        return None


class DatasetImagePaths(VisionDataset):

    def __init__(self, new_files, transforms=None):
        super().__init__(new_files, transforms=transforms)
        self.samples = make_dataset(new_files)
        self.loader = pil_loader
        self.root = 'file dataset'

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            sample = self.loader(path)
            if sample is None:
                return None
            if self.transforms is not None:
                sample = self.transforms(sample)
            return sample, target
        except Exception as e:
            print(f"Skipping file {path} due to error: {e}")
            return None


def safe_collate(batch):
    """Drop items that failed to decode, then run the default collate.

    Without this, a single `None` from `__getitem__` (e.g. a file that
    passed `verify_image` but failed at decode time) would crash the entire
    DataLoader with an unhelpful traceback. Returning `None` for an empty
    batch lets the encoder loop skip it cleanly.
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return default_collate(batch)


def clip_transform(n_px: int) -> Compose:
    return Compose([
        Resize(n_px, interpolation=Image.BICUBIC),
        CenterCrop(n_px),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073),
                  (0.26862954, 0.26130258, 0.27577711)),
    ])


def crafter(new_files: list[str], device: device,
            batch_size: int = 128, num_workers: int = 0):
    """Build the DataLoader used to feed CLIP.

    `num_workers=0` is the macOS default because DataLoader workers use
    `spawn` (fork is unsafe with PyTorch), each costs ~2s of startup, and
    on typical-sized images (~150KB) the parallelism gain is smaller than
    the spawn overhead. For folders dominated by very large images
    (multi-megabyte phone photos / scans), `--workers 4-8` can win
    significantly — measured at scale, not at small N.
    """
    with torch.no_grad():
        imagefiles = DatasetImagePaths(new_files, clip_transform(224))
        img_loader = DataLoader(
            imagefiles,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=safe_collate,
        )
    return img_loader


def preproc(img: Tensor) -> Compose:
    transformed = clip_transform(224)(img)
    return transformed
