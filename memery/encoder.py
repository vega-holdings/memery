import torch
import clip
from clip.model import CLIP
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import Tensor, device
from torchvision.transforms import Compose

def load_model(device: device) -> CLIP:
    model, _ = clip.load("ViT-B/32", device, jit=False)
    model = model.float()
    # Inference-only — disables any train-time behavior in submodules.
    # CLIP ViT-B/32 has no batchnorm or dropout that fires here, so this is
    # mostly defensive correctness, but it's free.
    model.eval()
    return model

def image_encoder(img_loader: DataLoader, device: device, model: CLIP):
    """Encode a DataLoader's worth of images to L2-normalized CLIP features.

    Returns ``(embeddings, labels)``:
      * ``embeddings``: ``(K, 512)`` CPU tensor of unit-norm features, where
        ``K`` is the number of images that *successfully* survived decoding
        and collation. May be smaller than ``len(dataset)`` if any items
        were dropped by ``crafter.safe_collate``.
      * ``labels``: ``(K,)`` CPU long tensor mapping each row of
        ``embeddings`` back to its original index in ``new_files``. The
        caller uses this to keep file paths and embeddings aligned even
        when some files failed to decode mid-batch.

    Implementation notes:
      * Features are accumulated in a Python list and concatenated once at
        the end. The previous in-loop ``torch.cat`` was O(n²) in the number
        of batches and on MPS paid a Metal kernel-launch cost per concat —
        ~15s wasted on a 702-batch run.
      * The final tensor is moved to CPU here. Everything downstream (annoy,
        torch.save) is CPU-only; leaving embeddings on MPS caused
        ``annoy.add_item`` to force a Metal ``waitUntilCompleted`` for every
        single float, i.e. ~46M GPU syncs on an 89k-image library.
    """
    feature_chunks = []
    label_chunks = []
    with torch.no_grad():
        for batch in tqdm(img_loader):
            if batch is None:
                # safe_collate returns None when every image in a batch
                # failed to decode — skip rather than crash.
                continue
            images, labels = batch
            feature_chunks.append(model.encode_image(images.to(device)))
            label_chunks.append(labels)

    if not feature_chunks:
        return (
            torch.empty((0, 512)),
            torch.empty((0,), dtype=torch.long),
        )

    image_embeddings = torch.cat(feature_chunks, dim=0)
    image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
    surviving_labels = torch.cat(label_chunks, dim=0)
    return image_embeddings.cpu(), surviving_labels.cpu()

def text_encoder(text: str, device: device, model: CLIP):
    with torch.no_grad():
        text = clip.tokenize(text).to(device)
        text_features = model.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return(text_features)

def image_query_encoder(image: Tensor, device: device, model: CLIP):
    with torch.no_grad():
        image_embed = model.encode_image(image.unsqueeze(0).to(device))
    image_embed = image_embed / image_embed.norm(dim=-1, keepdim=True)
    return(image_embed)