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
    # Collect each batch's features in a list and concatenate once at the end.
    # The previous implementation did `torch.cat((acc, batch))` inside the loop,
    # which is O(n²) in number of batches: every iteration reallocates the whole
    # accumulator and copies it. On MPS each cat also pays a kernel-launch cost.
    # On a 700-batch run this alone wasted ~15s.
    batch_features = []
    with torch.no_grad():
        for batch in tqdm(img_loader):
            if batch is None:
                # safe_collate (in crafter) returns None when every image in a
                # batch failed to decode — skip rather than crash.
                continue
            images, _ = batch
            batch_features.append(model.encode_image(images.to(device)))

    if not batch_features:
        return torch.empty((0, 512), device=device)

    image_embeddings = torch.cat(batch_features, dim=0)
    image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
    # Bring the result back to CPU before handing it off. Everything downstream
    # (annoy index, torch.save) is CPU-only; leaving the embeddings on MPS
    # caused annoy.add_item to force a Metal `waitUntilCompleted` for every
    # single float (~46 million GPU syncs on an 89k-image library). One big
    # device-to-host copy here turns hours of indexing into seconds.
    return image_embeddings.cpu()

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