from annoy import AnnoyIndex
import numpy as np
import torch
from tqdm import tqdm


EMBED_DIM = 512


def join_all(db, new_files, new_embeddings) -> dict:
    start = len(db)
    for i, file in enumerate(new_files):
        path, hash = file
        index = i + start
        db[index] = {
            'hash': hash,
            'fpath': path,
            'embed': new_embeddings[i],
        }
    return db


def _to_cpu_array(emb):
    """Coerce one stored embedding to a fast plain-Python list of floats.

    Annoy iterates the vector with PyFloat_AsDouble. If `emb` is a torch
    Tensor still on a GPU device, that path fires `Tensor.item()` once per
    element, and on MPS each call costs a full `waitUntilCompleted`. So
    we make sure annoy gets a CPU-resident plain sequence.
    """
    if isinstance(emb, torch.Tensor):
        # `.detach().cpu()` is a single bulk device-to-host copy.
        return emb.detach().cpu().numpy()
    if isinstance(emb, np.ndarray):
        return emb
    return np.asarray(emb, dtype=np.float32)


def build_treemap(db) -> AnnoyIndex:
    """Build an angular Annoy index over the database's embeddings.

    The previous implementation was the dominant bottleneck on macOS/MPS at
    scale: handing GPU tensors to annoy caused element-wise GPU syncs, so a
    library of ~90k images spent more time in this single function than in
    the entire CLIP encoding pass. This version converts the embeddings to
    CPU in a single bulk step per item and shows progress.
    """
    treemap = AnnoyIndex(EMBED_DIM, 'angular')
    if not db:
        treemap.build(5)
        return treemap

    for k, v in tqdm(db.items(), desc="Indexing", total=len(db)):
        treemap.add_item(k, _to_cpu_array(v['embed']))

    treemap.build(5)
    return treemap


def save_archives(root, treemap, db) -> tuple[str, str]:
    dbpath = root / 'memery.pt'
    if dbpath.exists():
        dbpath.unlink()
    torch.save(db, dbpath)

    treepath = root / 'memery.ann'
    if treepath.exists():
        treepath.unlink()
    treemap.save(str(treepath))

    return str(dbpath), str(treepath)
