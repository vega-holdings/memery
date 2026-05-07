import sys
from pathlib import Path
from typing import Optional

import typer

import memery
from memery.core import Memery
# Sometimes you just want to be able to pipe information through the terminal. This is that command

app = typer.Typer()

def main():
    app()

@app.command()
def recall(
    root: str = typer.Argument('.', help="Image folder to search"),
    text: str = typer.Option(None, "-t", "--text", help="Text query"),
    negative: str = typer.Option(None, "-nt", "--negative-text", help="Negative text query"),
    image: str = typer.Option(None, "-i", "--image", help="Filepath to image query"),
    number: int = typer.Option(10, "-n", "--number", help="Number of results to return"),
    ) -> None:
    """Search recursively over a folder from the command line."""
    memery = Memery()
    ranked = memery.query_flow(root, query=text, negative_query=negative, image_query=image)
    print(ranked[:number])

@app.command()
def serve(root: Optional[str] = typer.Argument(None)):
    """Runs the streamlit GUI in your browser"""
    # Importing here so `memery --help` doesn't pay the streamlit import cost
    from streamlit.web import cli as stcli

    app_path = str(Path(memery.__file__).parent / "streamlit_app.py")
    target_root = root if root is not None else "./images"
    sys.argv = ["streamlit", "run", app_path, "--", target_root]
    sys.exit(stcli.main())

@app.command()
def build(
    root: str = typer.Argument('.'),
    workers: int = typer.Option(
        2,
        "--workers", "-w",
        help=(
            "DataLoader workers for image preprocessing. Measured on macOS/MPS: "
            "2 wins ~15% on realistic corpora (4000+ images). Higher counts hurt "
            "because the GPU is a serial bottleneck and IPC overhead piles up. "
            "Pass 0 to disable workers entirely (slightly faster on tiny folders, "
            "or as a fallback if multiprocessing misbehaves in your environment)."
        ),
    ),
    ):
    '''
    Indexes the directory and all subdirectories
    '''
    memery = Memery()
    memery.index_flow(root, num_workers=workers)
    return None

@app.command()
def purge(root: str = typer.Argument('.')):
    """
    Cleans out all files saved by memery
    """
    memery = Memery()
    memery.clean(root)
    print("Purged files!")

if __name__ == "__main__":
    main()