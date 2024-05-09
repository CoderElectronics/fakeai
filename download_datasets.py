from progress.bar import *
import urllib.request
from pathlib import Path

pbar = None
def show_progress(block_num, block_size, total_size, msg="Downloading..."):
    global pbar
    if pbar is None:
        pbar = PixelBar(msg, max=total_size)

    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.goto(downloaded)
    else:
        pbar.finish()
        pbar = None

# Data set files
Path("data/orig").mkdir(parents=True, exist_ok=True)

urllib.request.urlretrieve("https://zenodo.org/record/4561253/files/WELFake_Dataset.csv?download=1", "data/orig/WELFake_Dataset.csv", lambda *x: show_progress(*x, msg="Downloading WELFake..."))
urllib.request.urlretrieve("https://dataset-dl-fakeai.surge.sh/FakeNewsNet.csv", "data/orig/FakeNewsNet.csv", lambda *x: show_progress(*x, msg="Downloading FakeNewsNet..."))

urllib.request.urlretrieve("https://fake-and-real-news-dataset.surge.sh/True.csv", "data/orig/True.csv", lambda *x: show_progress(*x, msg="Downloading True-or-Fake-dataset (true)..."))
urllib.request.urlretrieve("https://fake-and-real-news-dataset.surge.sh/Fake.csv", "data/orig/Fake.csv", lambda *x: show_progress(*x, msg="Downloading True-or-Fake-dataset (fake)..."))
