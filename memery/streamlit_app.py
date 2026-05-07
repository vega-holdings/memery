# Builtins
import argparse
import sys
from pathlib import Path

# Dependencies
import streamlit as st
from PIL import Image

# Local
from memery.core import Memery


# Parses the args from the command line
def parse_args(args: list[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument('root', nargs='?', default='./images',
                        help='starting directory to search')
    return parser.parse_args(args)

# Initialize session state
args = parse_args(sys.argv[1:])
if 'memery' not in st.session_state:
    st.session_state['memery'] = Memery()
memery: Memery = st.session_state['memery']

# Configs
st.set_page_config(page_title='Memery', layout="centered")

# Draw the sidebar
st.sidebar.title("Memery")

settings = st.sidebar.expander(label="Settings", expanded=False)
with settings:
    do_clear_cache = st.button(label="Clear Cache")
    num_workers = st.slider(label="Number of workers", max_value=8)

dir_l, dir_r = st.sidebar.columns([3, 1])
with dir_l:
    path = st.text_input(label='Directory', value=args.root)
with dir_r:
    st.title("")
    do_index = st.button(label="Index", key='do_index')

search_l, search_r = st.sidebar.columns([3, 1])
with search_l:
    text_query = st.text_input(label='Text query', value='')
    negative_text_query = st.text_input(label='Negative Text query', value='')
with search_r:
    st.title("")
    search_button = st.button(label="Search", key="search_button")


image_query = st.sidebar.file_uploader(label='Image query')
image_query_display = st.sidebar.container()
if image_query:  # Display the image query if there is one
    img = Image.open(image_query).convert('RGB')
    with image_query_display:
        st.image(img)
logbox = st.sidebar.empty()
skipped_files_box = st.sidebar.expander(label='Skipped files', expanded=False)

# Draw the main page
sizes = {'small': 115, 'medium': 230, 'large': 332, 'xlarge': 600}
l, m, r = st.columns([4, 1, 1])
with l:
    num_images = st.slider(label='Number of images', value=12)

with m:
    size_choice = st.selectbox(label='Image width', options=list(sizes.keys()), index=1)
with r:
    captions_on = st.checkbox(label="Caption filenames", value=False)
image_display_zone = st.container()


# Index the directory
def index(logbox, path, num_workers):
    if not Path(path).exists():
        logbox.warning(f'{path} does not exist!')
        return
    with logbox, st.spinner(f'Indexing {path}...'):
        memery.index_flow(path, num_workers)
    logbox.success('Done indexing')


# Clears out the database and treemap files
def clear_cache(root, logbox):
    memery.clean(root)
    logbox.info("Cleaned database and index files")


# Runs a search
def search(root, text_query, negative_text_query, image_query,
           image_display_zone, skipped_files_box,
           num_images, captions_on, sizes, size_choice):
    if not Path(root).exists():
        logbox.warning(f'{root} does not exist!')
        return
    with logbox, st.spinner('Searching...'):
        ranked = memery.query_flow(
            root, text_query, negative_text_query, image_query
        )
    if not ranked:
        logbox.info('No results.')
        return

    ims_to_display = {}
    size = sizes[size_choice]
    for o in ranked[:num_images]:
        name = o.replace(root, '')
        try:
            ims_to_display[name] = Image.open(o).convert('RGB')
        except Exception as e:
            with skipped_files_box:
                st.warning(f'Skipping bad file: {name}\ndue to {type(e)}')
    with image_display_zone:
        if captions_on:
            st.image(list(ims_to_display.values()),
                     width=size, channels='RGB',
                     caption=list(ims_to_display.keys()))
        else:
            st.image(list(ims_to_display.values()),
                     width=size, channels='RGB')
    logbox.success(f'Found {len(ranked)} matches.')


# Decide which actions to take
if do_clear_cache:
    clear_cache(path, logbox)
elif do_index:
    index(logbox, path, num_workers)
elif search_button or text_query or image_query:
    search(path, text_query, negative_text_query, image_query,
           image_display_zone, skipped_files_box,
           num_images, captions_on, sizes, size_choice)
