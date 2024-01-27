### LEEMOS DATOS

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

#!pip install llama-index==0.9.22

##### CHROMA

import llama_index
from llama_index import VectorStoreIndex,SummaryIndex, download_loader, readers
from IPython.display import Markdown, display
import os

loader = llama_index.readers.SimpleWebPageReader(html_to_text=True)
documents = loader.load_data(urls=['https://google.com'])

documents[0]
SummaryIndex.from_documents(documents)

index = VectorStoreIndex.from_documents(documents)
index.query('What language is on this website?')


### EMBEDDING




### LLM