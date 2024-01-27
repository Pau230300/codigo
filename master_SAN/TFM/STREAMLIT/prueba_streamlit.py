import streamlit as st

# Función de inicialización que se ejecutará solo una vez al inicio
@st.cache_resource
def start():
  RAW_DATA_PATH = 'drive/MyDrive/TFM/raw_data'

  import chromadb
  from chromadb.utils import embedding_functions
  import os
  import getpass
  import openai
  from llama_index.embeddings import HuggingFaceEmbedding

  # Creamos una instancia del cliente 'chromadb'
  chroma_client = chromadb.Client()

  print("Importamos sentence_transformer_ef")

  # Creamos una función de embedding usando Sentence Transformer
  sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
      model_name="paraphrase-multilingual-mpnet-base-v2"
  )

  print("Generamos la base de datos de chroma")

  # Creamos una colección en 'chromadb' con la función de embedding definida
  # Especificamos el nombre de la colección y usamos un espacio de similitud coseno
  db_chroma = chroma_client.create_collection(
      name="ChatBot_DDBB_prueba_cos",
      embedding_function=sentence_transformer_ef,
      metadata={"hnsw:space": "cosine"}
  )

  print("Creamos el HuggingFaceEmbedding del modelo paraphrase-multilingual-mpnet-base-v2")

  # Creamos un modelo de embedding usando HuggingFace
  embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
  # Instanciamos un modelo de embedding de HuggingFace, especificando el modelo de Sentence Transformer

  from cleantext import clean
  import re

  print("Importamos dependencias de llama index")

  from llama_index.schema import TransformComponent

  # no gpl pq nos mola el mercado privado
  class TextCleaner(TransformComponent):
    def __call__(self, nodes, **kwargs):
      for node in nodes:
        node.text = clean(node.text,
                          fix_unicode=True, to_ascii=True, lower=True,
                          no_line_breaks=True, no_urls=False, no_emails=False,
                          no_phone_numbers=False, no_numbers=False, no_digits=False,
                          no_currency_symbols=False, no_punct=False, replace_with_punct="")
        node.text = re.sub(r'\s+', ' ', node.text).strip()
      return nodes

  from pathlib import Path
  from llama_index import SimpleDirectoryReader, ServiceContext
  from llama_index.vector_stores import ChromaVectorStore
  from llama_index.storage.storage_context import StorageContext
  from llama_index.text_splitter import TokenTextSplitter
  from llama_index.ingestion import IngestionPipeline
  from llama_index import download_loader
  from llama_index import SimpleDirectoryReader
  # Importamos los loaders
  from llama_hub.file.unstructured import UnstructuredReader
  from llama_hub.file.pandas_excel import PandasExcelReader
  from llama_hub.file.markdown import MarkdownReader
  from llama_hub.file.pdf import PDFReader

  #De forma un poco aleatoria decide que le vale from llama_hub.file.unstructured import UnstructuredReader o no
  from llama_index import download_loader, SimpleDirectoryReader
  try:
    UnstructuredReader = download_loader('UnstructuredReader')
  except:
    pass

  # Cargamos el LLM para incluir en el pipe de llama index

  import torch
  from transformers import BitsAndBytesConfig

  print("Creamos configuracion de cuantización del modelo")

  quantization_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_compute_dtype=torch.float16,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_use_double_quant=True,
  )

  print("Cargamos el modelo Mistral-7B-Instruct-v0.1")

  model_id = "mistralai/Mistral-7B-Instruct-v0.1"
  from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
  model_4bit = AutoModelForCausalLM.from_pretrained( model_id, device_map="auto",quantization_config=quantization_config,)
  tokenizer = AutoTokenizer.from_pretrained(model_id)

  pipeline = pipeline(
          "text-generation",
          model=model_4bit,
          tokenizer=tokenizer,
          use_cache=True,
          device_map="auto",
          max_length=5000,
          do_sample=True,
          top_k=5,
          num_return_sequences=1,
          eos_token_id=tokenizer.eos_token_id,
          pad_token_id=tokenizer.eos_token_id,
  )

  print("Creamos la pipeline de HuggingFace")

  from langchain import HuggingFacePipeline
  from langchain import PromptTemplate, LLMChain
  llm = HuggingFacePipeline(pipeline=pipeline)

  # TODO: Pieza que quite aquellos documentos con una extension no esperada
  # si veis alguna mirad si hay un loader para esa extension

  print("Creamos los dir_reader")

  # Se cargan todos los documentos de la carpeta seleccionada
  dir_reader = SimpleDirectoryReader(RAW_DATA_PATH, file_extractor={ #! TODO controlar que hay en la carpeta y si es necesario limpiar los archivos que no es capaz de leer
    ".pdf": PDFReader(),
    ".html": UnstructuredReader(),
    ".eml": UnstructuredReader(),
    ".docx": UnstructuredReader(),
    ".pptx": UnstructuredReader(),
    ".jpg": UnstructuredReader(),
    ".png": UnstructuredReader(),
    ".xlsx": PandasExcelReader(),
    ".md": MarkdownReader(),
  })
  documents = dir_reader.load_data()

  print("Creamos el TokenTextSplitter")

  # Creamos un splitter de texto que divide los textos en fragmentos de 256 tokens
  text_splitter = TokenTextSplitter(chunk_size=256)                                                                                              # tokens? palabra? unidades???

  print("Creamos la vectore_store")

  # Inicializamos el almacenamiento de vectores con la colección de ChromaDB
  vector_store = ChromaVectorStore(chroma_collection=db_chroma)                                                                                  # bbdd¿ que es esto exactamente???

  print("Creamos el storage_context")

  # Creamos un contexto de almacenamiento con los valores predeterminados y el almacenamiento de vectores
  storage_context = StorageContext.from_defaults(vector_store=vector_store) #db structure, represents and manages the state and configuration of your storage

  print("Creamos el service_context")

  # Creamos un contexto de servicio con los valores predeterminados y el modelo de embedding
  service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=llm) #configuration??

  # Creación de una instancia de la transformación personalizada
  custom_clean_transform = TextCleaner()

  print("Creamos el pipeline de ingesta")

  # Construimos un pipeline de ingesta con las transformaciones especificadas
  # (limpieza, división de texto y embedding) y el almacenamiento de vectores
  pipeline = IngestionPipeline(
      transformations=[custom_clean_transform, text_splitter, embed_model],
      vector_store=vector_store
  )
  nodes = pipeline.run(documents=documents)

  from llama_index import (
      VectorStoreIndex,
      get_response_synthesizer,
  )
  from llama_index.retrievers import VectorIndexRetriever
  from llama_index.query_engine import RetrieverQueryEngine
  from llama_index.postprocessor import SimilarityPostprocessor
  from typing import List, Tuple, Any, Type, Optional
  from collections import namedtuple
  from llama_index.schema import NodeWithScore
  from llama_index import QueryBundle
  from llama_index.schema import NodeWithScore
  from llama_index.postprocessor.types import BaseNodePostprocessor
  from pydantic import BaseModel, Field
  # Construimos Index (para hacer queries) sobre los nodos de nuestra pipeline

  QueryResponse = namedtuple('query_response', ['text', 'score', 'metadata'])

  print("Creamos el VectorStoreIndex")

  index = VectorStoreIndex(nodes, service_context=service_context)
  # Retriever: https://docs.llamaindex.ai/en/stable/api_reference/query/query_engines/retriever_query_engine.html
  # Grafo: https://docs.llamaindex.ai/en/stable/api_reference/query/query_engines/knowledge_graph_query_engine.html
  # hay muchisimas cosas que se pueden hacer, para esto vamos a montarlo basico

  # postprocesador personalizado, simplemente elige aquellos que superan el umbral de distancia que le marcamos
  # Esto existia para (<) he hecho uno customp para nuestro caso (>), se le puede anadir lo que querais
  class Umbral_coseno(BaseNodePostprocessor):

      threshold: float = Field(default=None, description="Umbral a partir del cual no devuelve el resultado")

      @classmethod
      def class_name(cls) -> str:
          return "Umbral_coseno"

      def _postprocess_nodes(self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle] = None) -> List[NodeWithScore]:
          """Filtro en funcion de la distancia coseno"""
          sim_cutoff_exists = self.threshold is not None
          # caso nodos vacios
          if not nodes:
              filtradas = nodes
          # caso no hay corte
          if not sim_cutoff_exists:
            filtradas = nodes
          # caso hay corte y nodos no vacios
          caso_limite = nodes[0]
          filtradas = [node for node in nodes if node.score > self.threshold]
          if not filtradas:
            filtradas = [caso_limite]

          return filtradas

  # HACERLO EN LLAMA INDEX: Done
  # Función para obtener consulta en la base de datos: Done
  # Consulte el FAQ, y si recibe una pregunta que devuelva la respuesta y el metadato del fichero para mostrar en el streamlit : caso general
  # Si consulta un html que ponga los metadatos de un html: caso general
  # pdfs por ahora pasando: done

  # response_mode="no_text" es para que solo devuelva los docs, se puede integrar el query_engine con el LLM ¿queremos?
  def query(prompt: str, k: int=3, threshold: float=0.2) -> List[Tuple[str, float, any]]:
    # Devuelve los k vectores mas cercanos
    # el service_context es para pasarle nuestro modelo de embeddings
    retriever = VectorIndexRetriever(
        index=index,
        service_context=service_context,
        similarity_top_k=k,
    )
    # para info ver link de retrievers
    query_engine = RetrieverQueryEngine.from_args(
      retriever=retriever
      , response_synthesizer=None
      , service_context=service_context
      , node_postprocessors=[]
      , response_mode="no_text"
      , text_qa_templat=None
      , refine_template=None
      , summary_template=None
      , simple_template=None
      , output_cls=None
      , use_async=False
      , streaming=False
    )
    response = query_engine.query(prompt)
    # esto permite acceder a los datos como doc.text, doc.score, doc.metadata dentro de la tupa
    return [QueryResponse(node.text, node.score, node.metadata) for node in response.source_nodes]

  from typing import NewType

  Prompt = NewType('langchain.prompt', 'langchain_core.prompts.prompt.PromptTemplate')
  Llm = NewType('langchain.prompt', 'langchain_community.llms.huggingface_pipeline.HuggingFacePipeline')

  class LLM:

    prompt: Prompt = Field(default=None, description="Plantilla de prompt para el LLM")
    llm: Llm = Field(default=None, description="Pipeline del LLM que se ha cargado")
    llm_chain = Field(default=None, description="prompt+llm")

    def __init__(self, plantilla_prompt: Prompt, llm: Llm):
      self.llm_chain = LLMChain(prompt=plantilla_prompt, llm=llm)

    def llm_q(self, prompt: str) -> str:
      context_raw = query(prompt=prompt, k=3)
      # context_text = "\n".join([doc.text for doc in context_raw]) La que había anterior y funcionaba
      context_text = "\n".join([f'{idx}. {doc.text} (score: {doc.score})' for idx, doc in enumerate(context_raw, start=1)])
      response = self.llm_chain.run({"question": prompt, "context": context_text})
      return response

  def from_context_raw_to_json(context_raw_tuple: tuple) -> dict:
    return_dict = dict()

    for idx, context_raw_item in enumerate(context_raw_tuple, start=1):
      return_dict[f'response_{idx}'] = {
          'text': context_raw_item.text,
          'score': context_raw_item.score,
          'metadata': context_raw_item.metadata
      }

    return return_dict

  template = """<s>[INST] <<SYS>>
  Imagina un escenario en un banco donde un chatbot avanzado, como ChatGPT, está integrado en el sistema de atención al cliente del banco Santander
  para asistir con preguntas frecuentes (FAQs). Este chatbot está programado para comprender y responder consultas relacionadas con servicios bancarios
  como la apertura de cuentas, opciones de crédito, inversiones, transferencias internacionales y medidas de seguridad. El chatbot tiene la capacidad de
  ofrecer respuestas personalizadas y detalladas, guiando a los usuarios a través de procesos complejos y proporcionando consejos financieros útiles. Además,
  está equipado para aprender de las interacciones pasadas, mejorando constantemente su capacidad para asistir a los clientes de manera más eficiente.
  El chatbot está disponible las 24 horas del día, los 7 días de la semana, accesible a través del sitio web del banco, aplicaciones móviles y quioscos en las sucursales,
  asegurando que los clientes reciban asistencia inmediata en cualquier momento y lugar.

  Limítate a responder las preguntas de manera breve, concisa, educada y en español.

  Te vamos a mostrar las ocurrencias de mayor a menor similaridad, responde en base a aquellas que consideres la correcta.
  <</SYS>>

  {context}
  {question} [/INST]
  """
  prompt = PromptTemplate(template=template, input_variables=["question","context"])
  Mistral = LLM(prompt, llm)

  return Mistral, query

def from_context_raw_to_json(context_raw_tuple: tuple) -> dict:
  return_dict = dict()

  for idx, context_raw_item in enumerate(context_raw_tuple, start=1):
    return_dict[f'response_{idx}'] = {
        'text': context_raw_item.text,
        'score': context_raw_item.score,
        'metadata': context_raw_item.metadata
    }

  return return_dict

# Ejecuta la función de inicialización al inicio de la aplicación
Mistral, query = start()

# Crear un contenedor personalizado con fondo blanco
st.markdown(
    """
    <style>
    .stApp {
        background-color: rgb(254, 254, 254); /* Blanco como color de fondo */
    }

    /* Puedes agregar más personalizaciones de CSS aquí usando los colores extraídos */
    /* Color principal: rgb(234, 3, 4) - Rojo */
    /* Color secundario: rgb(240, 128, 129) - Rojo claro */

    /* Por ejemplo, para el título */
    h1 {
        color: rgb(234, 3, 4); /* Rojo para el título principal */
    }

    /* Para otros elementos, como botones o texto, puedes usar: */
    /* .stButton > button {
        background-color: rgb(240, 128, 129); /* Rojo claro para botones */
    /* }
    </style>
    """,
    unsafe_allow_html=True,
)

# Create centered main title with an image instead of emoji
st.markdown("""
    <h1 style="text-align: center;">
        <img src="https://1000logos.net/wp-content/uploads/2017/09/Santander-Logo.png" alt="Logo" style="height: 300px;"/>
    </h1>
    """, unsafe_allow_html=True)

# Create centered main title
st.title('Mistrander')
# Create a text input box for the user
prompt = st.text_input('Introduzca su pregunta aquí')

# If the user hits enter
if prompt:
  response = Mistral.llm_q(prompt)
  # ...and write it out to the screen
  st.write(response)

  # Display raw response object

  # Display source text
  with st.expander('Contexto obtenido'):
    rag_context_json = from_context_raw_to_json(query(prompt=prompt, k=3))
    st.write(rag_context_json)