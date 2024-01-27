# Nos aseguramos de tener una GPU en la instancia gratuita de colab (T4)
!nvidia-smi

# Instalamos las dependencias que vamos a usar
!pip install transformers==4.31.0
!pip install sentence-transformers==2.2.2
!pip install pinecone-client==2.2.2
!pip install datasets==2.14.0
!pip install accelerate==0.21.0
!pip install einops==0.6.1
!pip install pydantic==2.5.3
!pip install langchain==0.0.240
!pip install xformers==0.0.20
!pip install bitsandbytes==0.41.0
!pip install llama-index==0.9.22
!pip install PyMuPDF==1.23.8
!pip install PyMuPDFb==1.23.7
!pip install openpyxl==3.0.10
!pip install torch==1.13.0
!pip install torchaudio==0.13.0
!pip install torchaudio==0.14.0
!pip install chromadb-client==0.4.19
!pip install langchain==0.0.352
!pip install sentence_transformers
# Install the transformers library
!pip install git+https://github.com/huggingface/transformers
!pip install chromadb==0.4.21


import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import torch
import langchain
from langchain.llms import HuggingFacePipeline
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import pandas as pd
from hashlib import md5
import unicodedata
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
import time




############
# FUNCIONS #
############


def generate_id(html_content):
      html_content = str(html_content)
      return md5(html_content.encode('utf-8')).hexdigest()

def quitar_tildes(texto):
      return ''.join((c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn'))


def mostrar_query(resultados: list) -> None:
  import textwrap
  LINE_WIDTH = 80
  for i, resultado in enumerate(resultados):
    wrapped_string = textwrap.wrap(resultado, width=LINE_WIDTH)
    formatted_string = '\n'.join(wrapped_string)
    print(f"Respuesta_{i}:\n")
    print(formatted_string)
    print('\n\n')



########################
# VECTORSTORE : CHROMADB
########################

def prepare_chromaDB():
   # Params BBDD
   id_sent_transf_embedd_model = "paraphrase-multilingual-mpnet-base-v2"
   nombre_bbdd = "ENCOMIENDA"
   
   # Modelo de Embedding (esto es con método de Chroma)
   embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
           model_name = id_sent_transf_embedd_model
   )
   
   # otra forma de definir el embedding atraves de langchain
   embed_model = HuggingFaceEmbeddings(
         model_name=id_sent_transf_embedd_model)
   
   # Generamos entorno de CHROMBA (como un esquema)
   chroma_client = chromadb.PersistentClient(path='./CHROMA')
   chroma_client.list_collections()
   
   # Generamos colección o bbdd concreta dentro del entorno anterior (como una tabla)
   # Como parámetro introducimos el nombre de la bbdd a consultar dentro del entorno
   vectorstore_chroma = chroma_client \
       .create_collection(
           name = nombre_bbdd,
           embedding_function = embedder
   )
       
   chroma_client.list_collections()
   
   # Esto permite tener un insight de bbdd creada
   vectorstore_chroma.peek()
   vectorstore_chroma.count()
   vectorstore_chroma.get()

   
   # Leer excel
   import pandas as pd
   ruta ='FAQS_Particulares.xlsx'
   df = pd.read_excel(ruta)
   
   
   
   questions = 'Pregunta'
   answers = 'Respuesta App'
   df = pd.read_excel(ruta).drop_duplicates().reset_index()[[questions, answers]]
   df[questions] = df[questions].fillna('NONE').apply(quitar_tildes).str.replace('\n', '').replace('\t', '').replace('\r', '')
   df[answers] = df[answers].fillna('NONE').apply(quitar_tildes).str.replace('\n', '').replace('\t', '').replace('\r', '')
   df_respuesta_agg = df.groupby(questions)[answers].agg(list).reset_index()
   df_pregunta_agg = df.groupby(answers)[questions].agg(list).reset_index()
   df_respuesta_agg[f'{answers}_agg'] = df_respuesta_agg[answers]
   df_pregunta_agg[f'{questions}_agg'] = df_pregunta_agg[questions]
   df_respuesta_agg = df_respuesta_agg.drop(answers, axis=1)
   df_pregunta_agg = df_pregunta_agg.drop(questions, axis=1)
   
   
   df = pd.merge(df, df_respuesta_agg, on=['Pregunta'], how='left')
   df = pd.merge(df, df_pregunta_agg, on=['Respuesta App'], how='left')
   
   vec_p = df[questions].to_list()
   vec_r = df[answers].to_list()
   
   # De los vectores de RESPUESTAS AGREGADAS, nos quedamos con la pregunta indv o las mult respuestas
   # Encaminado a conocer PREGUNTA
   vec_p_agg = df_respuesta_agg['Pregunta'].to_list()
   vec_r_agg = df_respuesta_agg['Respuesta App_agg'].to_list()
   
   # De los vectores de PREGUNTAS AGREGADAS, nos quedamos con la respuesta indv o las mult preguntas
   # Encaminado a conocer RESPUESTA
   vec_p_agg_2 = df_pregunta_agg['Pregunta_agg'].to_list()
   vec_r_agg_2 = df_pregunta_agg['Respuesta App'].to_list()
   
   vec_p = df['Pregunta'].to_list()
   vec_r = df['Respuesta App'].to_list()
   
   #lista_id = [generate_id(pregunta + respuesta) for pregunta, respuesta in zip(vec_p, vec_r)]
   
   #Añadir en el metadato de la pregunta el id de la respuesta y viceversa
   metadata_p = [{'id': generate_id(vec_p_agg[i]), 'respuesta': '|'.join([generate_id(respuesta) for respuesta in vec_r_agg[i]])} for i in range(len(vec_p_agg))]
   metadata_r = [{'id': generate_id(vec_r_agg_2[i]), 'pregunta': '|'.join([generate_id(pregunta) for pregunta in vec_p_agg_2[i]])} for i in range(len(vec_r_agg_2))]
   
   #Listas de ids
   lista_id_p = [id['id'] for id in metadata_p]
   lista_id_r = [id['id'] for id in metadata_r]
   
   
   
   # ==========================================
   ### LOAD DATA INTO VECTORSTORE ( !!! Only to vectorstore de Chroma, not vectorstore_LC)
   # ==========================================
   VECTORSTORE = vectorstore_chroma
   
   # Anadimos vectores de pregunta
   VECTORSTORE.add(
       documents=vec_p_agg,
       metadatas=metadata_p,
       ids=lista_id_p
   )
   
   # Anadimos vectores de respuesta
   VECTORSTORE.upsert(
       documents=vec_r_agg_2,
       metadatas=metadata_r,
       ids=lista_id_r)
   

   # Método de Langchain (LC) permite importar bbdd de Chroma () como una pieza a usar por LC --> para RAG (LLM + BBDD)
   vectorstore_chroma_LC = Chroma(
    client = chroma_client,
    collection_name = nombre_bbdd,
    embedding_function = embed_model
    )
   
   print('se han insertado los siguientes vectores en ChromaDB', vectorstore_chroma_LC.count())

   return vectorstore_chroma_LC
   







#####################
# LLM ONLY // LLM-RAG
#####################


### LLM Download ###
def LLM_download(id_model):
  import torch
  from transformers import BitsAndBytesConfig
  
  quantization_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_compute_dtype=torch.float16,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_use_double_quant=True,
  )
  
  model = AutoModelForCausalLM.from_pretrained(id_model, quantization_config=quantization_config, low_cpu_mem_usage=True ,pad_token_id=0) # bigscience/bloom-1b3
  tokenizer = AutoTokenizer.from_pretrained(id_model)
  return model, tokenizer

# BLOOM --->
# id_model = "bigscience/bloom-560m"
# id_model = "bigscience/bloom"  (esto es un crimen para la RAM y la memoria en disco)
# id_model = "bigscience/bloom-7b1"
# id_model = "bigscience/bloom-3b"
# id_model = "bertin-project/bertin-gpt-j-6B"
# id_model= 'LLMs/WizardLM-13B-V1.0'
# id_model = 'mistralai/Mistral-7B-Instruct-v0.1'
id_model = "mistralai/Mistral-7B-Instruct-v0.2"
id_model = 'hipnologo/falcon-7b-qlora-finetune-chatbot'
id_model = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
model, tokenizer =  LLM_download(id_model)

class LLM:
  
  def __init__():
      pass
  
  @staticmethod
  def llm(prompt: str) -> str:
      # Si el LLM ya está inicializado simplemente llámalo
      # Si no está inicializado arrancarlo

      # ==========
      # Parameters
      # =========
      
      random_param = 0.2    # (randomness: [0,1] )
      answer_len = 500  # (answer length )
      repetition_param:float = 2.0  # (avoid next-word is repeated, [1, ..])
      
      # Definimos HuggingFace Pipeline
      pipeline_model = transformers.pipeline(
          model=model, tokenizer=tokenizer,
          return_full_text=True,
          task='text-generation',
          temperature = random_param,
          max_new_tokens = answer_len,
          repetition_penalty = repetition_param
      )
      llm = HuggingFacePipeline(pipeline = pipeline_model)


      # =================
      # RAG implementation (langchain)
      # =================
      
      VECTORSTORE = vectorstore_chroma_LC
      rag_pipeline = RetrievalQA.from_chain_type(
          llm = llm,
          chain_type = 'stuff',
          retriever = VECTORSTORE.as_retriever()
      )
      
      user_query = prompt
      RAG = rag_pipeline(user_query)

      return mostrar_query([RAG['result']])

      


def chromaDB_search_ocurrencias(user_query: str) -> list:


  documents = vectorstore_chroma_LC\
    .similarity_search(query = user_query,
                        k=2)

  vec_r = vec_r_agg_2
  dict_respuestas = {generate_id(respuesta): respuesta  for respuesta in vec_r}
  # aqui te devuelve o bien ids que son para la respuesta (te lo relaciona con una pregunta) o bien una respuesta porque te lo relaciona con una pregunta
  #resultados = [documento.metadata['respuesta'] if 'respuesta' in documento.metadata.keys() else documento.page_content for documento in documents]
  resultados = [documento.metadata['respuesta'] if 'respuesta' in documento.metadata.keys() else documento.metadata['id'] for documento in documents]

  respuestas_total =[]
  for n,respuesta in enumerate(resultados):
    #print('respuesta nº',n)
    #print(respuesta) #
    respuesta = respuesta.split('|') #
    #print(respuesta)
    if len (respuesta) > 1: #
       #print(len(respuesta))
       for k in respuesta: #
          #print(k) #
          respuestas_total.append(dict_respuestas[k])
    else: #
       respuesta = respuesta[0]
       #print('añadimos',respuesta) #
       respuestas_total.append(dict_respuestas[respuesta])



  respuestas_total = [respuestas_total]
  el_vistos = set()
  valores = [el for subel in respuestas_total for el in subel]
  result = [el for el in valores if el not in el_vistos and not el_vistos.add(el)]

  return mostrar_query(result)



   
   


######################################
## EJECUTADO POR STREAMLIT CADA VEZ ## ---> app.py
######################################

import streamlit as st
def main():
    # Título de la aplicación
    st.title("Quantumstral-LLM")

    # Introducir el prompt
    prompt = st.text_area("Introduce un prompt:", value="Escribe una frase sobre el tema...")

    # Botón para generar texto
    # desencadena proceso de meter el prompt al LLM (llamda a clase LLM)
    if st.button("Generar Texto"):
        response = LLM.llm(prompt=prompt)

        # Muestra la salida generada
        st.subheader("Texto Generado: ")
        st.write(response)

        # Añade una columna para mostrar las ocurrencias
        st.sidebar.title("Información similar recopilada")

        ''' 
        # Llama a la función query para obtener las ocurrencias
        occurrences = chromaDB_search_ocurrencias(prompt=prompt)


        # Muestra las ocurrencias en la barra lateral
        for i, (text_chunk, distance, metadata) in enumerate(occurrences, start=1):
            with st.sidebar.expander(f"Fragmento {i}"):
                st.write(f"Chunck encontrado: {text_chunk}")
                st.write(f"Similaridad: {distance:.4f}")

                 # Agrega un mini encabezado para los metadatos
                st.subheader("Metadato: ")
                for key in metadata:
                    st.write(f'{key}: {metadata[key]}')
        '''

if __name__ == "__main__":
    main()