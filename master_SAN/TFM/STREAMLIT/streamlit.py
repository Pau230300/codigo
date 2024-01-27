#########################
#### DEMO STREAMLIT #####
#########################

# Función de configuración, executed once
import streamlit as st
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import torch
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


# Función para generar un hash MD5 como ID único
def generate_id(html_content):
    from hashlib import md5
    html_content = str(html_content)
    return md5(html_content.encode('utf-8')).hexdigest()

# Función que contiene configuración
def config():
  #! Creamos el pipeline de embedings a guardar en nuestra base de datos vectorial
  #! (pinecone)
  from torch import cuda
  from langchain.embeddings.huggingface import HuggingFaceEmbeddings
  
  embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'

  device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

  embed_model = HuggingFaceEmbeddings(
      model_name=embed_model_id,
      model_kwargs={'device': device},
      encode_kwargs={'device': device, 'batch_size': 32}
  )

  # ---
  st.session_state['embed_model'] = embed_model
  # ---

  import os
  import pinecone
  #from google.colab import userdata

  # get API key from app.pinecone.io and environment from console
  api_key_0 = '7be80e83-b2c8-495d-b148-5474df2c2cd4'
  environment_0 = 'gcp-starter'
  pinecone.init(api_key = api_key_0,
                 environment = environment_0)

  import time
  index_name = 'santander-public-web'

  if index_name not in pinecone.list_indexes():
      pinecone.create_index(
          index_name,
          dimension=len(embeddings[0]),
          metric='cosine'
      )
      # wait for index to finish initialization
      while not pinecone.describe_index(index_name).status['ready']:
          time.sleep(1)

  index = pinecone.Index(index_name)
  index.describe_index_stats()

  # Leer excel
  import pandas as pd
  ruta ='/Users/pau/repo_pau/master_SAN/TFM/FAQS_Particulares.xlsx'
  df = pd.read_excel(ruta)
  

  df = df[['Pregunta', 'Respuesta App']]
  df.fillna('NONE')
  #df.head(30)

  

  


  #Separando el embeding de pregunta y respuesta
  # DATAFRAME
  import unicodedata
  def quitar_tildes(texto):
      return ''.join((c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn'))

  df_p = df[['Pregunta']].fillna('NONE')
  df_r = df[['Respuesta App']].fillna('NONE')

  df_p['Pregunta'] = df_p['Pregunta'].apply(quitar_tildes).str.replace('\n', '').replace('\t', '').replace('\r', '')
  df_r['Respuesta App'] = df_r['Respuesta App'].apply(quitar_tildes).str.replace('\n', '').replace('\t', '').replace('\r', '')

  #df_p
  #df_r
  vec_p = df_p['Pregunta'].to_list()
  vec_r = df_r['Respuesta App'].to_list()

  ## ---
  st.session_state['vec_r'] = vec_r
  st.session_state['vec_p'] = vec_p
  ## ---

  ### El embedding lo hace un transformer de HuggingFace
  lista_id = [generate_id(pregunta + respuesta) for pregunta, respuesta in zip(vec_p, vec_r)]
  embeds_p = embed_model.embed_documents(vec_p)  # Vectorización
  embeds_r = embed_model.embed_documents(vec_r)  # Vectorización

  ## vector_emb_p : R384 // vector_emb_r : R384
  data = [{"question": vector_emb_p, "answer": vector_emb_r} for vector_emb_p, vector_emb_r in zip(embeds_p, embeds_r)]
  #len(data), len(vec_p), len(vec_r)

  # Convert data to a format suitable for upsert, Generar los ids de pregunta/respuesta
  index_data_q = [{"id": generate_id(vec_p[i]), "vector": item["question"], "metadata": {"answer": generate_id(vec_r[i])}} for i, item in enumerate(data)]
  index_data_r = [{"id": generate_id(vec_r[i]), "vector": item["answer"], "metadata": {"question": generate_id(vec_p[i])}} for i, item in enumerate(data)]


  #Añadir en el metadato de la pregunta el id de la respuesta y viceversa
  metadata_p = [{'id': generate_id(vec_p[i]), 'respuesta': generate_id(vec_r[i]) } for i, item in enumerate(data)]
  metadata_r = [{'id': generate_id(vec_r[i]), 'pregunta': generate_id(vec_p[i]) } for i, item in enumerate(data)]

  #Listas de ids
  lista_id_p = [id['id'] for id in metadata_p]
  lista_id_r = [id['id'] for id in metadata_r]

  # id y pregunta
  id_text_pregunta = {generate_id(pregunta): pregunta  for pregunta in vec_p}
  id_text_respuesta = {generate_id(respuesta): respuesta  for respuesta in vec_r}

  len(embeds_p)

  tam = 1000

  for i in range((len(embeds_p)//tam)+1):
    if i == 0:
      index_start = 0
      index_end = tam
    elif i == (len(embeds_p)//tam):
      index_start = i * tam
      index_end = len(embeds_p)
    else:
      index_start = index_end
      index_end = (i+1) * tam

    index.upsert(vectors=zip(lista_id_p[index_start:index_end], embeds_p[index_start:index_end], metadata_p[index_start:index_end]))
    index.upsert(vectors=zip(lista_id_r[index_start:index_end], embeds_r[index_start:index_end], metadata_r[index_start:index_end]))

  pinecone.list_indexes()
  index.describe_index_stats()
  
  # ---
  st.session_state['index'] = index
  # ---
        
  return index, embed_model, vec_r, vec_p
  #return st.session_state['index'], st.session_state['embed_model'], st.session_state['vec_r'], st.session_state['vec_p']
index, embed_model , vec_r, vec_p = config()
  
# Generamos vectorstore a través de la integración de PINECODE DATA Y HUGGINGFACE TRANSFORMERS
# a través de LANGHAIN
def create_vectorstore(index, embed_model):
  from langchain.vectorstores import Pinecone
  id_field = 'id'  # Campo que contiene el id del html con la mayor similitud
  vectorstore = Pinecone(
      index,
      embed_model.embed_query,
      id_field
      )
  
  return vectorstore
vectorstore = create_vectorstore(index, embed_model)

# --------          ----------------            ----------

### FRONT PARÁMETROS
#####################
    
# Ponemos un título xuli
st.title("Chatbot app - DEMO")

# Código HTML que contiene foto que queramos
## he puesto un gatito por la broma
html_con_imagen = """
<img src="https://www.lanacion.com.ar/resizer/v2/la-paradoja-del-gato-de-schrodinger-es-la-mas-DCKLNNECFZCVBEAGM63FV54S3I.jpg?auth=d3b94ac7970f7cbb7d261115862034a19086a91ec480ea24fd873cc59f93eb45&width=420&height=280&quality=70&smart=true" alt="Gatito">
"""

# Mostrar la imagen mediante HTML
st.markdown(html_con_imagen, unsafe_allow_html=True)

# Área de entrada de texto para el usuario
input_usuario = st.text_input("Ingresa tu mensaje:")

dicc_all_respuesta_chatbot = {}
historial = {}

### FUNCIÓN CHATBOT 
###################
num_respuestas = 3
def obtener_respuesta(input_usuario, vec_r, vectorstore, historial, num_respuestas):
    
  # Aquí deberías llamar a tu modelo de chatbot y obtener la respuesta
  ## añadimos lo que un dia se habló

  size_historial = len(historial)

  # simulamos comportamiento de streamlit borrando variables
  dicc_all_respuesta_chatbot = {}

  query = input_usuario
  k_0 = num_respuestas
  documents = vectorstore.similarity_search(
    query,  # the search query
    k=k_0)  # returns top 3 most relevant chunks of text

  
  dict_respuestas = {generate_id(respuesta): respuesta  for respuesta in vec_r}
  resultados = [documento.metadata['respuesta'] if 'respuesta' in documento.metadata.keys() else documento.page_content for documento in documents]

  
  saludo_inicial = f"¡Hola! Has preguntado por: {input_usuario}, te mostraré {num_respuestas} posibles respuestas"
  print('saludo_inicial, key:',size_historial)
  dicc_all_respuesta_chatbot[size_historial] = saludo_inicial
 
  for respuesta,documento in zip(resultados,documents):
   
    size_historial+=1
    n = size_historial
    respuesta_chatbot = f" Mi respuesta n.{n}  es: \n {dict_respuestas[respuesta]}"
    print('respuestas, key:',n) # aumentamos 1 en cada iteracion
    dicc_all_respuesta_chatbot[n] = respuesta_chatbot
    
    #print('\n')
    #if documento.page_content in lista_id_p:
    #  print('El resultado te relaciona el input con una P:', '\nque es:', id_text_pregunta[documento.page_content])
    #elif documento.page_content in lista_id_r:
    #  print('El resultado te relaciona el input con una R:', '\nque es:',id_text_respuesta[documento.page_content] )

    
  return dicc_all_respuesta_chatbot

 
## OUTPUT
#########
if st.button("Enviar"):
    ## este diccionario genera las respuestas puntuales
    dict = obtener_respuesta(input_usuario,vec_r, vectorstore, historial, num_respuestas)

    import json
    ## si es diccionario_historial es mayor que diccionario lectura
    try:
      with open('dict.json', 'r') as archivo:
       historial = json.load(archivo)
       print('lee')
       with open('dict.json','w') as archivo:
         historial.update(dict)
         json.dump(historial, archivo)
         print('sobreescribe')
    except:
     with open('dict.json','w') as archivo:
       historial = dict
       json.dump(dict, archivo)
       print('crea y escribe 1º vez')

    

    # Obtener la respuesta del chatbot
    import numpy as np
    all_indexes = list(historial.keys())
    chatbot_index = [ str(i) for i in list(np.arange(0, len(historial), num_respuestas + 1)) ]
    user_index = [a for a in [ str(i) for i in all_indexes] if a not in chatbot_index]
    
    for i in historial.keys():
     cuadro_text = historial[i]
     print(cuadro_text)

     if str(i) in chatbot_index:
      
      print('chtabot', i)
      st.text_area("CHATBOT:", cuadro_text, height=25)

     elif str(i) in user_index:
       print('USER', i)
       st.text_area("RESPUESTA:", cuadro_text, height=100)
       
       
     