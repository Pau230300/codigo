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

#from transformers import BloomForCausalLM
#from transformers import BloomTokenizerFast


# El modelo está entrenado y funciona por sí solo,
# sin embargo nosotros queremos aplicar RAG, es decir
# que en el pipeline se integre la base de datos en la que se 
# almacena el FAQ.




########################
# VECTORSTORE : CHROMADB
########################


# Params BBDD
id_sent_transf_embedd_model = "paraphrase-multilingual-mpnet-base-v2"
nombre_bbdd = "ENCOMIENDA"

# Modelo de Embedding (esto es con método de Chroma)
embedder = embedding_functions\
    .SentenceTransformerEmbeddingFunction(
        model_name = id_sent_transf_embedd_model
)

# otra forma de definir el embedding atraves de langchain
embed_model = HuggingFaceEmbeddings(
      model_name=id_sent_transf_embedd_model
)

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

# Esto permite tener un insight de bbdd creada
vectorstore_chroma.peek()
vectorstore_chroma.count()
vectorstore_chroma.get()



# =========
# READ DATA 
# =========

def generate_id(html_content):
      html_content = str(html_content)
      return md5(html_content.encode('utf-8')).hexdigest()
  
def quitar_tildes(texto):
      return ''.join((c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn'))
  


# Leer excel
ruta ='/Users/pau/repo_pau/master_SAN/TFM/FAQS_Particulares.xlsx'
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


#vec_p_agg = df_respuesta_agg[questions].to_list()
#vec_r_agg = df_respuesta_agg[f'{answers}_agg'].to_list()
#vec_p_agg_2 = df_pregunta_agg[f'{questions}_agg'].to_list()
#vec_r_agg_2 = df_pregunta_agg[answers].to_list()

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
    ids=lista_id_r
)




# Para hacer queries sobre BBDD desde libreria de CHROMA
def query_user(user_query: str) -> list:
  
  query = vectorstore_chroma.query(
    query_texts=[user_query],
    n_results=5
  )


  #query_cond = collection.query(
  #  query_texts="phone manufacturers", 
  #  n_results=5, 
  #  where = {'GICS Sector': 'Information Technology'}
  #  where_document= {"$contains": "Apple"}
  # )

  l = []
  for md in query['metadatas'][0]:
    # si la bbdd relaciona la query user con una pregunta (nos devuelve un vector con 'respuesta' como metadato)
    # entonces estamos en caso óptimo, tenemos una respuesta tipo para la query user
    if ('respuesta' in md.keys()):
      l.append(md['respuesta'].split('|'))
    # si la bbdd relaciona la query user con una respuesta (nos devuelve un vector con 'pregunta' como metadato)
    # entonces lo que hacemos es obtener el id de la respuesta tipo y será la que mostraremos
    else:
      l.append([md['id']])

  l_sin_dup = []
  el_vistos = set()
  valores = [el for subel in l for el in subel]
  result = [el for el in valores if el not in el_vistos and not el_vistos.add(el)]
  respuestas = (vectorstore_chroma.get(
      ids = result,
      include = ["documents"]
  ))

  reorder_index = [respuestas['ids'].index(resultado) for resultado in result]

  return [respuestas['documents'][index] for index in reorder_index]

def mostrar_query(resultados: list) -> None:
  import textwrap
  LINE_WIDTH = 80
  for i, resultado in enumerate(resultados):
    wrapped_string = textwrap.wrap(resultado, width=LINE_WIDTH)
    formatted_string = '\n'.join(wrapped_string)
    print(f"Respuesta_{i}:\n")
    print(formatted_string)
    print('\n\n')




# Para hacer queries sobre BBDD desde libreria de LANGCHAIN que puede importar chroma como una pieza

# Método de Langchain (LC) permite importar bbdd de Chroma () como una pieza a usar por LC --> para RAG (LLM + BBDD)
vectorstore_chroma_LC = Chroma(
    client = chroma_client,
    collection_name = nombre_bbdd,
    embedding_function = embed_model
)

def query_user_2(user_query: str) -> list:
  
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
  
  return result





#####################
# LLM ONLY // LLM-RAG
#####################

# ==========
# Definition
# ==========

# id_model = "bigscience/bloom-560m"
# id_model = "bigscience/bloom"
# id_model = "bigscience/bloom-7b1"
# id_model = "bigscience/bloom-3b"
# id_model = "berkeley-nest/Starling-LM-7B-alpha"
id_model = "bofenghuang/vigogne-7b-chat"
id_model = "bertin-project/bertin-gpt-j-6B"
id_model='TheBloke/FashionGPT-70B-V1.1-AWQ'
id_model = "fangloveskari/ORCA_LLaMA_70B_QLoRA"

model = AutoModelForCausalLM.from_pretrained(id_model, use_cache=True) # bigscience/bloom-1b3
tokenizer = AutoTokenizer.from_pretrained(id_model, use_cache=True)
set_seed(123)
model.__class__.__name__
model.eval()

# ==========
# Parameters
# =========

random_param = 0.5    # (randomness: [0,1] )
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


# ============
# Querying LLM
# ============


# preguntas evaluativas
# user_query = 'hola,¿ qué tal estas?' # (what you want to ask)
user_query = '¿Qué diferencia hay entre la firma electrónica y la clave de acceso?'
user_query = '¿Qué es el código IBAN?'
user_query = 'Que es el codigo SWIFT o BIC?'
user_query = '¿Que diferencia hay entre hipoteca fija y variable?'
user_query = 'hola'

user_query = quitar_tildes(user_query)

start = time.time()
print('CONSULTA A LA BBDD DE CHROMA POR SIMILARITY SEARCH \n')
user_query = quitar_tildes(user_query)
print('USER_QUERY: ', user_query, '\n')
resultados_2 = query_user_2(user_query)
mostrar_query(resultados_2)
end = time.time()
print('Tiempo de ejecución:', -start + end)


start = time.time()
print('RESPUESTAS DEL LLM (sin acceso a FAQs) \n')
print('USER_QUERY', user_query, '\n')
answ = llm(prompt = user_query)
mostrar_query([answ])
#output = print(answ)
end = time.time()
print('Tiempo de ejecución:', - start + end)

print('========================== \n')


# =================
# RAG implementation (langchain)
# =================

VECTORSTORE = vectorstore_chroma_LC
rag_pipeline = RetrievalQA.from_chain_type(
    llm = llm,
    chain_type = 'stuff',
    retriever = VECTORSTORE.as_retriever()
)

print('RESPUESTAS DEL LLM CON RAG IMPLEMENTADO (con acceso a FAQs) \n')
start = time.time()
RAG = rag_pipeline(user_query)
print(RAG['query'])
mostrar_query([RAG['result']])
#print(RAG['result'])
end = time.time()
print('Tiempo de ejecución:', - start + end)








#####################
# NEXT WORD GENERATOR
#####################

def f_next_word_generator(starting_sentence):

  input_ids = tokenizer(starting_sentence, return_tensors='pt')
  sample = model.generate(**input_ids,
                          max_length=200,
                          num_beams = 4,
                          num_beam_groups =2,
                          top_k=2,
                          temperature=0.9,
                          repetition_penalty=2.0,
                          diversity_penalty=0.9)
  return print(tokenizer.decode(sample[0]))
  
context = 'Quiero que seas un chatbot cuya función sea responder preguntas a clientes del Banco Santander'
context='Banco Santander codigo BIC o SWIFT'
f_next_word_generator(context)




#####################
# LLM CON CONTEXTO
#####################


def llm_con_contexto():
  print('RESPUESTAS DEL LLM + contexto (sin acceso a FAQs) \n')
  from langchain import PromptTemplate, LLMChain
  
  template = """[INST] Imagina un escenario en un banco donde un chatbot avanzado, como ChatGPT, está integrado en el sistema de atención al cliente
  para asistir con preguntas frecuentes (FAQs). Este chatbot está programado para comprender y responder consultas relacionadas con servicios bancarios
  como la apertura de cuentas, opciones de crédito, inversiones, transferencias internacionales y medidas de seguridad. El chatbot tiene la capacidad de
  ofrecer respuestas personalizadas y detalladas, guiando a los usuarios a través de procesos complejos y proporcionando consejos financieros útiles. Además,
  está equipado para aprender de las interacciones pasadas, mejorando constantemente su capacidad para asistir a los clientes de manera más eficiente.
  El chatbot está disponible las 24 horas del día, los 7 días de la semana, accesible a través del sitio web del banco, aplicaciones móviles y quioscos en las sucursales,
  asegurando que los clientes reciban asistencia inmediata en cualquier momento y lugar.
  
  Limítate simplemente a responder con respuestas cortas referentes al FAQ
  {context}
  {question} [/INST] 
  """
  
  prompt = PromptTemplate(
     template = template,
      input_variables = ["question","context"]
  )
  
  llm_with_context = LLMChain(prompt = prompt, llm = llm )
  
  contexto_moldear_respuesta_a_query = "Tu eres un chatbot"
  answer = llm_with_context.run({"question": user_query,
                                 "context": contexto_moldear_respuesta_a_query})
  print(answer)
  print('========================== \n')
  
