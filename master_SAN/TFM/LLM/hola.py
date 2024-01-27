# Params BBDD

nombre_bbdd = "langchain"
# Modelo de Embedding (esto es con método de Chroma)
embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name = id_embedder
)
# otra forma de definir el embedding atraves de langchain
embed_model = HuggingFaceEmbeddings(
      model_name=id_embedder)
# Generamos entorno de CHROMBA (como un esquema)
#chroma_client = chromadb.PersistentClient(path='./CHROMA')
chroma_client = chromadb.Client()
try:
  chroma_client.delete_collection(name='langchain')
except:
  pass
print('no deberia haber niguna colleccicón:' ,chroma_client.list_collections())
#chroma_client.delete_collection(name='langchain')
# Creamos vectorstore desde CHROMADB
vectorstore_chroma = chroma_client \
  .get_or_create_collection(
     name = 'langchain',
     embedding_function = embedder,
     metadata={"hnsw:space": "cosine"})

print('ahora si, debe estar langchain:' ,chroma_client.list_collections())
print('el vectorstore debe estar vacio: ', vectorstore_chroma.count())


# Creamos vectorstore desde LANGCHAIN
vectorstore_chroma_LC = Chroma(
   client = chroma_client,
   collection_name = 'langchain',
   embedding_function = embed_model)

# Esto permite tener un insight de bbdd creada
vectorstore_chroma.peek()
vectorstore_chroma.count()
vectorstore_chroma.get()