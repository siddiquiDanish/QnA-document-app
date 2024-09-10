
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(),override=True)

def load_document(file):
    import os
    #os.path.splitext() method in Python is used to split the path name into a pair root and ext.
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain_community.document_loaders import PyPDFLoader
        print(f"loading {file}...")
        loader = PyPDFLoader(file)
        data = loader.load()
    elif extension == '.docx':
        from langchain_community.document_loaders import Docx2txtLoader
        print(f"loading {file}...")
        loader = Docx2txtLoader(file)
    else :
        print('Document format not supported')
        return None

    data = loader.load()
    return data

def load_from_wikipedia(query, lang='en', load_max_docs=2):
    from langchain_community.document_loaders import WikipediaLoader
    loader = WikipediaLoader(query=query, lang=lang, load_max_docs=load_max_docs )
    wiki_data = loader.load()
    return wiki_data

def chunk_document_data(data, chunk_size=250, chunk_overlap=0):
   from langchain.text_splitter import RecursiveCharacterTextSplitter
   text_splitter= RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
   chunks = text_splitter.split_documents(data)
   return chunks

# Embedding and Uploading to a Vector DB

def insert_or_fetch_embeddings(index_name, chunks):
    import pinecone
    from langchain_community.vectorstores import Pinecone
    from langchain_openai import OpenAIEmbeddings
    # ServerlessSpec - necessary to create an index
    from pinecone import ServerlessSpec

    pc = pinecone.Pinecone() #pinecone api_key already in .env(dotenv)
    embeddings = OpenAIEmbeddings(models= 'text-embedding-3-small', dimensions=1536)

    if index_name in pc.list_indexes().names():
        print(f"Index name {index_name} already exists. Loading embeddings...", end='')
        vector_store = Pinecone.from_existing_index(index_name, embeddings)

    else:
        print(f"Creating new index {index_name} and embeddings...", end='')
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric='cosine',
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        vector_store = Pinecone.from_documents(chunks,embeddings,index_name=index_name)

    return  vector_store

def delete_pinecone_index(index_name='all'):
    import pinecone
    pc = pinecone.Pinecone()  #pinecone api_key already in .env(dotenv)

    if index_name == 'all':

        for index_name in pc.list_indexes().names():
            pc.delete_index(index_name)
        print("Deleted all index.")

    else :
        pc.delete_index(index_name)
        print(f"Deleted {index_name} index.")






