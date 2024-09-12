import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(),override=True)
########################################################################################################################
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
    elif extension == '.txt':
        from langchain_community.document_loaders import TextLoader
        print(f"loading {file}...")
        loader = TextLoader(file)
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
########################################################################################################################
def chunk_document_data(data, chunk_size=250, chunk_overlap=0):
   from langchain.text_splitter import RecursiveCharacterTextSplitter
   text_splitter= RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
   chunks = text_splitter.split_documents(data)
   return chunks
########################################################################################################################
# Embedding and Uploading to a Vector DB (PINECONE)
def insert_or_fetch_embeddings(index_name, chunks):
    import pinecone
    from langchain_community.vectorstores import Pinecone
    from langchain_openai import OpenAIEmbeddings
    # ServerlessSpec - necessary to create an index
    from pinecone import ServerlessSpec

    pc = pinecone.Pinecone() #pinecone api_key already in .env(dotenv)
    embeddings = OpenAIEmbeddings(models= 'text-embedding-3-small', dimension=1536)

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
########################################################################################################################
def ask_and_get_answer(vector_store_db, que, k=3):
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI

    llm_model = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)
    db_retriever = vector_store_db.as_retriever(search_type='similarity', search_kwargs = {'k': k})
    chain = RetrievalQA.from_chain_type(llm=llm_model, chain_type='stuff', retriever=db_retriever)
    ans = chain.run(que)
    return ans


# index_name = 'askdocument'
# vector_store = insert_or_fetch_embeddings(index_name, chunk) {create data chunk and pass 2nd arg}
#
# question = input('Ask your question here :')
# response = ask_and_get_answer(vector_store, question)
# print(response)
########################################################################################################################
# Embedding and Uploading to a Vector DB (Chroma DB)
def create_embedding_chroma(chunk, persist_directory='./chroma_db'):
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)
    vectorstore = Chroma.from_documents(chunk, embedding=embeddings, persist_directory=persist_directory)
    return vectorstore

def load_embeddings_chroma(persist_directory='./chroma_db'):
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return vector_store
########################################################################################################################
########################################################################################################################

# Adding Memory (Chat History)
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain  # Import class for building conversational AI chains
from langchain.memory import ConversationBufferMemory  # Import memory for storing conversation history

# Instantiate a ChatGPT LLM (temperature controls randomness)
llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)

# Configure vector store to act as a retriever (finding similar items, returning top 5)
vector_store = load_embeddings_chroma()
retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 5})
# Create a memory buffer to track the conversation
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
crc = ConversationalRetrievalChain.from_llm(
    llm=llm,  # Link the ChatGPT LLM
    retriever=retriever,  # Link the vector store based retriever
    memory=memory,  # Link the conversation memory
    chain_type='stuff',  # Specify the chain type
    verbose=False  # Set to True to enable verbose logging for debugging
)
########################################################################################################################
########################################################################################################################
# Using custom prompt template
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)
retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 5})
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)


system_template = r'''
Use the following pieces of context to answer the user's question.
Before answering translate your response to Spanish.
If you don't find the answer in the provided context, just respond "I don't know."
---------------
Context: ```{context}```
'''

user_template = '''
Question: ```{question}```
'''

messages= [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template(user_template)
]

qa_prompt = ChatPromptTemplate.from_messages(messages)

cc = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    chain_type='stuff',
    combine_docs_chain_kwargs={'prompt': qa_prompt },
    verbose=True
)
########################################################################################################################

#############################                        STREAMLIT UI                  #####################################
#Clear histore from st.session state
def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']
#####################################

if __name__ == '__main__':

    st.image('img.png', width=75)
    st.subheader('LLM Document Q&A Application')



    with st.sidebar:
        api_key = st.text_input('OpenAI Key : ', type='password')
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key
        upload_file = st.file_uploader('Upload your file: ', type=['pdf', 'docx', 'txt'])
        chunk_size = st.number_input('Chunk size: ', min_value=100, max_value=2048, value=300, on_change=clear_history)
        k = st.number_input('k: ', min_value=1, max_value=20, value=3, on_change=clear_history)
        add_data = st.button('Add Data',  on_change=clear_history)

        if upload_file and add_data:
            with st.spinner('Loading, Chunking and Embedding...'):
                byte_data = upload_file.read()
                file_name = os.path.join('./', upload_file.name)
                with open(file_name, 'wb') as f:
                    f.write(byte_data)

                    data = load_document(file_name)
                    chunks = chunk_document_data(data, chunk_size=chunk_size)
                    st.write(f'Chunk size: {chunk_size}, Chunks: {len(chunks)}')

                    vector_store = create_embedding_chroma(chunks)
                    st.session_state.vs = vector_store
                    st.success('File loaded, chunked and embedded successfully')

    que = st.text_input('Ask a question about content of your files')
    if que:
        if 'vs' in st.session_state:
            vectorstore = st.session_state.vs
            st.write(f'k: {k}')
            answer = ask_and_get_answer(vectorstore, que, k)
            st.text_area('LLM Answer: ', value=answer)

            #Adding chat history to streamlit session state
            st.divider()
            if 'history' not in st.session_state:
                st.session_state.history = ''
            value = f'Question: {que} \nAnswer: {answer}'
            st.session_state.history = f'{value} {"-"*100} \n {st.session_state.history}'
            h_chat = st.session_state.history
            st.text_area(label='Chat History', value=h_chat, key='history', height=400)



