import os
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter

from langchain_community.vectorstores import Chroma

from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings

from load_docs import load_docs
from streamlit_chat import message
import streamlit as st



load_dotenv()



os.environ["AZURE_OPENAI_API_KEY"] =os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")



model = AzureChatOpenAI(
    azure_deployment="gpt-4o-mini",  # Deployment name from Azure
    azure_endpoint=endpoint,
    api_version="2024-05-01-preview",
    temperature=0.7
)

embedding_model = AzureOpenAIEmbeddings(
    model="text-embedding-3-large",
    # dimensions: Optional[int] = None, # Can specify dimensions with new text-embedding-3 models
    azure_endpoint="",
    api_key="", # Can provide an API key directly. If missing read env variable AZURE_OPENAI_API_KEY
    openai_api_version="2024-02-01", # If not provided, will read env variable AZURE_OPENAI_API_VERSION
    dimensions=384

)


docs = load_docs()
chat_history = []

text_split = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

vectordb =Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="./data",
)



qa_chain = ConversationalRetrievalChain.from_llm(
    llm=model,
    retriever=vectordb.as_retriever(search_kwargs={"k": 7}),
    return_source_documents=True,
    verbose=False
)


st.title("Docs QA Bot using Langchain")
st.header("Ask anything about your documents... 🤖")

if 'generated' not in st.session_state:
    st.session_state['generated'] = []
    
if 'past' not in st.session_state:
    st.session_state['past'] = []
    
def get_query():
    input_text = st.chat_input("Ask a question about your documents...")
    return input_text


# retrieve the user input
user_input = get_query()
if user_input:
    result = qa_chain({'question': user_input, 'chat_history': chat_history})
    st.session_state.past.append(user_input)
    st.session_state.generated.append(result['answer'])
    
    
if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])):
        message(st.session_state['generated'][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i)+ '_user')
