from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA 
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_core.documents import Document 
import mysql.connector
import os
from dotenv import load_dotenv
import shutil
import time
import uuid
import threading 



def cleanup_old_temp_chroma_dbs(base_path):
    """Remove diretórios temporários de Chroma DB que podem ter sobrado."""
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    for item in os.listdir(parent_dir):
        if item.startswith("chroma_db_temp_") and os.path.isdir(os.path.join(parent_dir, item)):
            temp_path = os.path.join(parent_dir, item)
            print(f"Limpeza: Encontrado diretório temporário órfão: {temp_path}. Tentando remover...")
            try:
                shutil.rmtree(temp_path)
                print(f"Limpeza: Diretório {temp_path} removido com sucesso.")
            except OSError as e:
                print(f"Limpeza: Aviso: Não foi possível remover o diretório temporário {temp_path}: {e}")
                print("Pode ser necessário remover manualmente.")
# =====================
# CONFIGURAÇÕES GERAIS
# =====================
load_dotenv()
DB_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "localhost"),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", "*****"),
    "database": os.getenv("MYSQL_DB", "rag"),
    "port": int(os.getenv("MYSQL_PORT", 3306)),
    "autocommit": True,
}

CHROMA_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
print(f"Chroma DB Path configurado para: {CHROMA_DB_PATH}")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_FILES_DIR = os.path.join(BASE_DIR, "web")
print(f"Caminho para arquivos estáticos: {STATIC_FILES_DIR}")
app = Flask(__name__, static_folder=STATIC_FILES_DIR)
CORS(app)

# =============
# VARIÁVEIS GLOBAIS
# =============
vectorstore = None
retriever = None
llm = ChatOpenAI(temperature=0)
qa_chain = None
reindex_lock = threading.Lock()  

# =============
# FUNÇÕES
# =============
def get_conn():
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except mysql.connector.Error as e:
        print(f"Erro ao conectar ao banco de dados: {e}")
        return None

def reindexar():
    """
    Recria a base vetorial a partir dos registros da tabela rag_base.
    Implementa "hot swapping": cria a nova base em um diretório temporário
    e, após o sucesso, troca para a nova base, minimizando o tempo de inatividade.
    """
    global vectorstore, retriever, qa_chain, reindex_lock

    with reindex_lock:
        print("\n--- Iniciando reindexação ---")
        conn = None
        cur = None
        textos = []
        old_chroma_path = CHROMA_DB_PATH
        new_chroma_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"chroma_db_temp_{uuid.uuid4()}")

        try:
            conn = get_conn()
            if conn is None:
                # Tenta limpar o diretório temporário se a conexão falhar cedo
                if os.path.exists(new_chroma_path):
                    shutil.rmtree(new_chroma_path)
                return False
            cur = conn.cursor()
            cur.execute("SELECT conteudo FROM rag_base")
            textos = [linha[0] for linha in cur.fetchall()]

            if not textos:
                print("Nenhum registro encontrado para reindexar. Limpando Chroma DBs.")
                if os.path.exists(old_chroma_path):
                    try:
                        shutil.rmtree(old_chroma_path)
                        time.sleep(0.5) # Pequena pausa para garantir a remoção
                    except OSError as e:
                        print(f"Aviso: Não foi possível remover o diretório Chroma DB antigo ({old_chroma_path}): {e}")
                if os.path.exists(new_chroma_path):
                    shutil.rmtree(new_chroma_path)
                return True

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            documents = [Document(page_content=t) for t in textos]
            texts = splitter.split_documents(documents)

            embeddings = OpenAIEmbeddings()
            print(f"Criando nova base Chroma DB em: {new_chroma_path}")
            new_vectorstore = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=new_chroma_path)            
            print("Nova Chroma DB persistida com sucesso no diretório temporário.")
            time.sleep(0.5) # Pausa para o SO liberar handles após a criação do Chroma DB

            # Liberar referências antigas antes de atribuir as novas
            if qa_chain is not None:
                del qa_chain
            if retriever is not None:
                del retriever
            if vectorstore is not None:
                del vectorstore
            
            time.sleep(0.5) # Pequena pausa para garantir que os handles de arquivos sejam liberados (Windows)

            vectorstore = new_vectorstore
            retriever = vectorstore.as_retriever()
            # O prompt DEVE incluir '{context}' para que a cadeia RAG funcione corretamente.
            system_template = """Sua instrução aqui e contexto        
            Use o seguinte contexto recuperado para responder à pergunta. Se a pergunta não for relevante para o contexto, diga 
            "No momento ainda nao tenho essa informação, por favor, entre em contato...."
            
            
            Contexto: {context}
            """
            human_template = "{question}"

            prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(system_template),
                HumanMessagePromptTemplate.from_template(human_template)
            ])

            # Cria a cadeia RetrievalQA passando o prompt customizado via chain_type_kwargs.
            # O 'stuff' chain_type automaticamente lida com a StuffDocumentsChain e LLMChain.
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff", # Usa o tipo 'stuff' para combinar documentos
                retriever=retriever,
                chain_type_kwargs={"prompt": prompt} # Passa o prompt customizado
            )
            print("Swap para a nova base Chroma DB e cadeia QA concluídos.")

            # Remover o diretório antigo (agora que a nova base está em uso)
            if os.path.exists(old_chroma_path) and old_chroma_path != new_chroma_path:
                try:
                    print(f"Removendo diretório Chroma DB antigo: {old_chroma_path}")
                    shutil.rmtree(old_chroma_path)
                    time.sleep(0.5) # Pequena pausa após a remoção
                except OSError as e:
                    print(f"Aviso: Não foi possível remover o diretório Chroma DB antigo ({old_chroma_path}): {e}")
                    print("Pode ser necessário remover manualmente. O sistema está usando a nova base.")

            print("Reindexação concluída com sucesso!")
            return True

        except Exception as e:
            print(f"Erro durante reindexação: {e}")
            # Em caso de erro, tenta limpar o novo diretório temporário para evitar lixo.
            if os.path.exists(new_chroma_path):
                time.sleep(0.5) # Adicionar sleep antes de tentar remover o diretório temporário em caso de erro
                try:
                    shutil.rmtree(new_chroma_path)
                except OSError as e_clean:
                    print(f"Aviso: Não foi possível limpar o diretório temporário ({new_chroma_path}) após erro: {e_clean}")
            return False
        finally:
            if cur: cur.close()
            if conn: conn.close()
            print("--- Fim da reindexação ---")


# =============
# ROTAS API
# =============
@app.route("/pergunta", methods=["POST"])
def pergunta():
    dados = request.json
    pergunta_usuario = dados.get("pergunta")
    if not pergunta_usuario:
        return jsonify({"resposta": "Por favor, forneça uma pergunta."}), 400

    with reindex_lock: # O lock é importante aqui
        if qa_chain:
            try:                          
                # O input para qa_chain é 'query' no RetrievalQA
                resposta = qa_chain.invoke({"query": pergunta_usuario})
                return jsonify({"resposta": resposta["result"]})
            except Exception as e:
                print(f"Erro ao consultar a cadeia RAG: {e}")
                return jsonify({"resposta": "Desculpe, não consegui processar sua pergunta no momento."}), 500
        else:
            return jsonify({"resposta": "O sistema RAG não está pronto. Por favor, reindexe os dados."}), 503

@app.route("/reindexar", methods=["POST"])
def reindex_route():
    ok = reindexar()
    return jsonify({"status": "ok" if ok else "vazio"})

if __name__ == "__main__":
    # Limpar diretórios temporários de Chroma DB que podem ter sobrado
    cleanup_old_temp_chroma_dbs(CHROMA_DB_PATH)
    print("Tentando reindexar na inicialização do main.py...")

    initial_reindex_success = reindexar()
    if not initial_reindex_success:
        print("Reindexação inicial falhou.")
    app.run(debug=False, host='0.0.0.0', port=8000, use_reloader=False)
