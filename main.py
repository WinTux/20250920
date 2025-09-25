# source bin/activate
# pip install flask groq ollama
# ollama serve
# export GROQ_API_KEY="mi_api_key_desde_iPhone (tel o note)"
# http://127.0.0.1:5000

from flask import Flask, render_template, request, jsonify, Response
from groq import Groq
import ollama as ollama_client
import os

app = Flask(__name__)




@app.route('/')
def hello_world():
    return render_template('index.html')


clienteGroq = Groq(api_key=os.getenv("GROQ_API_KEY"))
mensajeGroq = [{"role":"system", "content":"Eres el personaje Mario Bros del videojuego, solamente hablas Español"}]

@app.route('/chatGroq')
def chateandoGroq():
    return render_template('chatgroq.html')

@app.route('/chatGroqAPI', methods=['POST'])
def chateandoGroqAPI():
    mensaje_usuario = request.json.get("mensajeUsuario")
    print(mensaje_usuario)
    if not mensaje_usuario:
        return jsonify({"error":"No se recibio un mensaje"})

    mensajeGroq.append({"role":"user", "content":mensaje_usuario})

    respuesta = clienteGroq.chat.completions.create(
        model="gemma2-9b-it",
        messages=mensajeGroq    
    )



    respuesta_texto = respuesta.choices[0].message.content
    mensajeGroq.append({"role":"assistant", "content":respuesta_texto})

    return jsonify({"mensajeBot":respuesta_texto})



















historialOllama = [{"role":"system", "content":"Eres Kirbi el personaje de un videojuego, eres poderoso, valiente, solamente hablas Español"}]

@app.route('/chatOllama')
def chatOllama():
    return render_template('chatOllama.html')

@app.route('/chateandoConOllama', methods=['POST'])
def chateandoOllama():
    
    mensaje_usuario = request.json.get("mensajeUsuario")

    if not mensaje_usuario:
        return jsonify({"error":"No se recibio un mensaje"})

    historialOllama.append({"role":"user", "content":mensaje_usuario})

    respuesta = ollama_client.chat(
        model="gemma2:2b",
        messages=historialOllama
    )

    respuesta_texto = respuesta['message']['content']
    historialOllama.append({"role":"assistant", "content":respuesta_texto})

    return jsonify({"mensajeBot":respuesta_texto})





























historialOllamaStream = [{"role":"system", "content":"Eres Steve Jobs, con ligeras tendencias narcisistas, siempre propositivo y con sugerencias disrruptivas solamente hablas Español"}]

@app.route('/chatOllamaStream')
def chatOllamaStreamVentana():
    return render_template('chatOllamaStream.html')


@app.route('/chatOllamaStreamChat', methods=['POST'])
def chatOllamaStreamChat():
    data = request.json
    mensaje_usuario = data.get("mensaje") if data else None

    if not mensaje_usuario:
        return jsonify({"error":"No se recibio un mensaje"})
    
    historialOllamaStream.append({"role":"user", "content":mensaje_usuario})

    
    def generar():
        respuesta_texto = ""
        for chunk in ollama_client.chat(
            model="gemma2:2b",
            messages=historialOllamaStream,
            stream=True
        ):
            if "message" in chunk and "content" in chunk["message"]:
                token = chunk["message"]["content"]
                respuesta_texto += token
                yield token

        historialOllamaStream.append({"role":"assistant", "content":respuesta_texto})

    return Response(generar(), mimetype="text/plain")























#pip install chromadb langchain_ollama

import chromadb
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
import uuid

llm = OllamaLLM(model="llama2:latest")
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection("chat_history")


def load_history():
    history = []
    results = collection.get(include=["documents", "metadatas"])

    for doc, meta in zip(results["documents"], results["metadatas"]):
        if meta["role"] == "human":
           history.append(HumanMessage(content=doc))
        else:
            history.append(AIMessage(content=doc))
    return history

def save_message(role, content):
    collection.add(
        ids=[str(uuid.uuid4())],
        documents=[content],
        metadatas=[{"role": role}]
    )


 

@app.route("/chatOllamaMemoriaLP")
def chatOllamaMemoria():
    return render_template("chatOllamaMemoriaLP.html")

@app.route("/chatOllamaMemoriaLPchat", methods=["POST"])
def chatOllamaMemoriaLPchat():
    data = request.get_json()
    mensaje_usuario = data.get("mensajeUsuario")
    if not mensaje_usuario:
        return jsonify({"error":"No se recibio un mensaje"})

    history = load_history()

    promt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "Tu nombre es Carlitos. Responde de forma breve y haz preguntas al usuario para saber más. Solo hablas español"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ]
    )

    chain = promt_template | llm

    response = chain.invoke({"input":mensaje_usuario, "chat_history":history})
    
    save_message("human", mensaje_usuario)
    save_message("ai", response)

    return jsonify({"mensajeBot":response})







    

            


            





































if __name__ == '__main__':
    app.run(debug=True)
