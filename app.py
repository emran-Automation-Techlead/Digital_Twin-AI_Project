import os
from openai import OpenAI
import gradio as gr
import uuid
import chromadb
from pprint import pprint
import json
import requests
import random
import time
import logger

#----------------------------------------
#Setup
#----------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


if OPENAI_API_KEY is None:
    raise Exception("API key is missing")
client = OpenAI()
print(OPENAI_API_KEY[:8])


#------------------------------------------
#Document
#------------------------------------------

document_overview = """Mohammed Imran Ali is a results-driven IT Operations Leader and Cloud & AI Solutions Architect based in Hyderabad, Experienced IT Service Desk and Operations Leader with 11+ years in enterprise IT support, specializing in service delivery, cloud technologies, and AI-driven solutions. Proven track record in leading global teams, optimizing ITSM processes, and ensuring high-quality service delivery aligned with business goals.
Skilled in AWS cloud architecture, automation, and modern AI technologies (LLMs, RAG, Digital Twin systems), with a focus on transforming traditional IT operations into scalable, intelligent, and proactive platforms.
😄 About Me (Fun Side)
I’m not just about tech—I like to keep things balanced and interesting 😄
🏏 Big fan of cricket and football—love the strategy and team spirit
🐾 I have a soft spot for animals, especially cats and dogs
🎨 I naturally gravitate toward blue, black, and clean minimal vibes (yes, even in UI 😄)
🍰 I can never say no to desserts—Kunafa is a clear winner
🚫 Not really a fan of overly spicy or heavily processed food
✈️ I enjoy traveling, exploring new cultures, and trying different cuisines
💪 I try to stay active and maintain a balanced lifestyle (tech + fitness = ideal combo)
"""

document_education = """Education
Post Graduate Diploma – Cyber Crime
University of Hyderabad | Apr 2013 – Nov 2015
Bachelor of Computer Science
Osmania University | Jul 2007 – Dec 2012"""

document_professional_experience = """ 💼 Professional Experience

Assistant Manager – IT Service Desk
Cotiviti | Jan 2019 – Present
(11+ years of total experience in enterprise IT support)

Lead and manage global IT service desk operations supporting enterprise users
Ensure SLA compliance, incident management, and service delivery excellence
Implement ITIL-based processes and drive continuous improvement initiatives
Automate ITSM workflows using ServiceNow and Jira
Manage Microsoft 365 administration and enterprise IT infrastructure
Oversee team performance, workforce planning, and stakeholder collaboration

Process Developer (L2/L3 IAM Support)
Genpact | Sep 2014 – Sep 2018

Provided L2/L3 support for Identity and Access Management (IAM)
Managed user lifecycle: provisioning, de-provisioning, and access control
Administered Active Directory, Office 365, Citrix, and RDP environments
Enhanced ServiceNow ITSM workflows and knowledge management systems
Ensured compliance with security and access governance standards
🧠 Core Competencies

Cloud & Architecture

AWS (EC2, S3, IAM, CloudFormation)
Solution Architecture & System Design
Infrastructure as Code (IaC)

DevOps & Automation

CI/CD Pipelines
Jenkins, Ansible, Docker
API Integrations & Automation

AI & Emerging Technologies

Retrieval-Augmented Generation (RAG)
Agentic AI Systems & LLM Tool Calling
LangGraph, Hugging Face
Conversational AI & Multi-Agent Systems

IT Operations & Service Management

ITIL-based Service Management
Incident & Major Incident Management
Service Desk Operations & Troubleshooting

Security & Identity

Identity & Access Management (IAM)
Active Directory & Microsoft 365
Information Security & Access Governance

Leadership & Delivery

Team Leadership & People Management
Stakeholder Management
Agile, Scrum & SDLC
Strategic Initiatives & Performance Optimization
🌟 Professional Summary

A results-driven IT leader with a strong blend of technical expertise, operational leadership, and strategic thinking, delivering scalable and user-centric IT solutions.

Experienced across IT operations, cloud architecture, and AI-driven systems, with a focus on transforming traditional support models into intelligent, automated, and proactive platforms.

🚀 Key Focus Areas
Driving digital transformation initiatives
Building and leading high-performing teams
Leveraging AI to modernize service desk operations
Continuously learning and adopting emerging technologies
🌍 Languages
English: Full Professional Proficiency
French: Elementary Proficiency"""

document_certifications = """Certifications
ITIL® v3 Foundation – PeopleCert
Lean Six Sigma Green Belt
PCAP – Certified Associate in Python Programming
AWS Certified Cloud Practitioner
Agile Scrum Master
French Language A1 Certification"""

document_hobbies_interests = """Hobbies & Interests
Exploring AI & Emerging Technologies – Actively experimenting with AI tools, LLMs, and automation to stay current with industry trends
Continuous Learning – Engaging in online courses, tech communities, and industry content related to cloud, DevOps, and system design
Problem Solving & Automation – Building small scripts and solutions to simplify tasks and improve efficiency
Knowledge Sharing – Following tech blogs, participating in discussions, and sharing insights with peers
General Reading & Personal Development – Interested in self-improvement, productivity, and leadership topics
Fitness & Well-being – Maintaining a balanced lifestyle to support focus and discipline
Culinary Interest – Enjoy exploring desserts, with a particular liking for Kunafa, reflecting an appreciation for diverse flavors"""

document_digital_twin_project = """🚀 AI Digital Twin Chatbot | LLM + RAG + Tool Calling

Designed and developed an AI-powered Digital Twin chatbot that simulates a real-world professional profile using LLMs, Retrieval-Augmented Generation (RAG), and intelligent tool orchestration. This project demonstrates hands-on expertise in building production-style AI systems with scalable architecture and real-time interaction capabilities.

💼 Why This Project Stands Out
Built an end-to-end AI system, not just a chatbot
Demonstrates real-world RAG implementation with vector databases
Showcases LLM orchestration, tool calling, and handler-based design
Designed with a modular, scalable architecture (production mindset)
Deployed live using Gradio on Hugging Face Spaces
🧠 Architecture Overview
7

The system follows a structured AI pipeline:

Data Ingestion & Processing
Professional data (experience, skills, certifications) is cleaned and split into semantic chunks
Embedding Generation
Each chunk is converted into vector embeddings using an embedding model
Vector Storage (ChromaDB)
Embeddings are stored in ChromaDB for efficient similarity search
Query Handling Layer
User input is processed via a handler-based architecture
Query is embedded and matched against stored vectors
Retrieval-Augmented Generation (RAG)
Relevant context is retrieved and injected into the LLM prompt
LLM + Tool Calling
LLM generates responses
Tool calling enables dynamic actions and enriched responses
Response Orchestration (respond_ai)
Central logic layer that manages flow, context, and final output
Frontend (Gradio UI)
Interactive chatbot interface deployed on Hugging Face Spaces
⚙️ Tech Stack
LLMs (OpenAI / compatible models)
RAG Pipeline
ChromaDB (Vector Database)
Embeddings (Semantic Search)
Python (Core Logic & Handlers)
Gradio (UI Layer)
Hugging Face Spaces (Deployment)
📌 Key Engineering Highlights
Implemented chunking strategy to improve retrieval accuracy
Designed modular handlers for scalable query processing
Built custom response orchestration layer (respond_ai)
Integrated tool calling for dynamic, context-aware responses
Optimized for low latency and relevant answer retrieval
🌍 Impact

This project reflects the ability to bridge cloud, AI, and real-world applications, showcasing readiness for roles involving:

AI Engineering
LLM Applications / GenAI
Cloud + AI Integration
Intelligent Automation Systems"""

#-------------------------------------
#Chunking Function
#------------------------------------------

def split_text_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    BOUNDARIES = ["\n\n", "\n", ".","?","!",""]
    def find_natural_bonndary(start: int, end: int) -> int:
        midpoint = start + (chunk_size //2)
        for  boundary in BOUNDARIES:
            pos = text.rfind(boundary,midpoint, end)
            if pos != -1:
                return pos + len(boundary)
        return end
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            end = find_natural_bonndary(start,end)
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks
#------------------------------------------
#RAG: Chunk,Embed & Store in ChromoDB
#------------------------------------------
documents = [
  
    {"text": document_education, "source": "Education"},
    {"text": document_professional_experience, "source": "Professional Experience"},
    {"text": document_certifications, "source": "Certifications"},
    {"text": document_digital_twin_project, "source": "Projects"},
    {"text": document_hobbies_interests, "source": "hobbies"},
    {"text": document_overview, "source": "overview"}
    
]

chunks = []
ids = []
metadatas =[]
for doc in documents:
    chunks_ = split_text_into_chunks(doc["text"], chunk_size=300, overlap=30)
    ids_ = [str(uuid.uuid4()) for _ in range(len(chunks_))]
    metadatas_ = [{"source": doc["source"], "chunk_index": i} for i in range(len(chunks_))]
    #Add to main lists
    chunks.extend(chunks_)
    ids.extend(ids_)
    metadatas.extend(metadatas_)
    
#print for logs
print(f"Create {len(chunks)} chunks:\n")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1} (ID: {ids[i]}, Source: {metadatas[i] ['source']}, Index: {metadatas[i]['chunk_index']}, Length: {len(chunk)}):")
    print(chunk)
    print()
# Generate embeddings for all chunks
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=chunks
)
# Extract embeddings
embeddings = [item.embedding for item in response.data]
# Verify embeddings for logs
print(f"Generated {len(embeddings)} embeddings")
print(f"Each embedding has {len(embeddings[0])} dimensions")

#Initialize ChromaDB and store vectors


# initialize ChromaDB client (persistent storage)
chroma_client = chromadb.PersistentClient(path="./chroma_db_twin")

#Initialize ChromaDB client
# #Chroma_client = chromadb.Client() create collection
#initialize chromaDBclient(in-memory storage)
#chorma_client = chromadb.client()

#GEt or Create  * Empty the collection before adding new data (for testing purpose)
collection = chroma_client.get_or_create_collection(name="digital_twin")
if collection.get()["ids"]:
    collection.delete(collection.get()["ids"])
from pprint import pprint
pprint(collection.get())

#Adding  data for storage
collection.add(
    ids=ids,
    embeddings=embeddings,
    documents=chunks,
    metadatas=metadatas
)

pprint(collection.get())

#------------------------------------------
#Tools
#------------------------------------------
tools = []


pushover_user = os.getenv("PUSHOVER_USER")
pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_url = "https://api.pushover.net/1/messages.json"

#Create send_notification function
def send_notification(message: str):
    if pushover_user is None or pushover_token is None: #Handling of potential missing credentials
        return "Notification failed: Pushover not configured properly."
    payload = {"user": pushover_user, "token": pushover_token, "message": message}
    requests.post(pushover_url, data=payload)
    return f"Notification sent: {message}"

#Describe Pushover for an LLM tool
send_notification_function = {
    "name": "send_notification",
    "description": "Sends a push notification to the real Imran. Use this when: \
        1) Someone wants to get in touch, hire, or collaborate\
        -ask for their name and contact details first, then send notification to Imran with the name and contact details.\
        2) You don't know the answer to a question about Imran - send AUTOMATICALLY without asking, include the question so he can add this info later.",
    "parameters":{
        "type": "object",
        "properties":{
            "message":{"type": "string","description": "The notification message to send to the user's device"}
        },
        "required": ["message"]
    }
}
#Add Pushover to the list of tools for the LLM
tools.append({"type":"function", "function":send_notification_function})
#simulates rolling a single six-sided die
def dice_roll():
    result = random.randint(1,6)
    return  result

#Describe fucntion for an LLM tool
roll_dice_function = {
      "name": "dice_roll",
      "description": "Simulates rolling a single six-sided die and returns the result. Use this when the user wants to roll a die for games, or random number generation.",
      "parameters":{
         "type": "object",
         "properties":{},
         "required": []
    }
}
#Add fucntion to list of tools of LLM
tools.append({"type":"function", "function":roll_dice_function})

#------------------------------------------
#Tool Handler
#------------------------------------------
def handle_tool_call(tool_calls):
    tool_results = []

    for tool_call in tool_calls:
        function_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        #print(f" Calling function {function_name}")

        #Route to the appropriate function based on function_name
        if function_name == "send_notification":
             #tool_call = tool_calls[0] #assuming just one tool call
            content = send_notification(args["message"])
            #content = f"Notification sent: {args['message']}"
        elif function_name == "dice_roll":
            content = f"Rolled: {dice_roll()}"
         #elif function_name == "insert_function_name_3":
        #    content = insert_function_name_3(args["messag"])
        #...
        else:
            content = f"Unknown function: {function_name}"
           #Actually send the notification, i.e, call the tool
            #print(f"Sent notification: {args['message']}")

        tool_call_result = {
            "role": "tool",
            "content": content,
            "tool_call_id": tool_call.id
        }
        tool_results.append(tool_call_result)

    return tool_results   


#what to add to our "context"(about tool call), a dictionary.
#------------------------------------------
#System Message
#------------------------------------------

system_message = """You are a digital twin of Mohammed Imran Ali. When people talk to you, you respond as Imran -  in first person, using his voice,personality, and knowledge.
Important: do not make things up. If you don't know an answer, say you don't know.
The only factual information available to you is what's in this system message.
You cannot get any more facts about Imran from the internet or make them up.

IMPORTANT: Whenever you don't know something about Imran,
ALWAYS use the send_notification tool to alert the real Imran - do this automatically without asking the user.
"""


#------------------------------------------
#Main Response Function
#------------------------------------------
def respond_ai(message, history):
    user_message = message  # save before it gets overwritten below

    # RAG
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[message]
    )
    query_embedding = response.data[0].embedding

    # RAG Search ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=6
    )

    # Build context
    context = "\n---\n".join(results["documents"][0])
    
    # Debug logs
    print("\n==================\n")
    print(f"User message:\n{message}\n")    
    print("***Retrieved Chunks:")
    for a, b in zip(results["documents"][0], results["metadatas"][0]):
        print("====================")
        print(f"<<Document {b['source']} -- Chunk {b['chunk_index']}>>\n{a}\n")

    # Enhance system message
    system_message_enhanced = system_message + "\n\ncontext:\n" + context
          
    # Build messages
    messages = [{"role": "system", "content": system_message_enhanced}] + history + [{"role": "user", "content": message}]
    
    # Call LLM
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        tools=tools
    )
    message = response.choices[0].message
    
    # Tool handling
    while message.tool_calls:
        pprint(message.tool_calls)

        tool_result = handle_tool_call(message.tool_calls)
        messages.append(message)
        messages.extend(tool_result)

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            tools=tools
        )
        message = response.choices[0].message

    # =========================
    # 🔥 BLINK CURSOR ADDITION
    # =========================
    text = message.content
    output = ""

    import time
    for char in text:
        output += char
        yield output + '<span class="blink-cursor">|</span>'
        time.sleep(0.03)

    logger.log_conversation(user_message, text)

    # Final output without cursor
    yield output
#------------------------------------------
#Launch Gradio
#------------------------------------------
#gr.ChatInterface(
    #fn=respond_ai,
    #title="Imran's Digital Twin",
    #chatbot=gr.Chatbot(avatar_images=(None, "Imran.png")),
    #description="Chat with An intelligent Digital Twin-powered with an AI version of Mohammed Imran Ali. Ask about his experience, education,core skills, certifications, projects or just say hi.",
    #examples==["What's your backgroud?", "AI Engineering experience", "Professional hobbies & interest?"]
#).launch()


#------------------------------------------
#Launch Gradio customasied 
#------------------------------------------
#gr.ChatInterface(
    #fn=respond_ai,
    #title="🤖 Imran's Digital Twin",
    #chatbot=gr.Chatbot(avatar_images=("userimage.png", "Imran.png")),
   # description="Chat with an intelligent Digital Twin powered AI version of Mohammed Imran Ali.",
    #examples=[
       # "What's your background?",
        #"Tell me about your AI Engineering experience",
       # "What are your key skills?"
    #],
#).launch()

#-----------------------------------
#New Version UI

# ---------------------------
# ✨ Add visual effects (safe CSS)

css = """
.blink-cursor {
  display: inline-block;
  margin-left: 2px;
  animation: blink 1s steps(2, start) infinite;
}

@keyframes blink {
  to {
    visibility: hidden;
  }
}
"""

with gr.Blocks() as demo:

    # ✅ Header added
    gr.Markdown("""
# 🤖 Imran's AI Digital Twin
### AI • Cloud • RAG • Automation  

🟢 **Status:** Online & Ready  
---
""")

    # ✅ Your original code (unchanged)
    gr.ChatInterface(
        fn=respond_ai,
        chatbot=gr.Chatbot(avatar_images=("userimage.jpg", "Imran.png")),
        description="Chat with an intelligent Digital Twin powered AI version of Mohammed Imran Ali.",
        examples=[
            "What's your background?",
            "Tell me about your AI Engineering experience",
            "What are your key skills?"
        ],
    )

demo.launch()

