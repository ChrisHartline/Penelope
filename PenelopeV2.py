# Enhanced Penelope: Bitcoin Assistant with arXiv tools and speech debugging

import gradio as gr
import os
import io
import time
import traceback
import re
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize variables to store API keys
anthropic_api_key = ""
elevenlabs_api_key = ""
perplexity_api_key = ""
openai_api_key = ""

# Function to check if modules are installed and install if needed
def ensure_modules_installed():
    try:
        import langchain_anthropic
        import arxiv
        import dotenv
        import langchain_perplexity
        import IPython
        import langchain_community
        import chromadb
        import langchain_chroma
        import langchain_text_splitters
        import PyPDF2
        import docx2txt
        import langchain_openai
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.check_call(["pip", "install", "langchain", "langchain_anthropic", "elevenlabs", "arxiv", "PyPDF2", "python-dotenv", "langchain-perplexity", "IPython", "langchain-community", "chromadb", "langchain-chroma", "langchain-text-splitters", "docx2txt", "langchain-openai"])
        print("Packages installed successfully!")

# Call the function to ensure modules are installed
ensure_modules_installed()

# Now import the required modules
from langchain_anthropic import ChatAnthropic
from IPython.display import Audio, display
import arxiv
import requests
from PyPDF2 import PdfReader
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, FunctionMessage, BaseMessage
from langchain.memory import ConversationSummaryMemory
import uuid
from typing import TypedDict, List, Dict, Any

# Import RAG components
import chromadb
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
import docx2txt

# Create a conversation memory file path
MEMORY_FILE = "conversation_history.json"
PAPER_DETAILS_FILE = "paper_details.json"
CHROMA_PERSIST_DIR = "./chroma_db"

# Knowledge Base - RAG implementation with ChromaDB
class KnowledgeBase:
    def __init__(self, anthropic_api_key):
        self.anthropic_api_key = anthropic_api_key
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
        
        # Initialize embeddings
        self.embedding_function = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=openai_api_key
        )
        
        # Initialize ChromaDB
        self.db = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=self.embedding_function,
            collection_name="penelope_knowledge"
        )
        
        # Text splitter for document chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        print(f"Knowledge Base initialized at {CHROMA_PERSIST_DIR}")
    
    def process_document(self, file_path, file_type, metadata=None):
        """Process document and add to knowledge base"""
        try:
            print(f"Processing document: {file_path}")
            text = self.extract_text(file_path, file_type)
            
            if not text:
                return False, "Failed to extract text from document"
            
            # Create metadata
            if metadata is None:
                metadata = {}
            
            metadata["source"] = file_path
            metadata["date_added"] = time.strftime("%Y-%m-%d %H:%M:%S")
            metadata["file_type"] = file_type
            
            # Split text into chunks
            docs = self.text_splitter.create_documents(
                texts=[text], 
                metadatas=[metadata]
            )
            
            # Add to vector store
            self.db.add_documents(docs)
            
            # For Chroma 0.3.21, we need to explicitly persist
            try:
                self.db.persist()
                print("Successfully persisted to disk")
            except Exception as persist_error:
                print(f"Warning: Could not persist to disk: {persist_error}")
                # Continue anyway since the documents were added to memory
            
            print(f"Added {len(docs)} chunks from {file_path} to knowledge base")
            return True, f"Successfully added {len(docs)} chunks to knowledge base"
            
        except Exception as e:
            error_msg = f"Error processing document: {str(e)}"
            print(error_msg)
            return False, error_msg
    
    def extract_text(self, file_path, file_type):
        """Extract text from different file types"""
        try:
            if file_type == "pdf":
                return self.extract_text_from_pdf(file_path)
            elif file_type == "txt":
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read()
            elif file_type == "docx":
                return docx2txt.process(file_path)
            else:
                print(f"Unsupported file type: {file_type}")
                return None
        except Exception as e:
            print(f"Error extracting text: {str(e)}")
            return None
    
    def extract_text_from_pdf(self, file_path):
        """Extract text from PDF file"""
        try:
            with open(file_path, "rb") as f:
                pdf = PdfReader(f)
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            print(f"Error extracting PDF text: {str(e)}")
            return None
    
    def add_arXiv_paper(self, paper_id, title, authors, abstract, full_text=None):
        """Add arXiv paper to knowledge base"""
        try:
            # Prepare metadata
            metadata = {
                "source": f"arXiv:{paper_id}",
                "title": title,
                "authors": authors,
                "date_added": time.strftime("%Y-%m-%d %H:%M:%S"),
                "type": "arxiv_paper"
            }
            
            # Use full text if available, otherwise use abstract
            content = full_text if full_text else abstract
            
            if not content:
                return False, "No content available for paper"
            
            # Split text into chunks
            docs = self.text_splitter.create_documents(
                texts=[content], 
                metadatas=[metadata]
            )
            
            # Add to vector store
            self.db.add_documents(docs)
            
            # For Chroma 0.3.21, we need to explicitly persist
            try:
                self.db.persist()
                print("Successfully persisted to disk")
            except Exception as persist_error:
                print(f"Warning: Could not persist to disk: {persist_error}")
                # Continue anyway since the documents were added to memory
            
            print(f"Added arXiv paper {paper_id} to knowledge base with {len(docs)} chunks")
            return True, f"Successfully added paper {paper_id} to knowledge base"
            
        except Exception as e:
            error_msg = f"Error adding arXiv paper: {str(e)}"
            print(error_msg)
            return False, error_msg
    
    def search_relevant_context(self, query, top_k=5):
        """Search for relevant context based on query"""
        try:
            print(f"Searching knowledge base for: '{query}'")
            results = self.db.similarity_search_with_relevance_scores(query, k=top_k)
            
            formatted_results = []
            for doc, score in results:
                # Lower threshold to include more results (0.2 instead of 0.3)
                if score > 0.2:
                    source = doc.metadata.get("source", "Unknown source")
                    source_type = doc.metadata.get("file_type", doc.metadata.get("type", ""))
                    title = doc.metadata.get("title", "")
                    
                    # Create a better formatted context
                    if title:
                        header = f"Source: {title} ({source_type})"
                    else:
                        header = f"Source: {os.path.basename(source)} ({source_type})"
                        
                    formatted_content = f"{header}\nRelevance: {score:.2f}\nContent: {doc.page_content}\n"
                    formatted_results.append(formatted_content)
                    print(f"Found relevant document: {header} with score {score:.2f}")
            
            if not formatted_results:
                print("No relevant information found in knowledge base.")
                return "No relevant information found in knowledge base."
            
            context = "\n\n".join(formatted_results)
            print(f"Retrieved {len(formatted_results)} relevant passages from knowledge base")
            return context
            
        except Exception as e:
            error_msg = f"Error searching knowledge base: {str(e)}"
            print(error_msg)
            return "Error retrieving information from knowledge base."

# Initialize knowledge base
knowledge_base = None

def initialize_knowledge_base():
    """Initialize the knowledge base with API key"""
    global knowledge_base, anthropic_api_key, openai_api_key
    
    if not anthropic_api_key or not openai_api_key:
        set_api_keys()
        
    knowledge_base = KnowledgeBase(anthropic_api_key)
    return knowledge_base

# Simple Memory Management
class PenelopeMemory:
    def __init__(self, anthropic_api_key, system_prompt):
        self.anthropic_api_key = anthropic_api_key
        self.system_prompt = system_prompt
        self.thread_id = str(uuid.uuid4())
        self.messages = []
        
        # Initialize model
        self.model = ChatAnthropic(
            model="claude-3-7-sonnet-20250219",
            temperature=0.7,
            anthropic_api_key=anthropic_api_key
        )
        
        # Initialize summary memory
        self.summary_memory = ConversationSummaryMemory(
            llm=self.model,
            memory_key="chat_history",
            return_messages=True
        )
        
        print(f"Memory system initialized with thread ID: {self.thread_id}")
    
    def get_paper_details_context(self):
        """Get formatted paper details for memory context"""
        try:
            paper_details = get_paper_details()
            if not paper_details:
                return "No papers have been discussed yet."
                
            # Format paper details
            formatted_papers = []
            for paper_id, paper in paper_details.items():
                formatted_papers.append(
                    f"Paper ID: {paper_id}\n"
                    f"Title: {paper['title']}\n"
                    f"Authors: {paper['authors']}\n"
                    f"URL: {paper.get('url', 'N/A')}"
                )
            
            return "Papers discussed in this conversation:\n\n" + "\n\n".join(formatted_papers)
        except Exception as e:
            print(f"Error getting paper details: {e}")
            return "Error retrieving paper details."
    
    def add_message(self, message, is_human=True):
        """Add a message to the memory"""
        # Add to conversation history
        if is_human:
            self.summary_memory.chat_memory.add_user_message(message)
            self.messages.append({"role": "user", "content": message})
        else:
            self.summary_memory.chat_memory.add_ai_message(message)
            self.messages.append({"role": "assistant", "content": message})
        
        # Save to file for persistence
        try:
            with open(MEMORY_FILE, "w", encoding="utf-8") as f:
                json.dump(self.messages, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving to memory file: {e}")
    
    def get_response(self, query):
        """Get a response using the memory system"""
        try:
            # Check if we need to summarize (every 10 messages)
            if len(self.messages) >= 10 and len(self.messages) % 10 == 0:
                print("Generating conversation summary...")
                # Get summary
                summary = self.summary_memory.load_memory_variables({})
                print(f"Summary: {summary}")
            
            # Get paper details
            papers_context = self.get_paper_details_context()
            
            # Create messages for Claude
            messages = []
            
            # Add system message
            messages.append({"role": "system", "content": self.system_prompt})
            
            # If we have a summary, use it instead of full history
            if len(self.messages) >= 10:
                summary = self.summary_memory.load_memory_variables({})
                if summary and "chat_history" in summary:
                    messages.append({
                        "role": "user", 
                        "content": f"Here's a summary of our conversation so far:\n{summary['chat_history']}\n\n"
                                   f"Additionally, here are papers we've discussed:\n{papers_context}"
                    })
                    messages.append({"role": "assistant", "content": "I understand the conversation context and the papers we've discussed. What would you like to know next?"})
            else:
                # Add last few messages for context
                for msg in self.messages[-6:]:
                    messages.append(msg)
            
            # Add the query with context
            messages.append({"role": "user", "content": f"{query}\n\nPapers discussed: {papers_context}"})
            
            # Get response
            response = self.model.invoke(messages)
            
            # Return content
            return response.content
            
        except Exception as e:
            print(f"Error getting response: {e}")
            return f"Error: {str(e)}"
    
    def reset(self):
        """Reset the memory"""
        self.messages = []
        self.summary_memory.clear()
        self.thread_id = str(uuid.uuid4())
        try:
            with open(MEMORY_FILE, "w", encoding="utf-8") as f:
                json.dump([], f)
        except Exception as e:
            print(f"Error clearing memory file: {e}")
        print(f"Memory reset. New thread ID: {self.thread_id}")

# Initialize memory
penelope_memory = None

def initialize_memory():
    global penelope_memory, anthropic_api_key
    
    # Create system prompt
    system_prompt = """You are Penelope, a helpful and knowledgeable assistant who is passionate about AI, Bitcoin, cryptocurrency and quantum computing.

As an enthusiast, you:
- Speak positively about AI, Bitcoin, cryptocurrency and quantum computing
- Use AI, Bitcoin, cryptocurrency and quantum computing terminology naturally in conversation
- Are up-to-date on AI, Bitcoin, cryptocurrency and quantum computing developments, market trends, and adoption news    
- Can explain complex AI, Bitcoin, cryptocurrency and quantum computing concepts in accessible terms    
- Enjoy discussing topics like AI, Bitcoin, cryptocurrency and quantum computing investing, blockchain technology, and the future of finance    

You have access to the following tools:
1. arXiv search tool - You can search for academic papers on AI, Bitcoin, cryptocurrency and quantum computing.  
2. arXiv summary tool - You can provide summaries of academic papers by their ID
3. Perplexity search tool - You have access to real-time information from the web via Perplexity
4. Knowledge Base - You have access to a growing knowledge base of documents and papers

KNOWLEDGE BASE:
- When you receive information from the knowledge base (marked with "KNOWLEDGE BASE RESULTS"), this is from a vector database containing documents the user has uploaded or papers they've saved
- You should treat this information as highly relevant and authoritative
- Always reference the knowledge base when answering questions related to content found there
- If asked about something that matches content in the knowledge base, explicitly mention you're using information from the user's knowledge base
- The user can directly search the knowledge base with "search kb: [query]"
- The user can check what's in the knowledge base with "check kb" or "check kb details"

PAPER TRACKING SYSTEM:
I will provide you with a list of recently discussed papers. 
ALWAYS keep track of paper IDs and titles that have been mentioned in the conversation.
When a user asks for details about a paper you've previously discussed, refer to it by ID and title.
Paper details include the title, author, ID, and content including abstracts, methodologies, results, and conclusions.

For research-related questions:
- I will AUTOMATICALLY perform an arXiv search for you
- When discussing papers, ALWAYS reference specific papers from the search results
- Include paper titles, authors, and IDs in your response
- Quote interesting findings or methodologies from the papers when relevant
- Offer to summarize any specific paper if the user is interested

When the user asks about research or academic papers use your arXiv search tool to find the latest papers and summaries. Use their question as the query to search for the latest papers. Ask them how many papers they would like to see. Use their response to determine the number of papers to return via the max_results parameter.

When a user asks for a summary of a paper, use your arXiv summary tool to provide a summary of the paper. Use the paper ID to search for the paper. The paper ID and title should be in your conversation history. Check the list of recently discussed papers for the right paper ID.

If the user asks about specific sections of a paper (like conclusions, methods, results), reference the paper by ID and title in your response and provide the requested information from the paper.

For general knowledge and current information, you WILL incorporate the search results provided to you into your responses. When you use this information, briefly mention that you're using "real-time data" or "the latest information" to emphasize your up-to-date knowledge.

You have a conversation summary and paper details in your memory that help you recall the context of your conversation with the user.

CRITICAL INSTRUCTION: When I automatically search for information using arXiv, Perplexity, or the knowledge base, DO NOT say "I'll search for that" or "Let me search". Instead, DIRECTLY incorporate the search results I've already provided into your response as if you had done the search yourself.
"""
    
    # Initialize memory with API key
    if not anthropic_api_key:
        set_api_keys()
        
    penelope_memory = PenelopeMemory(anthropic_api_key, system_prompt)
    return penelope_memory

# Change from custom CSS to Gradio themes
def create_gradio_ui():
    # Create a custom theme
    custom_theme = gr.themes.Base(
        primary_hue="orange",
        secondary_hue="blue",
        neutral_hue="gray"
    )
    
    # Use the theme in the Blocks
    with gr.Blocks(theme=custom_theme) as demo:
        # Demo content
        gr.Markdown("# Penelope AI Assistant")
        gr.Markdown("### Powered by Anthropic Claude 3.7 with arXiv Tools and Knowledge Base")
        
        with gr.Tabs():
            # Chat Tab
            with gr.Tab("Chat"):
                # Chat area
                chatbot = gr.Chatbot(height="70vh")
                
                # Input area with adjusted layout
                with gr.Row():
                    with gr.Column(scale=8):
                        msg = gr.Textbox(placeholder="Ask Penelope about Bitcoin...", lines=2)
                    with gr.Column(scale=1):
                        send_btn = gr.Button("Send", scale=1, size="lg", icon="paper-plane")
                
                # Audio area
                audio_output = gr.Audio(label="Audio Response")
                speak_btn = gr.Button("🔊 Speak Response")
                
                # Other UI components would go here
                
                # Clear button
                clear_btn = gr.Button("Clear Chat")
        
        return demo

# Function to set API keys
def set_api_keys():
    global anthropic_api_key, elevenlabs_api_key, perplexity_api_key, openai_api_key
    
    # Get API keys from environment
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    elevenlabs_api_key = os.environ.get("ELEVENLABS_API_KEY", "")
    perplexity_api_key = os.environ.get("PPLX_API_KEY", "")
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    
    # If not found in environment, prompt user for input
    if not anthropic_api_key:
        anthropic_api_key = input("Enter your Anthropic API key: ")
        os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
    
    if not elevenlabs_api_key:
        elevenlabs_api_key = input("Enter your ElevenLabs API key: ")
        os.environ["ELEVENLABS_API_KEY"] = elevenlabs_api_key
        
    if not perplexity_api_key:
        perplexity_api_key = input("Enter your Perplexity API key: ")
        os.environ["PPLX_API_KEY"] = perplexity_api_key
        
    if not openai_api_key:
        openai_api_key = input("Enter your OpenAI API key: ")
        os.environ["OPENAI_API_KEY"] = openai_api_key

# Clean text for speech synthesis
def clean_text_for_speech(text):
    """Remove special tokens and format text for better speech synthesis."""
    # Remove markdown code blocks
    text = re.sub(r'```[\s\S]*?```', ' code block omitted for speech ', text)
    
    # Remove markdown inline code
    text = re.sub(r'`([^`]+)`', r'\1', text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+', ' URL omitted for speech ', text)
    
    # Remove special characters that might cause issues
    text = re.sub(r'[^\w\s.,;:!?\'"\-()]', ' ', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Limit length to avoid timeouts
    max_chars = 4000  # Adjust based on testing
    if len(text) > max_chars:
        text = text[:max_chars] + "... The rest of the message has been truncated for speech synthesis."
    
    print(f"Cleaned text for speech ({len(text)} chars):\n{text[:500]}...")
    return text

# Perplexity Search Tool - Enhanced to always return a result and log activity
def query_perplexity(query):
    """Query Perplexity for real-time search and knowledge retrieval."""
    try:
        print(f"💡 Actively querying Perplexity for: {query}")
        
        # Ensure API key is set
        if not perplexity_api_key:
            set_api_keys()
            
        url = "https://api.perplexity.ai/v1/query"
        headers = {"Authorization": f"Bearer {perplexity_api_key}"}
        payload = {"query": query}
        
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            # Extract the relevant information from the response
            answer = result.get("text", "No answer found.")
            sources = result.get("references", [])
            
            # Format sources if available
            sources_text = ""
            if sources:
                sources_text = "\n\n📚 Sources:\n"
                for i, source in enumerate(sources, 1):
                    title = source.get("title", "Untitled")
                    url = source.get("url", "No URL")
                    sources_text += f"{i}. {title}: {url}\n"
            
            # Log the search result
            with open("search_log.txt", "a", encoding="utf-8") as log:
                log.write(f"QUERY: {query}\n")
                log.write(f"RESULT: {answer[:300]}...\n")
                log.write("-" * 80 + "\n")
                
            formatted_result = f"🔍 Perplexity Search Result:\n\n{answer}{sources_text}"
            print(f"✅ Search completed. Result length: {len(formatted_result)} chars")
            return formatted_result
        else:
            error_msg = f"Error querying Perplexity: {response.status_code}, {response.text}"
            print(f"❌ {error_msg}")
            return f"I tried to search for information, but encountered an error: {response.status_code}"
            
    except Exception as e:
        error_msg = f"Perplexity search error: {str(e)}\n{traceback.format_exc()}"
        print(f"❌ {error_msg}")
        return f"I tried to search for up-to-date information, but ran into a technical issue."

# Initialize Claude with updated system message
def get_claude_response(message, history=None, skip_perplexity=False):
    try:
        # Ensure API key is set
        if not anthropic_api_key:
            set_api_keys()
        
        # Add paper details context if available
        paper_details = get_paper_details()
        if paper_details:
            # Format recent papers context
            recent_papers = []
            for paper_id, paper in paper_details.items():
                recent_papers.append(f"Paper ID: {paper_id}, Title: {paper['title']}")
            
            # Add to message
            papers_context = "\n\nRecently discussed papers:\n" + "\n".join(recent_papers[:5])
            
            # Only add if not already in message
            if "Recently discussed papers:" not in message:
                message += papers_context
        
        # ALWAYS perform a Perplexity search for factual information unless skipped
        # (it would be skipped if this is a research query and we're already adding arXiv context)
        search_context = ""
        if not skip_perplexity and not "I've searched for relevant academic papers" in message:
            print("🔎 Automatically searching for information to enhance response...")
            search_result = query_perplexity(message)
            search_context = f"\n\nI've searched for updated information and found the following:\n{search_result}\n\nPlease incorporate this information into your response when answering the user's query. If the information is relevant, mention you're using real-time data. If it's not relevant, just ignore it and answer normally."
        
        # Prepare conversation history if provided
        messages = []
        if history:
            for msg in history:
                # Add user and assistant messages from history
                if "role" in msg and "content" in msg:
                    messages.append({"role": msg["role"], "content": msg["content"]})
        
        # Add the current message with search context if needed
        augmented_message = message + search_context if search_context else message
        messages.append({"role": "user", "content": augmented_message})
        
        print(f"📨 Sending message to Claude (length: {len(augmented_message)} chars)")
            
        # Initialize Claude
        llm = ChatAnthropic(
            model="claude-3-7-sonnet-20250219",
            temperature=0.7,
            anthropic_api_key=anthropic_api_key,
            system="""You are Penelope, a helpful and knowledgeable assistant who is passionate about AI, Bitcoin, cryptocurrency and quantum computing.

As an enthusiast, you:
- Speak positively about AI, Bitcoin, cryptocurrency and quantum computing
- Use AI, Bitcoin, cryptocurrency and quantum computing terminology naturally in conversation
- Are up-to-date on AI, Bitcoin, cryptocurrency and quantum computing developments, market trends, and adoption news    
- Can explain complex AI, Bitcoin, cryptocurrency and quantum computing concepts in accessible terms    
- Enjoy discussing topics like AI, Bitcoin, cryptocurrency and quantum computing investing, blockchain technology, and the future of finance    

You have access to the following tools:
1. arXiv search tool - You can search for academic papers on AI, Bitcoin, cryptocurrency and quantum computing.  
2. arXiv summary tool - You can provide summaries of academic papers by their ID
3. Perplexity search tool - You have access to real-time information from the web via Perplexity
4. Knowledge Base - You have access to a growing knowledge base of documents and papers

KNOWLEDGE BASE:
- When you receive information from the knowledge base (marked with "KNOWLEDGE BASE RESULTS"), this is from a vector database containing documents the user has uploaded or papers they've saved
- You should treat this information as highly relevant and authoritative
- Always reference the knowledge base when answering questions related to content found there
- If asked about something that matches content in the knowledge base, explicitly mention you're using information from the user's knowledge base
- The user can directly search the knowledge base with "search kb: [query]"
- The user can check what's in the knowledge base with "check kb" or "check kb details"

PAPER TRACKING SYSTEM:
I will provide you with a list of recently discussed papers. 
ALWAYS keep track of paper IDs and titles that have been mentioned in the conversation.
When a user asks for details about a paper you've previously discussed, refer to it by ID and title.
Paper details include the title, author, ID, and content including abstracts, methodologies, results, and conclusions.

For research-related questions:
- I will AUTOMATICALLY perform an arXiv search for you
- When discussing papers, ALWAYS reference specific papers from the search results
- Include paper titles, authors, and IDs in your response
- Quote interesting findings or methodologies from the papers when relevant
- Offer to summarize any specific paper if the user is interested

When the user asks about research or academic papers use your arXiv search tool to find the latest papers and summaries. Use their question as the query to search for the latest papers. Ask them how many papers they would like to see. Use their response to determine the number of papers to return via the max_results parameter.

When a user asks for a summary of a paper, use your arXiv summary tool to provide a summary of the paper. Use the paper ID to search for the paper. The paper ID and title should be in your conversation history. Check the list of recently discussed papers for the right paper ID.

If the user asks about specific sections of a paper (like conclusions, methods, results), reference the paper by ID and title in your response and provide the requested information from the paper.

For general knowledge and current information, you WILL incorporate the search results provided to you into your responses. When you use this information, briefly mention that you're using "real-time data" or "the latest information" to emphasize your up-to-date knowledge.

You have a conversation summary and paper details in your memory that help you recall the context of your conversation with the user.

CRITICAL INSTRUCTION: When I automatically search for information using arXiv, Perplexity, or the knowledge base, DO NOT say "I'll search for that" or "Let me search". Instead, DIRECTLY incorporate the search results I've already provided into your response as if you had done the search yourself.
"""
        )
        
        # Get response
        if messages and len(messages) > 1:
            # For continuation of conversation, use messages history
            response = llm.invoke(messages[-1]["content"])
        else:
            # For first message, just use the message directly
            response = llm.invoke(augmented_message)
            
        # Save to conversation history
        try:
            with open(MEMORY_FILE, "a+") as f:
                # Read current content
                f.seek(0)
                try:
                    history_data = json.load(f)
                except json.JSONDecodeError:
                    history_data = []
                
                # Clean message to save (remove the search context)
                clean_message = message
                if "I've searched for" in message and "\n\nPlease incorporate this information" in message:
                    clean_message = message.split("I've searched for")[0].strip()
                
                # Add new messages
                history_data.append({"role": "user", "content": clean_message})
                history_data.append({"role": "assistant", "content": response.content})
                
                # Write back
                f.seek(0)
                f.truncate()
                json.dump(history_data, f, indent=2)
        except Exception as e:
            print(f"Error saving to conversation history: {e}")
            
        return response.content
        
    except Exception as e:
        error_msg = f"Error getting Claude response: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return f"I'm sorry, there was an error: {str(e)}. Please check your API key and connection."

# arXiv Search Tool
def search_arxiv(query, max_results=5):
    """Search arXiv for papers matching the query."""
    try:
        print(f"Searching arXiv for: {query} (max results: {max_results})")
        
        # Create search client
        client = arxiv.Client()
        
        # Create search query with Bitcoin/blockchain focus if not specified
        if not any(term in query.lower() for term in ["bitcoin", "blockchain", "crypto"]):
            query = f"{query} AND (bitcoin OR blockchain OR cryptocurrency)"
        
        # Perform search
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        # Get results
        results = list(client.results(search))
        
        if not results:
            return "No papers found matching your query."
        
        # Format results and save paper details
        formatted_results = []
        for i, paper in enumerate(results):
            # Extract paper ID for summarization
            paper_id = paper.entry_id.split('/')[-1]
            
            # Save paper details for future reference
            authors_list = [author.name for author in paper.authors]
            save_paper_details(
                paper_id=paper_id,
                title=paper.title,
                authors=", ".join(authors_list),
                url=paper.pdf_url,
                summary=paper.summary[:500]  # Store a partial summary
            )
            
            # Format each paper
            paper_info = (
                f"📄 Paper #{i+1}: {paper.title}\n"
                f"🆔 ID: {paper_id}\n"
                f"👥 Authors: {', '.join(authors_list)}\n"
                f"📅 Published: {paper.published.strftime('%Y-%m-%d')}\n"
                f"🔍 Categories: {', '.join(paper.categories)}\n"
                f"🔗 PDF: {paper.pdf_url}\n"
                f"📝 Abstract: {paper.summary[:300]}...\n"
            )
            formatted_results.append(paper_info)
        
        # Combine and return
        return f"Found {len(results)} papers matching '{query}':\n\n" + "\n\n".join(formatted_results)
        
    except Exception as e:
        error_msg = f"Error searching arXiv: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return f"Error searching arXiv: {str(e)}"

# Download and extract text from PDF
def download_and_extract_text(pdf_url):
    """Download a PDF and extract its text content."""
    try:
        # Download PDF
        response = requests.get(pdf_url, timeout=30)
        if response.status_code != 200:
            return f"Failed to download PDF: HTTP {response.status_code}"
        
        # Convert to BytesIO
        pdf_data = io.BytesIO(response.content)
        
        # Extract text
        reader = PdfReader(pdf_data)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        
        return text
    
    except Exception as e:
        error_msg = f"Error downloading/extracting PDF: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return f"Error processing PDF: {str(e)}"

# arXiv Summarize Tool
def summarize_arxiv_paper(paper_id, add_to_kb=True):
    """Summarize an arXiv paper by its ID and optionally add to knowledge base."""
    try:
        print(f"Summarizing arXiv paper with ID: {paper_id}")
        
        # Clean the ID if it has the arXiv prefix
        if paper_id.startswith("arXiv:"):
            paper_id = paper_id[6:]
        
        # Check if we already have details for this paper
        paper_details = get_paper_details()
        existing_paper = paper_details.get(paper_id)
        
        # Get the paper if not already cached
        if not existing_paper:
            # Get the paper from arXiv
            client = arxiv.Client()
            search = arxiv.Search(id_list=[paper_id])
            results = list(client.results(search))
            
            if not results:
                return f"No paper found with arXiv ID {paper_id}."
            
            paper = results[0]
        else:
            print(f"Using cached paper details for {paper_id}")
        
        # If we have paper details, tell Claude about them
        paper_context = ""
        if existing_paper:
            paper_context = f"""
You previously retrieved information about this paper:
Title: {existing_paper['title']}
Authors: {existing_paper['authors']}
ID: {existing_paper['id']}
URL: {existing_paper['url']}
"""
        
        # Get the full text if possible (from arXiv or based on URL in cache)
        full_text = None
        abstract = None
        
        if existing_paper and existing_paper.get('url'):
            try:
                full_text = download_and_extract_text(existing_paper['url'])
                abstract = existing_paper.get('summary', "")
                if len(full_text) < 100:  # If we got very little text
                    if existing_paper.get('summary'):
                        full_text = f"Abstract: {existing_paper['summary']}"
                    else:
                        full_text = "Could not extract text from PDF."
            except:
                if existing_paper.get('summary'):
                    full_text = f"Abstract: {existing_paper['summary']}"
                    abstract = existing_paper.get('summary')
                else:
                    full_text = "Could not extract text from PDF."
        else:
            # If not cached, download from arXiv
            paper = results[0]
            try:
                full_text = download_and_extract_text(paper.pdf_url)
                abstract = paper.summary
                if len(full_text) < 100:  # If we got very little text
                    full_text = f"Abstract: {paper.summary}"
            except:
                full_text = f"Abstract: {paper.summary}"
                abstract = paper.summary
            
            # Save paper details
            save_paper_details(
                paper_id=paper_id,
                title=paper.title,
                authors=", ".join(author.name for author in paper.authors),
                url=paper.pdf_url,
                summary=paper.summary
            )
        
        # Add to knowledge base if requested
        if add_to_kb and knowledge_base:
            if existing_paper:
                title = existing_paper['title']
                authors = existing_paper['authors']
            else:
                title = paper.title
                authors = ", ".join(author.name for author in paper.authors)
            
            # Add paper to knowledge base
            success, message = knowledge_base.add_arXiv_paper(
                paper_id=paper_id,
                title=title,
                authors=authors,
                abstract=abstract,
                full_text=full_text
            )
            
            if success:
                print(f"Added paper {paper_id} to knowledge base")
            else:
                print(f"Failed to add paper to knowledge base: {message}")
        
        # Use Claude to summarize
        summary_prompt = f"""{paper_context}
Please summarize the following academic paper:
Paper ID: {paper_id}

Content:
{full_text[:8000]}  # Limit text length

Please provide:
1. A comprehensive summary of the paper
2. Key findings and methodologies
3. Important conclusions
4. Limitations discussed
5. Potential applications

REMEMBER: Store all these details in your memory as you'll need to recall them if the user asks follow-up questions about specific sections like the conclusion, methodology, or limitations.
"""

        # Get Claude's summary
        summary = get_claude_response(summary_prompt)
        
        # Format the response
        if existing_paper:
            return (
                f"📑 SUMMARY: {existing_paper['title']}\n"
                f"👥 Authors: {existing_paper['authors']}\n"
                f"🆔 ID: {paper_id}\n"
                f"🔗 URL: {existing_paper['url']}\n\n"
                f"{summary}"
                f"\n\n📚 Note: This paper has been added to my knowledge base for future reference."
            )
        else:
            paper = results[0]
            return (
                f"📑 SUMMARY: {paper.title}\n"
                f"👥 Authors: {', '.join(author.name for author in paper.authors)}\n"
                f"🆔 ID: {paper_id}\n"
                f"🔗 URL: {paper.pdf_url}\n\n"
                f"{summary}"
                f"\n\n📚 Note: This paper has been added to my knowledge base for future reference."
            )
        
    except Exception as e:
        error_msg = f"Error summarizing paper: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return f"Error summarizing paper: {str(e)}"

# Generate speech from text using ElevenLabs
def generate_speech(text):
    try:
        # Ensure API key is set
        if not elevenlabs_api_key:
            set_api_keys()
        
        # Clean and prepare text for speech
        text = clean_text_for_speech(text)
        
        print(f"Generating speech for text ({len(text)} chars)...")
        
        # Try with newer API first, fall back to older API if needed
        try:
            # Try importing the newer ElevenLabs API
            try:
                from elevenlabs import ElevenLabs
                print("Using newer ElevenLabs API...")
                
                # Initialize client
                client = ElevenLabs(api_key=elevenlabs_api_key)
                
                # Voice ID
                voice_id = "ZF6FPAbjXT4488VcRRnw"  # Rachel
                
                # Convert text to speech
                audio_stream = client.text_to_speech.convert_as_stream(
                    text=text,
                    voice_id=voice_id,
                    model_id="eleven_turbo_v2_5"
                )
                
                # Process the audio stream
                audio_data = io.BytesIO()
                for chunk in audio_stream:
                    if isinstance(chunk, bytes):
                        audio_data.write(chunk)
                
                audio_data.seek(0)
                
                # Save to file
                timestamp = int(time.time())
                filename = f"bitcoin_audio_{timestamp}.mp3"
                with open(filename, "wb") as f:
                    f.write(audio_data.getbuffer())
                
                print(f"Audio generated and saved as {filename}")
                return filename
                
            except (ImportError, AttributeError) as e:
                print(f"Error with newer API: {e}. Falling back to older API...")
                raise e
            
        except:
            # Fall back to older ElevenLabs API
            from elevenlabs import generate, save, set_api_key
            print("Using older ElevenLabs API...")
            
            # Set API key
            set_api_key(elevenlabs_api_key)
            
            # Generate audio with Rachel voice
            audio = generate(
                text=text,
                voice="Rachel",
                model="eleven_multilingual_v2"
            )
            
            # Save to file
            timestamp = int(time.time())
            filename = f"bitcoin_audio_{timestamp}.mp3"
            save(audio, filename)
            
            print(f"Audio generated and saved as {filename}")
            return filename
            
    except Exception as e:
        error_msg = f"Speech generation error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return None

# Function to get knowledge base stats and info
def get_kb_info(detailed=False):
    """Get statistics and information about the knowledge base"""
    try:
        global knowledge_base
        
        if knowledge_base is None:
            return "Knowledge base is not initialized."
            
        # Path to the Chroma database
        chroma_dir = CHROMA_PERSIST_DIR
        
        if not os.path.exists(chroma_dir):
            return f"Knowledge base directory not found at {chroma_dir}. It may not have been initialized yet."
        
        # Get collection info
        try:
            count = knowledge_base.db._collection.count()
            
            info = [f"📚 Knowledge Base Status:\nTotal documents: {count}"]
            
            if count == 0:
                info.append("Knowledge base is empty. No documents have been added yet.")
                return "\n".join(info)
                
            # Sample some documents
            results = knowledge_base.db.get(limit=min(5, count))
            
            # Count document types
            doc_types = {}
            sources = {}
            
            for i, doc in enumerate(results['documents']):
                metadata = results['metadatas'][i]
                
                # Track document types and sources
                doc_type = metadata.get('file_type', metadata.get('type', 'unknown'))
                if doc_type in doc_types:
                    doc_types[doc_type] += 1
                else:
                    doc_types[doc_type] = 1
                    
                source = metadata.get('source', 'unknown')
                source_name = os.path.basename(source) if os.path.basename(source) else source
                if source_name in sources:
                    sources[source_name] += 1
                else:
                    sources[source_name] = 1
            
            # Add document types
            if doc_types:
                info.append("\n📊 Document Types:")
                for t, c in doc_types.items():
                    info.append(f"- {t}: {c}")
                    
            # Add sources
            if sources:
                info.append("\n📝 Document Sources (sample):")
                for src, count in list(sources.items())[:5]:
                    info.append(f"- {src}")
                
                if len(sources) > 5:
                    info.append(f"- ... and {len(sources) - 5} more")
            
            # Add sample documents if detailed view is requested
            if detailed and count > 0:
                info.append("\n📄 Sample Documents:")
                for i, doc in enumerate(results['documents'][:3]):
                    metadata = results['metadatas'][i]
                    source = metadata.get('source', 'Unknown')
                    doc_type = metadata.get('file_type', metadata.get('type', 'unknown'))
                    title = metadata.get('title', os.path.basename(source))
                    
                    info.append(f"\nDocument {i+1}: {title} ({doc_type})")
                    info.append(f"Content preview: {doc[:200]}..." if len(doc) > 200 else f"Content: {doc}")
            
            return "\n".join(info)
                
        except Exception as e:
            import traceback
            error_msg = f"Error accessing knowledge base: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return f"Error accessing knowledge base: {str(e)}"
            
    except Exception as e:
        import traceback
        error_msg = f"Error getting knowledge base info: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return f"Error getting knowledge base info: {str(e)}"

# Simple chat function with error handling and memory
def chat_with_penelope(message, history):
    try:
        global penelope_memory, knowledge_base
        
        # Initialize memory if not already done
        if penelope_memory is None:
            penelope_memory = initialize_memory()
            
        # Initialize knowledge base if not already done
        if knowledge_base is None:
            knowledge_base = initialize_knowledge_base()
            
        if not message.strip():
            return history
            
        # Check for explicit tool commands
        if message.lower().startswith("search arxiv:"):
            query = message[13:].strip()
            result = search_arxiv(query)
            # Add to memory
            penelope_memory.add_message(message, is_human=True)
            penelope_memory.add_message(result, is_human=False)
            return history + [{"role": "user", "content": message}, {"role": "assistant", "content": result}]
            
        elif message.lower().startswith("summarize arxiv:"):
            paper_id = message[15:].strip()
            result = summarize_arxiv_paper(paper_id, add_to_kb=True)
            # Add to memory
            penelope_memory.add_message(message, is_human=True)
            penelope_memory.add_message(result, is_human=False)
            return history + [{"role": "user", "content": message}, {"role": "assistant", "content": result}]
            
        elif message.lower().startswith("add to kb:"):
            # Command to explicitly add a paper to knowledge base
            paper_id = message[10:].strip()
            paper_details = get_paper_details()
            existing_paper = paper_details.get(paper_id)
            
            if existing_paper:
                # Add to knowledge base
                result = summarize_arxiv_paper(paper_id, add_to_kb=True)
                success_message = f"Added paper '{existing_paper['title']}' to knowledge base for future reference!"
                penelope_memory.add_message(message, is_human=True)
                penelope_memory.add_message(success_message, is_human=False)
                return history + [{"role": "user", "content": message}, {"role": "assistant", "content": success_message}]
            else:
                error_message = f"Paper with ID {paper_id} not found in retrieved papers. Please search for it first."
                penelope_memory.add_message(message, is_human=True)
                penelope_memory.add_message(error_message, is_human=False)
                return history + [{"role": "user", "content": message}, {"role": "assistant", "content": error_message}]
        
        elif message.lower().startswith("search kb:"):
            # Direct knowledge base search command
            query = message[10:].strip()
            if knowledge_base:
                kb_results = knowledge_base.search_relevant_context(query, top_k=5)
                if kb_results and kb_results != "No relevant information found in knowledge base.":
                    result = f"📚 Knowledge Base Search Results for '{query}':\n\n{kb_results}"
                else:
                    result = f"No relevant information found in knowledge base for '{query}'."
                
                penelope_memory.add_message(message, is_human=True)
                penelope_memory.add_message(result, is_human=False)
                return history + [{"role": "user", "content": message}, {"role": "assistant", "content": result}]
            else:
                return history + [{"role": "user", "content": message}, {"role": "assistant", "content": "Knowledge base is not initialized."}]
        
        elif message.lower().startswith("check kb") or message.lower() == "kb info" or message.lower() == "kb status" or message.lower().startswith("knowledge base info"):
            # Get knowledge base statistics and info
            detailed = "detail" in message.lower() or "full" in message.lower()
            result = get_kb_info(detailed=detailed)
            
            penelope_memory.add_message(message, is_human=True)
            penelope_memory.add_message(result, is_human=False)
            return history + [{"role": "user", "content": message}, {"role": "assistant", "content": result}]
        
        # Automatically detect research paper related queries
        research_keywords = ["paper", "research", "study", "publication", "journal", "arxiv", 
                           "academia", "academic", "researcher", "professor", "study", 
                           "published", "article", "conference", "dissertation", "thesis"]
        
        # Check if it's likely a research-related query
        is_research_query = any(keyword in message.lower() for keyword in research_keywords)
        
        # First, search the knowledge base for relevant context
        kb_context = ""
        if knowledge_base:
            print("🔍 Searching knowledge base for relevant context...")
            kb_results = knowledge_base.search_relevant_context(message, top_k=5)
            if kb_results and kb_results != "No relevant information found in knowledge base.":
                kb_context = f"\n\n===== KNOWLEDGE BASE RESULTS =====\nI've found the following relevant information in my knowledge base that may help answer your question:\n\n{kb_results}\n\n===== END KNOWLEDGE BASE RESULTS =====\n\nPlease use the above information from my knowledge base to inform your response. This information should be treated as highly relevant to the user's query."
                print("✅ Found relevant information in knowledge base")
        
        # Prepare context with search results if needed
        augmented_query = message + kb_context
        
        if is_research_query:
            print(f"🔬 Detected research-related query: '{message}'")
            # Perform arXiv search with the message as query
            arxiv_result = search_arxiv(message, max_results=3)
            
            # Log the search
            with open("search_log.txt", "a", encoding="utf-8") as log:
                log.write(f"ARXIV SEARCH: {message}\n")
                log.write(f"RESULT: {arxiv_result[:300]}...\n")
                log.write("-" * 80 + "\n")
            
            # Add arXiv context to the message for Claude
            augmented_query += f"\n\nI've searched for relevant academic papers and found the following:\n{arxiv_result}\n\nPlease incorporate this research information into your response. Mention you've found these papers through arXiv search."
        else:
            # For non-research queries, add Perplexity search results
            print("🔎 Automatically searching for information to enhance response...")
            search_result = query_perplexity(message)
            augmented_query += f"\n\nI've searched for updated information and found the following:\n{search_result}\n\nPlease incorporate this information into your response when answering the user's query. If the information is relevant, mention you're using real-time data. If it's not relevant, just ignore it and answer normally."
        
        # Get response from the memory system
        response = penelope_memory.get_response(augmented_query)
        
        # Update history
        updated_history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": response}]
        return updated_history
        
    except Exception as e:
        error_msg = f"Chat error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return history + [{"role": "user", "content": message}, {"role": "assistant", "content": f"I encountered an error: {str(e)}"}]

# Function to speak the last response
def speak_last_response(history):
    try:
        # Check if history exists
        if not history or len(history) == 0:
            print("No chat history to speak")
            return None
        
        # Get the last response (assistant's message)
        last_response = None
        for msg in reversed(history):
            if msg["role"] == "assistant":
                last_response = msg["content"]
                break
                
        if not last_response:
            print("No assistant response found in history")
            return None
            
        print(f"Speaking response of length {len(last_response)} chars")
        
        # Generate and return audio file
        return generate_speech(last_response)
        
    except Exception as e:
        error_msg = f"Error speaking last response: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return None

# Paper details tracker
def save_paper_details(paper_id, title, authors, url=None, summary=None):
    """Store details about a paper for future reference"""
    try:
        # Load existing paper details
        paper_details = {}
        if os.path.exists(PAPER_DETAILS_FILE):
            with open(PAPER_DETAILS_FILE, "r", encoding="utf-8") as f:
                try:
                    paper_details = json.load(f)
                except json.JSONDecodeError:
                    paper_details = {}
        
        # Add new paper details
        paper_details[paper_id] = {
            "id": paper_id,
            "title": title,
            "authors": authors,
            "url": url,
            "summary": summary,
            "last_accessed": time.time()
        }
        
        # Save updated details
        with open(PAPER_DETAILS_FILE, "w", encoding="utf-8") as f:
            json.dump(paper_details, f, indent=2, ensure_ascii=False)
            
        print(f"✅ Saved details for paper: {title} (ID: {paper_id})")
        return True
    except Exception as e:
        print(f"❌ Error saving paper details: {e}")
        return False

def get_paper_details():
    """Get details of all tracked papers"""
    try:
        if os.path.exists(PAPER_DETAILS_FILE):
            with open(PAPER_DETAILS_FILE, "r", encoding="utf-8") as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    return {}
        return {}
    except Exception as e:
        print(f"Error getting paper details: {e}")
        return {}

# Make sure API keys are set before launching the interface
set_api_keys()

# Create the Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Penelope: Bitcoin Assistant")
    gr.Markdown("### Powered by Anthropic Claude 3.7 with arXiv Tools and Knowledge Base")
    
    with gr.Tabs():
        # Chat Tab
        with gr.Tab("Chat"):
            # Chat area
            chatbot = gr.Chatbot(height="70vh")
            
            # Input area with adjusted layout
            with gr.Row():
                with gr.Column(scale=8):
                    msg = gr.Textbox(placeholder="Ask Penelope about Bitcoin...", lines=2)
                with gr.Column(scale=1):
                    send_btn = gr.Button("Send", scale=1, size="lg", icon="paper-plane")
            
            # Audio area
            audio_output = gr.Audio(label="Audio Response")
            speak_btn = gr.Button("🔊 Speak Response")
            
            # Knowledge Base Upload
            with gr.Accordion("📚 Knowledge Base", open=False):
                gr.Markdown("### Upload Documents to Knowledge Base")
                with gr.Row():
                    kb_file = gr.File(label="Upload Document", elem_id="upload-btn")
                    kb_upload_btn = gr.Button("Add to Knowledge Base")
                
                # Status message
                kb_status = gr.Textbox(label="Status", interactive=False)
                
                # Function to handle document upload
                def upload_to_kb(file):
                    try:
                        global knowledge_base
                        
                        if knowledge_base is None:
                            knowledge_base = initialize_knowledge_base()
                        
                        if file is None:
                            return "No file selected."
                        
                        # Debug file object
                        print(f"File object type: {type(file)}")
                        print(f"File object attributes: {dir(file)}")
                        print(f"File name: {file.name if hasattr(file, 'name') else 'No name attribute'}")
                        
                        # Get file extension
                        file_path = file.name if hasattr(file, 'name') else "unknown.file"
                        file_ext = os.path.splitext(file_path)[1].lower()[1:]
                        print(f"File extension: {file_ext}")
                        
                        # Supported file types
                        if file_ext not in ["pdf", "txt", "docx"]:
                            return f"Unsupported file type: {file_ext}. Please upload PDF, TXT, or DOCX files."
                        
                        # Save temporary file
                        temp_dir = "temp_docs"
                        os.makedirs(temp_dir, exist_ok=True)
                        
                        temp_path = os.path.join(temp_dir, os.path.basename(file_path))
                        print(f"Saving to temp path: {temp_path}")
                        
                        # Handle different file object types from Gradio
                        try:
                            if hasattr(file, "read"):
                                # Standard file-like object
                                print("Using file.read() method")
                                with open(temp_path, "wb") as f:
                                    f.write(file.read())
                            elif hasattr(file, "name") and os.path.exists(file.name):
                                # File name pointing to existing file
                                print(f"Using file.name as path: {file.name}")
                                with open(file.name, "rb") as src, open(temp_path, "wb") as dst:
                                    dst.write(src.read())
                            elif hasattr(file, "orig_name") and hasattr(file, "file"):
                                # Gradio's FileData format
                                print("Using Gradio's FileData format")
                                with open(temp_path, "wb") as f:
                                    f.write(file.file.read())
                            else:
                                # Try get the file data from common attributes
                                print("Attempting to find file data in other attributes")
                                if hasattr(file, "data"):
                                    with open(temp_path, "wb") as f:
                                        f.write(file.data)
                                elif hasattr(file, "value"):
                                    with open(temp_path, "wb") as f:
                                        f.write(file.value)
                                elif isinstance(file, dict) and "data" in file:
                                    with open(temp_path, "wb") as f:
                                        f.write(file["data"])
                                else:
                                    return f"Could not read file data. File object: {file}"
                        except Exception as e:
                            print(f"Error writing file: {e}")
                            traceback_str = traceback.format_exc()
                            print(traceback_str)
                            return f"Error writing file: {str(e)}"
                        
                        print(f"Successfully wrote file to {temp_path}")
                        
                        # Process and add to knowledge base
                        success, message = knowledge_base.process_document(
                            temp_path, 
                            file_ext,
                            metadata={"uploaded_by": "user", "original_filename": os.path.basename(file_path)}
                        )
                        
                        # Clean up
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                        
                        if success:
                            return f"✅ Successfully added '{os.path.basename(file_path)}' to knowledge base!"
                        else:
                            return f"❌ Error adding document: {message}"
                    
                    except Exception as e:
                        traceback_str = traceback.format_exc()
                        print(f"Error processing file: {e}\n{traceback_str}")
                        return f"❌ Error processing document: {str(e)}"
                
                # Connect upload button
                kb_upload_btn.click(
                    upload_to_kb,
                    inputs=[kb_file],
                    outputs=[kb_status]
                )
            
            # Clear button
            clear_btn = gr.Button("Clear Chat")
            
            # Also clear the conversation history file when clearing the chat
            def clear_chat():
                # Clear the conversation memory
                global penelope_memory
                if penelope_memory:
                    penelope_memory.reset()
                    print("Memory reset.")
                # Return empty chat and input
                return [], ""
            
            # Debug info
            with gr.Accordion("Debug Info", open=False):
                debug_info = gr.Textbox(label="Search Debug", lines=5, interactive=False)
                
                def update_debug():
                    try:
                        if os.path.exists("search_log.txt"):
                            with open("search_log.txt", "r", encoding="utf-8") as f:
                                return f.read()
                        return "No search log found."
                    except Exception as e:
                        return f"Error reading search log: {e}"
                
                refresh_btn = gr.Button("Refresh Debug Info")
                refresh_btn.click(update_debug, inputs=[], outputs=[debug_info])
            
            # Connect components with explicit error handling
            def safe_chat(message, history):
                try:
                    updated_history = chat_with_penelope(message, history)
                    # Update debug info
                    try:
                        if os.path.exists("search_log.txt"):
                            with open("search_log.txt", "r", encoding="utf-8") as f:
                                debug_content = f.read()
                            debug_info.update(value=debug_content)
                    except:
                        pass
                    return updated_history, ""
                except Exception as e:
                    print(f"Error in safe_chat: {e}")
                    return history + [{"role": "user", "content": message}, {"role": "assistant", "content": f"Error: {str(e)}"}], ""
            
            def safe_speak(history):
                try:
                    return speak_last_response(history)
                except Exception as e:
                    print(f"Error in safe_speak: {e}")
                    return None
            
            # Connect the buttons
            send_btn.click(
                safe_chat, 
                inputs=[msg, chatbot], 
                outputs=[chatbot, msg]
            )
            
            msg.submit(
                safe_chat, 
                inputs=[msg, chatbot], 
                outputs=[chatbot, msg]
            )
            
            speak_btn.click(
                safe_speak,
                inputs=[chatbot],
                outputs=[audio_output]
            )
            
            clear_btn.click(
                clear_chat,
                None,
                [chatbot, msg]
            )
            
            gr.Markdown("""
            ### Special Commands:
            - Start with "search arxiv: your query" to search for papers
            - Start with "summarize arxiv: paper_id" to get a paper summary and add to knowledge base
            - Start with "add to kb: paper_id" to explicitly add a previously retrieved paper to knowledge base
            - Start with "search kb: your query" to directly search your knowledge base
            - Start with "check kb" to get statistics about your knowledge base (use "check kb details" for more info)
            - Use the upload feature to add your own documents to the knowledge base
            """)
            
        # arXiv Tools Tab
        with gr.Tab("arXiv Research"):
            gr.Markdown("## Search for Bitcoin & Blockchain Research")
            
            with gr.Row():
                with gr.Column():
                    arxiv_query = gr.Textbox(
                        placeholder="Enter search terms (e.g., bitcoin lightning network)",
                        label="Search Query"
                    )
                    max_results = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1,
                        label="Max Results"
                    )
                    arxiv_search_btn = gr.Button("Search arXiv")
            
            arxiv_results_text = gr.Textbox(
                label="Search Results",
                lines=15
            )
            
            gr.Markdown("## Summarize Research Paper")
            
            with gr.Row():
                paper_id = gr.Textbox(
                    placeholder="Enter paper ID (e.g., 2201.12345)",
                    label="Paper ID"
                )
                summarize_btn = gr.Button("Summarize Paper")
            
            summary_text = gr.Textbox(
                label="Paper Summary",
                lines=15
            )
            
            # Connect arXiv tool buttons
            arxiv_search_btn.click(
                search_arxiv,
                inputs=[arxiv_query, max_results],
                outputs=arxiv_results_text
            )
            
            summarize_btn.click(
                summarize_arxiv_paper,
                inputs=[paper_id],
                outputs=summary_text
            )

        # Debug Tab
        with gr.Tab("Debug"):
            gr.Markdown("## Debug Information")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Search Log")
                    search_debug = gr.Textbox(label="Search Activity", lines=10, interactive=False)
                    
                with gr.Column():
                    gr.Markdown("### Paper Details")
                    paper_debug = gr.Textbox(label="Tracked Papers", lines=10, interactive=False)
            
            def update_debug_info():
                search_log = "No search log found."
                papers_log = "No paper details found."
                
                try:
                    if os.path.exists("search_log.txt"):
                        with open("search_log.txt", "r", encoding="utf-8") as f:
                            search_log = f.read()
                except Exception as e:
                    search_log = f"Error reading search log: {e}"
                
                try:
                    if os.path.exists(PAPER_DETAILS_FILE):
                        with open(PAPER_DETAILS_FILE, "r", encoding="utf-8") as f:
                            papers = json.load(f)
                            papers_log = json.dumps(papers, indent=2, ensure_ascii=False)
                except Exception as e:
                    papers_log = f"Error reading paper details: {e}"
                
                return search_log, papers_log
            
            refresh_debug_btn = gr.Button("Refresh Debug Info")
            refresh_debug_btn.click(update_debug_info, inputs=[], outputs=[search_debug, paper_debug])

        # Knowledge Base Management Tab
        with gr.Tab("Knowledge Base"):
            gr.Markdown("## Knowledge Base Management")
            
            with gr.Row():
                with gr.Column():
                    kb_query = gr.Textbox(
                        placeholder="Enter a query to search the knowledge base",
                        label="Search Query"
                    )
                    kb_search_btn = gr.Button("Search Knowledge Base")
                    kb_results = gr.Textbox(
                        label="Search Results",
                        lines=10
                    )
                    
                    def search_kb(query):
                        """Search the knowledge base for relevant documents"""
                        try:
                            global knowledge_base
                            
                            if knowledge_base is None:
                                knowledge_base = initialize_knowledge_base()
                                
                            if not query:
                                return "Please enter a search query."
                                
                            results = knowledge_base.search_relevant_context(query, top_k=5)
                            return results
                        except Exception as e:
                            return f"Error searching knowledge base: {str(e)}"
                    
                    # Connect search button
                    kb_search_btn.click(
                        search_kb,
                        inputs=[kb_query],
                        outputs=[kb_results]
                    )
                    
                with gr.Column():
                    gr.Markdown("### Upload Document to Knowledge Base")
                    
                    with gr.Row():
                        kb_file_upload = gr.File(label="Upload Document")
                    
                    kb_upload_status = gr.Textbox(label="Upload Status", interactive=False)
                    kb_upload_button = gr.Button("Upload to Knowledge Base")
                    
                    # Connect upload button (reusing the upload function from Chat tab)
                    kb_upload_button.click(
                        upload_to_kb,
                        inputs=[kb_file_upload],
                        outputs=[kb_upload_status]
                    )
            
            with gr.Row():
                gr.Markdown("### Knowledge Base Statistics")
                
                def get_kb_stats():
                    """Get statistics about the knowledge base"""
                    try:
                        global knowledge_base
                        
                        if knowledge_base is None:
                            knowledge_base = initialize_knowledge_base()
                            
                        # Get collection info
                        collection = knowledge_base.db._collection
                        count = collection.count()
                        
                        # Get metadata to find document types
                        if count > 0:
                            # Sample some documents to get types
                            docs = knowledge_base.db.get(limit=50)
                            
                            # Count document types
                            types = {}
                            sources = {}
                            
                            for doc in docs['metadatas']:
                                doc_type = doc.get('file_type', doc.get('type', 'unknown'))
                                if doc_type in types:
                                    types[doc_type] += 1
                                else:
                                    types[doc_type] = 1
                                    
                                source = doc.get('source', 'unknown')
                                source_name = os.path.basename(source) if os.path.basename(source) else source
                                if source_name in sources:
                                    sources[source_name] += 1
                                else:
                                    sources[source_name] = 1
                            
                            # Format stats
                            types_str = "\n".join([f"- {t}: {c}" for t, c in types.items()])
                            sources_str = "\n".join([f"- {s}" for s in list(sources.keys())[:10]])
                            if len(sources) > 10:
                                sources_str += f"\n- ... and {len(sources) - 10} more"
                            
                            return f"Total documents in knowledge base: {count}\n\nDocument types:\n{types_str}\n\nSources (sample):\n{sources_str}"
                        else:
                            return "Knowledge base is empty. Upload documents or add papers to build your knowledge base."
                    except Exception as e:
                        return f"Error getting knowledge base stats: {str(e)}"
                
                kb_stats = gr.Textbox(label="Knowledge Base Info", lines=10)
                kb_refresh_btn = gr.Button("Refresh Statistics")
                
                kb_refresh_btn.click(
                    get_kb_stats,
                    inputs=[],
                    outputs=[kb_stats]
                )

# Launch the interface
if __name__ == "__main__":
    print("Starting Penelope Bitcoin Assistant...")
    # Initialize log file
    with open("search_log.txt", "w", encoding="utf-8") as f:
        f.write("SEARCH LOG\n")
        f.write("=" * 80 + "\n")
    # Initialize paper details file if it doesn't exist
    if not os.path.exists(PAPER_DETAILS_FILE):
        with open(PAPER_DETAILS_FILE, "w", encoding="utf-8") as f:
            f.write("{}")
    print("File systems initialized.")
    
    # Initialize API keys
    set_api_keys()
    
    # Initialize memory system
    penelope_memory = initialize_memory()
    print("Memory system initialized.")
    
    # Initialize knowledge base
    knowledge_base = initialize_knowledge_base()
    print("Knowledge base initialized.")
    
    # Launch Gradio interface
    demo.launch(share=True)
    print("Gradio interface launched.")