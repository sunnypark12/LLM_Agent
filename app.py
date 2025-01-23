# streamlit_app.py
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import streamlit as st
import os
import numpy as np
import faiss
import whisper
from langchain_community.vectorstores import FAISS

## VOICE CHAT
from io import BytesIO
from streamlit_mic_recorder import speech_to_text
import azure.cognitiveservices.speech as speechsdk

## CHANGE TO CLAUDEAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic

## VIDEO READER
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import MarkdownTextSplitter
from pymupdf4llm import to_markdown  # Importing for Markdown export from PDF
import fitz 

from transformers import BlipProcessor, BlipForConditionalGeneration

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from langchain.chains import ConversationalRetrievalChain
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.prompts import PromptTemplate

from PyPDF2 import PdfReader
import pytesseract
from PIL import Image
import requests
from bs4 import BeautifulSoup
import tempfile
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
import yt_dlp
from pytube import YouTube
import re
from newspaper import Article

import requests
import tempfile

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = openai_api_key

@st.cache_resource
def load_embeddings_model():
    return HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

embeddings_model = load_embeddings_model()

@st.cache_resource
def load_whisper_model():
    try:
        return whisper.load_model("base")
    except Exception as e:
        st.error(f"Error loading Whisper model: {e}")
        return None

model = load_whisper_model()
if model is None:
    st.stop()

@st.cache_resource
def load_image_captioning_model():
    try:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        model_blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
        return processor, model_blip
    except Exception as e:
        st.error(f"Error loading BLIP model: {e}")
        return None, None

processor, model_blip = load_image_captioning_model()
if processor is None or model_blip is None:
    st.stop()

# Azure Speech TTS Setup
VOICE_OPTIONS = {
    "Jenny (US)": "en-US-JennyNeural",
    "Guy (US)": "en-US-GuyNeural",
    "Libby (UK)": "en-GB-LibbyNeural",
    "William (AU)": "en-AU-WilliamNeural",
}

def setup_speech_synthesizer(voice_name):
    speech_key = os.getenv("AZURE_SPEECH_KEY")
    speech_region = os.getenv("AZURE_SPEECH_REGION")
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
    speech_config.speech_synthesis_voice_name = voice_name
    audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
    return synthesizer

def text_to_speech_azure(text, synthesizer):
    result = synthesizer.speak_text_async(text).get()
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        audio_data = result.audio_data
        audio = BytesIO(audio_data)
        audio.seek(0)
        return audio
    else:
        st.error("Speech synthesis failed.")
        return None

def transcribe_audio():
    st.write("Please record your question in English:")
    transcribed_text = speech_to_text(
        language='en',
        use_container_width=True,
        just_once=True,
        key=f"STT-{st.session_state.get('recording_key', 0)}"
    )
    if transcribed_text:
        return transcribed_text
    else:
        st.warning("No input detected. Please try recording your question again.")
        return None

def generate_image_caption(image_path):
    raw_image = Image.open(image_path).convert('RGB')
    inputs = processor(raw_image, return_tensors="pt")
    out = model_blip.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def extract_pdf_to_markdown(file_path):
    try:
        md_text = to_markdown(file_path)
        return md_text
    except Exception as e:
        st.write(f"Error extracting Markdown from PDF: {e}")
        return None

def extract_images_from_pdf(file_path):
    images = []
    doc = fitz.open(file_path)
    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image.get("ext", "png")
            image_filename = f"page{page_num}_img{img_index}.{image_ext}"
            with open(image_filename, "wb") as image_file:
                image_file.write(image_bytes)
            images.append(image_filename)
    return images

def extract_text_from_image(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text

def render_pdf_pages_to_images(file_path):
    doc = fitz.open(file_path)
    images = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap()
        image_filename = f"page_{page_num}.png"
        pix.save(image_filename)
        images.append(image_filename)
    return images

def process_folder(folder_path, llm):
    docs = []
    st.write(f"Processing folder: {folder_path}")

    if not os.path.exists(folder_path):
        st.write(f"Folder path does not exist: {folder_path}")
        return docs

    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if file_name.startswith("tmp"):
                st.write(f"Skipping temporary file: {file_name}")
                continue

            file_path = os.path.join(root, file_name)
            file_ext = file_name.split('.')[-1].lower()
            st.write(f"Processing file: {file_path}")

            if file_ext == 'pdf':
                media_type = 'pdf'
            elif file_ext in ['mp4', 'mov', 'avi', 'mkv']:
                media_type = 'video'
            elif file_ext == 'txt':
                media_type = 'text'
            elif file_ext in ['jpg', 'jpeg', 'png']:
                media_type = 'image'
            else:
                st.write(f"Unsupported file type: {file_ext}")
                continue

            relative_path = os.path.relpath(root, folder_path)
            subfolder_name = relative_path.split(os.sep)[0] if relative_path != '.' else 'root'

            file_docs = process_and_store_data(file_path, media_type, subfolder_name, llm)
            if file_docs:
                docs.extend(file_docs)
            else:
                st.write(f"No documents were processed from file: {file_path}")
    return docs

def extract_folder_name(user_input):
    patterns = [
        r'\bonly\s+(?:refer to|use|from)\s+(\w+)',
        r'\b(?:in terms of|regarding|with respect to|about)\s+(\w+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, user_input, re.IGNORECASE)
        if match:
            return match.group(1)
    return None

def process_and_store_data(file_path, media_type, subfolder_name='', llm=None):
    file_name = os.path.basename(file_path)
    if media_type == 'pdf':
        md_text = extract_pdf_to_markdown(file_path)
        if md_text is None:
            return None

        image_paths = extract_images_from_pdf(file_path)
        captions = []
        for image_path in image_paths:
            caption = generate_image_caption(image_path)
            ocr_text = extract_text_from_image(image_path)
            combined_text = f"Image caption: {caption}\nExtracted text: {ocr_text}"
            captions.append(combined_text)
            os.remove(image_path)

        if captions:
            captions_text = "\n\n".join(captions)
            md_text += "\n\n" + captions_text

        tags = []
        splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = splitter.create_documents([md_text])

        docs = [Document(page_content=t.page_content, metadata={
            "source": file_name,
            "folder": subfolder_name,
            "tags": tags
        }) for t in texts]
        return docs

    elif media_type == 'text':
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        urls = extract_url_from_text(text)
        docs = []
        if urls:
            st.write(f"URLs found in the text: {urls}")
            for url in urls:
                url_docs = process_and_store_data(url, 'link')
                if url_docs:
                    docs.extend(url_docs)
        else:
            tags = []
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            texts = text_splitter.split_text(text)
            docs = [Document(page_content=t, metadata={
                "source": file_name,
                "folder": subfolder_name,
                "tags": tags
            }) for t in texts]
        return docs

    elif media_type == 'link':
        url = file_path  
        if url.lower().endswith('.pdf'):
            try:
                st.write(f"Downloading PDF from URL: {url}")
                pdf_response = requests.get(url, stream=True)
                if pdf_response.status_code == 200:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf_file:
                        for chunk in pdf_response.iter_content(chunk_size=8192):
                            tmp_pdf_file.write(chunk)
                    tmp_pdf_file_path = tmp_pdf_file.name
                    docs = process_and_store_data(tmp_pdf_file_path, 'pdf') 
                    os.unlink(tmp_pdf_file_path) 
                    return docs
                else:
                    st.write(f"Failed to download the PDF. Status code: {pdf_response.status_code}")
                    return None
            except Exception as e:
                st.write(f"Error downloading PDF from the URL: {e}")
                return None

        elif "youtube.com" in url or "youtu.be" in url:
            video_id = extract_video_id(url)
            if video_id:
                st.write(f"Processing YouTube video: {url}")
                transcript = get_youtube_transcript(video_id, url)
                if transcript:
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                    texts = text_splitter.split_text(transcript)
                    tags = []
                    docs = [Document(page_content=t, metadata={"source": url, "folder": 'YouTube', "tags": tags}) for t in texts]
                    return docs
            else:
                st.write("Invalid YouTube URL or video ID.")
                return None
        else:
            try:
                st.write(f"Processing article or document URL: {url}")
                article = Article(url)
                article.download()
                article.parse()
                text = article.text
                if not text.strip():
                    st.write("No text extracted from the URL content.")
                    return None
                tags = []
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                texts = text_splitter.split_text(text)
                docs = [Document(page_content=t, metadata={"source": url, "folder": 'Web', "tags": tags}) for t in texts]
                return docs
            except Exception as e:
                st.write(f"Error processing URL: {e}")
                return None

    elif media_type == 'video':
        text = transcribe_video(file_path)
        tags = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_text(text)
        docs = [Document(page_content=t, metadata={
            "source": file_name,
            "folder": subfolder_name,
            "tags": tags
        }) for t in texts]
        return docs

    elif media_type == 'image':
        image = Image.open(file_path)
        text_ocr = pytesseract.image_to_string(image)
        caption = generate_image_caption(file_path)
        text = text_ocr + "\n\n" + caption
        if not text.strip():
            st.write("No text or captions extracted from the image file.")
            return None
        tags = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_text(text)
        docs = [Document(page_content=t, metadata={
            "source": file_name,
            "folder": subfolder_name,
            "tags": tags
        }) for t in texts]
        return docs

    else:
        st.write("Unsupported media type for processing.")
        return None

def extract_video_id(url):
    import re
    pattern = r'(?:v=|\/)([0-9A-Za-z_-]{11}).*'
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    else:
        st.write("Could not extract video ID from the URL.")
        return None

def get_youtube_transcript(video_id, video_url):
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = ' '.join([d['text'] for d in transcript_list])
        return transcript
    except Exception as e:
        st.write(f"Could not retrieve transcript using youtube-transcript-api: {e}")
        st.write("Attempting to fetch video title and description...")
        try:
            yt = YouTube(video_url)
            title = yt.title
            description = yt.description
            text = f"Title: {title}\nDescription: {description}"
            return text
        except Exception as e:
            st.write(f"Failed to fetch video details: {e}")
            return ""

def transcribe_video(video_file_path):
    try:
        result = model.transcribe(video_file_path)
        transcribed_text = result['text']
        return transcribed_text
    except Exception as e:
        st.error(f"Error during transcription: {e}")
        return ""

def get_vector_store():
    return st.session_state.get('vector_store', None)

def save_vector_store(vector_store):
    st.session_state['vector_store'] = vector_store

def transcribe_video_tool(video_url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': '%(id)s.%(ext)s',
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(video_url, download=True)
        video_file = ydl.prepare_filename(info_dict)

    model = whisper.load_model("base")
    result = model.transcribe(video_file)
    transcript = result["text"]
    os.remove(video_file)
    return transcript

def extract_url_from_text(text):
    url_pattern = r'(https?://[^\s]+)'
    urls = re.findall(url_pattern, text)
    return urls

def find_related_youtube_links(query, context='', num_links=3):
    search_query = f"{query} {context}"
    search_url = f"https://www.youtube.com/results?search_query={search_query}"
    response = requests.get(search_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    links = []
    for a in soup.select('a'):
        href = a.get('href')
        if href and '/watch?v=' in href:
            full_url = 'https://www.youtube.com' + href
            if full_url not in links:
                links.append(full_url)
            if len(links) >= num_links:
                break
    return links

def main():
    # Add custom CSS for chat bubbles
    st.markdown("""
    <style>
    .user-bubble {
        background-color: #DCF8C6;
        color: #000;
        padding: 10px 15px;
        border-radius: 20px;
        max-width: 60%;
        margin: 10px 0;
        text-align: left;
        display: inline-block;
    }
    .assistant-bubble {
        background-color: #E8E8E8;
        color: #000;
        padding: 10px 15px;
        border-radius: 20px;
        max-width: 60%;
        margin: 10px 0;
        text-align: left;
        display: inline-block;
    }
    .user-row {
        display: flex;
        justify-content: flex-end;
    }
    .assistant-row {
        display: flex;
        justify-content: flex-start;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("Master RAG Agent Chatbot")
    st.write("Chat with your data and get holistic responses along with related YouTube links.")

    # Initialize session state variables
    if 'uploaded_documents' not in st.session_state:
        st.session_state['uploaded_documents'] = []
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []
    if 'vector_store' not in st.session_state:
        st.session_state['vector_store'] = None
    if 'recording_key' not in st.session_state:
        st.session_state['recording_key'] = 0
    if 'flags' not in st.session_state:
        st.session_state['flags'] = []  # Initialize flags as empty list

    # Initialize LLM 
    llm = ChatAnthropic(api_key=openai_api_key, model="claude-3-5-sonnet-20240620", temperature=0)

    # Sidebar: Flag Management
    st.sidebar.header("Flags Management")

    st.sidebar.subheader("Create New Flag")
    with st.sidebar.form(key='create_flag_form', clear_on_submit=True):
        new_flag_name = st.text_input("Flag Name", "")
        new_flag_color = st.color_picker("Flag Color", "#FF5733")
        create_flag_submit = st.form_submit_button("Create Flag")
        if create_flag_submit:
            if new_flag_name.strip() == "":
                st.sidebar.error("Flag name cannot be empty.")
            else:
                existing_flag_names = [flag['name'] for flag in st.session_state['flags']]
                if new_flag_name in existing_flag_names:
                    st.sidebar.error(f"Flag '{new_flag_name}' already exists.")
                else:
                    new_flag = {"name": new_flag_name, "color": new_flag_color}
                    st.session_state['flags'].append(new_flag)
                    st.sidebar.success(f"Flag '{new_flag_name}' created.")

    st.sidebar.subheader("Filter Documents by Flag")
    selected_flag = st.sidebar.selectbox("Select a Flag to Filter", ["All Flags"] + [flag['name'] for flag in st.session_state['flags']])

    if selected_flag != "All Flags":
        st.sidebar.markdown(f"### Documents with Flag: **{selected_flag}**")
        filtered_docs = [doc for doc in st.session_state['uploaded_documents'] if any(tag['name'] == selected_flag for tag in doc['tags'])]
        if filtered_docs:
            for doc in filtered_docs:
                st.sidebar.markdown(f"- **{doc['folder']}/{doc['name']}**")
        else:
            st.sidebar.write("No documents have this flag.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("All Flags")
    if st.session_state['flags']:
        for flag in st.session_state['flags']:
            st.sidebar.markdown(
                f"<span style='background-color:{flag['color']}; color:white; padding:3px 8px; border-radius:4px;'>{flag['name']}</span>",
                unsafe_allow_html=True
            )
    else:
        st.sidebar.write("No flags created yet.")

    st.sidebar.header("Uploaded Documents")
    if st.session_state['uploaded_documents']:
        for idx, doc in enumerate(st.session_state['uploaded_documents']):
            folder = doc.get('folder', 'root')
            name = doc.get('name', 'Unnamed')
            tags = doc.get('tags', [])
            with st.sidebar.expander(f"{folder}/{name}"):
                if tags:
                    st.write("**Flags:**")
                    for tag in tags:
                        tag_name = tag.get('name', 'Unnamed Flag')
                        tag_color = tag.get('color', '#000000')
                        st.markdown(f"<span style='background-color:{tag_color}; color:white; padding:3px 8px; border-radius:4px;'>{tag_name}</span>", unsafe_allow_html=True)
                else:
                    st.write("No flags assigned.")

                st.write("**Assign Flags:**")
                available_flags = [flag['name'] for flag in st.session_state['flags']]
                selected_flags = st.multiselect(
                    f"Select flags for {name}",
                    options=available_flags,
                    default=[tag['name'] for tag in tags],
                    key=f"select_flags_{idx}"
                )
                assign_flags = st.button(f"Assign Flags to {name}", key=f"assign_flags_{idx}")
                
                if assign_flags:
                    new_tags = []
                    for flag_name in selected_flags:
                        flag = next((f for f in st.session_state['flags'] if f['name'] == flag_name), None)
                        if flag:
                            new_tags.append({"name": flag['name'], "color": flag['color']})
                    st.session_state['uploaded_documents'][idx]['tags'] = new_tags
                    st.success(f"Flags updated for {name}.")
    else:
        st.sidebar.write("No documents uploaded yet.")

    st.header("Chat with Your Data")

    # Voice selection
    selected_voice = st.selectbox("Choose a voice", list(VOICE_OPTIONS.keys()), key='voice_selection')
    synthesizer = setup_speech_synthesizer(VOICE_OPTIONS[selected_voice])

    # Choose input method
    input_method = st.radio("Select input method:", ("Text Input", "Voice Input"))

    user_input = None
    if input_method == "Text Input":
        with st.form(key='chat_form', clear_on_submit=True):
            user_input = st.text_input("You:", "")
            submitted = st.form_submit_button("Send")
    else:
        user_input = transcribe_audio()
        submitted = user_input is not None

    if submitted and user_input:
        st.session_state['messages'].append({"role": "user", "content": user_input})

        vector_store = get_vector_store()

        if vector_store is None:
            st.write("No documents available for querying. Please upload some data first.")
        else:
            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 10}
            )

            relevant_docs = retriever.get_relevant_documents(user_input)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])

            if len(relevant_docs) == 0:
                st.write("No relevant documents found.")
            else:
                references = set(doc.metadata.get('source', 'Unknown source') for doc in relevant_docs)

                conversation_history = ""
                for msg in st.session_state['messages']:
                    role = "User" if msg['role'] == 'user' else "Assistant"
                    conversation_history += f"{role}: {msg['content']}\n"

                prompt_template = """
                Use the following conversation history and context to answer the user's question.

                Conversation History:
                {conversation_history}

                Context:
                {context}

                Note: The context may include descriptions of images extracted from documents.

                Question:
                {question}

                Answer:
                """
                prompt = PromptTemplate(
                    input_variables=["conversation_history", "context", "question"],
                    template=prompt_template.strip(),
                )
                chain = prompt | llm
                response = chain.invoke({
                    "conversation_history": conversation_history,
                    "context": context,
                    "question": user_input,
                })

                if hasattr(response, 'content'):
                    response_text = response.content
                else:
                    response_text = str(response)

                if references:
                    response_text += f"\n\n**References:**\nFiles used: {', '.join(references)}"

                youtube_links = find_related_youtube_links(user_input, context)
                if youtube_links:
                    youtube_links_formatted = "\n".join([f"- [{url}]({url})" for url in youtube_links])
                    response_text += "\n\n**Related YouTube links:**\n" + youtube_links_formatted

                response_audio = text_to_speech_azure(response_text, synthesizer)
                st.session_state['messages'].append({"role": "assistant", "content": response_text, "audio": response_audio})

                st.session_state['recording_key'] += 1

    # Display chat history in a bubble style
    st.write("---")
    for message in st.session_state['messages']:
        if message['role'] == 'user':
            st.markdown(
                f"""
                <div class="user-row">
                    <div class="user-bubble">{message['content']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="assistant-row">
                    <div class="assistant-bubble">{message['content']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            if 'audio' in message and message['audio'] is not None:
                st.audio(message['audio'], format="audio/wav")

    st.write("---")
    st.header("Upload Folder")

    folder_path = st.text_input("Enter the path to the folder:")

    if st.button("Process Folder") and folder_path:
        vector_store = get_vector_store()

        docs = process_folder(folder_path, llm)

        if docs:
            if vector_store is None:
                vector_store = FAISS.from_documents(docs, embeddings_model)
                st.session_state['uploaded_documents'] = []
                existing_docs = set()
                for doc in docs:
                    folder = doc.metadata.get('folder', 'Unknown')
                    name = doc.metadata.get('source', 'Unknown')
                    tags = doc.metadata.get('tags', [])
                    doc_id = (folder, name)
                    if doc_id not in existing_docs:
                        st.session_state['uploaded_documents'].append({'folder': folder, 'name': name, 'tags': tags})
                        existing_docs.add(doc_id)
            else:
                vector_store.add_documents(docs)
                existing_docs = set((doc['folder'], doc['name']) for doc in st.session_state['uploaded_documents'])
                for doc in docs:
                    folder = doc.metadata.get('folder', 'Unknown')
                    name = doc.metadata.get('source', 'Unknown')
                    tags = doc.metadata.get('tags', [])
                    doc_id = (folder, name)
                    if doc_id not in existing_docs:
                        st.session_state['uploaded_documents'].append({'folder': folder, 'name': name, 'tags': tags})
                        existing_docs.add(doc_id)
            save_vector_store(vector_store)
            st.write("Folder data has been processed and added to the vector store.")
        else:
            st.write("No documents were created from the folder.")
    else:
        st.write("No documents were created from the folder.")

if __name__ == "__main__":
    main()
