import gradio as gr
import os
import time
import json
import traceback
import re
import pyttsx3
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.documents import Document


# --- Initialize TTS ---
class TTSManager:
    def __init__(self, rate=150, volume=0.9):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", rate)
        self.engine.setProperty("volume", volume)
        self.speaking = False

    def speak(self, text):
        if not text:
            return
        try:
            self.speaking = True
            self.engine.say(text)
            self.engine.runAndWait()
            self.speaking = False
        except Exception as e:
            print("[TTS error]", e)
            self.speaking = False

    def stop(self):
        if self.speaking:
            self.engine.stop()
            self.speaking = False

tts_manager = TTSManager()

# --- Safe JSON parsing ---
def safe_parse_json(llm_output: str):
    try:
        return json.loads(llm_output)
    except json.JSONDecodeError:
        cleaned = llm_output.strip()
        cleaned = re.sub(r"(?<!\\)\\n", "\\\\n", cleaned)
        cleaned = re.sub(r"(?<!\\)\\", r"\\\\", cleaned)
        cleaned = re.sub(r"\n", "\\n", cleaned)
        try:
            return json.loads(cleaned)
        except Exception as e:
            print("JSON parsing error after cleaning:", e)
            print("Raw output:", llm_output)
            return None

# --- Context retrieval ---
def retrieve_context(index, query, oa_client, top_k=5):
    emb = oa_client.embeddings.create(
        model="text-embedding-ada-002", input=[query]
    ).data[0].embedding

    res = index.query(vector=emb, top_k=top_k, include_metadata=True)
    docs = [
        Document(
            page_content=m.metadata.get("text", ""),
            metadata={
                "video_id": m.metadata.get("video_id", "Unknown ID"),
                "video_title": m.metadata.get("video_title", "Unknown Video"),
                "score": m.score,
            },
        )
        for m in res.matches
    ]
    return docs

def format_context(docs):
    ctx, vids = "", set()
    for i, d in enumerate(docs):
        txt = re.sub(r"^\s*[\n\s.]+", "", d.page_content).strip()
        ctx += (
            f"--- Document {i + 1} "
            f"(Video: {d.metadata.get('video_title', 'N/A')}, "
            f"Score: {d.metadata.get('score'):.3f}) ---\n{txt}\n\n"
        )
        vids.add(d.metadata.get("video_title", "Unknown Video"))
    return ctx, list(vids)

# --- Simple conversation memory ---
class SimpleConversationMemory:
    def __init__(self, max_history=4):
        self.max_history = max_history
        self.history = []

    def add_qa_pair(self, question, answer, topic=None):
        self.history.append({
            'question': question,
            'answer': answer,
            'topic': topic,
            'timestamp': time.time()
        })
        if len(self.history) > self.max_history:
            self.history.pop(0)

# --- Immersive Story Agent ---
class ImmersiveStoryAgent:
    def __init__(self):
        load_dotenv()

        self.oa = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        idx_name = "preprocessed-transcripts"
        if idx_name not in [i.name for i in pc.list_indexes()]:
            print(f"Warning: Pinecone index '{idx_name}' not found.")
            self.index = None
        else:
            self.index = pc.Index(idx_name)

        self.llm = ChatOpenAI(model="gpt-4", temperature=0.85)
        self.qa_llm = ChatOpenAI(model="gpt-4", temperature=0.2)

        self.story_prompt = PromptTemplate(
            input_variables=["question", "context", "video_references_list"],
            template="""
You are a master storyteller guiding a vivid, immersive journey through ancient history.

Create an engaging introduction that draws the listener into the location or topic: "{question}"

Use ONLY the information in the context. If the context is insufficient,
respond exactly: "I can't answer that based on the available context."

---

Context:
{context}

Video References List: {video_references_list}

Question: {question}

---

Respond in this JSON format:

{{
  "story": "Your immersive story...",
  "video_references": ["Video title 1","Video title 2"],
  "suggested_followup": "A suggested follow-up question about this story"
}}
"""
        )

        self.qa_prompt = PromptTemplate(
            input_variables=["question", "context"],
            template="""
Answer the question using ONLY the context below.

If the context is insufficient, say exactly:
"I can't answer that based on the available context."

Context:
{context}

Question: {question}
Answer:"""
        )

        self.story_chain = LLMChain(llm=self.llm, prompt=self.story_prompt)
        self.qa_chain = LLMChain(llm=self.qa_llm, prompt=self.qa_prompt)
        self.memory = SimpleConversationMemory(max_history=4)

        self.current_story = None
        self.waiting_for_followup = False
        self.followup_question = None

    def generate_story(self, topic):
        if not self.index:
            return {
                "story": "Pinecone index not available. Please check API keys and index name.",
                "video_references": [],
                "suggested_followup": None
            }
        docs = retrieve_context(self.index, topic, self.oa, top_k=5)
        if not docs:
            return {
                "story": "I can't answer that based on the available context.",
                "video_references": [],
                "suggested_followup": None
            }

        ctx, vids = format_context(docs)
        if len(ctx.strip()) < 50:
            return {
                "story": "I can't answer that based on the available context.",
                "video_references": [],
                "suggested_followup": None
            }

        try:
            raw = self.story_chain.invoke({
                "question": topic,
                "context": ctx,
                "video_references_list": ", ".join(vids)
            })

            if hasattr(raw, "text"):
                output = raw.text
            elif hasattr(raw, "content"):
                output = raw.content
            elif isinstance(raw, dict) and "text" in raw:
                output = raw["text"]
            else:
                output = str(raw)

            parsed = safe_parse_json(output)
            if not parsed or not parsed.get("story"):
                raise ValueError("No story generated")

            self.current_story = parsed["story"]
            self.followup_question = parsed.get("suggested_followup")
            self.waiting_for_followup = True if self.followup_question else False

            self.memory.add_qa_pair(topic, parsed["story"], topic.lower())
            return parsed

        except Exception as e:
            traceback.print_exc()
            return {
                "story": f"An error occurred while generating the story: {str(e)}",
                "video_references": [],
                "suggested_followup": None
            }

    def answer_question(self, question):
        if not self.index:
            return "Pinecone index not available. Please check API keys and index name."

        docs = retrieve_context(self.index, question, self.oa, top_k=5)
        ctx, _ = format_context(docs)

        if len(ctx.strip()) >= 50:
            try:
                resp = self.qa_chain.invoke({
                    "question": question,
                    "context": ctx
                })

                if hasattr(resp, "text"):
                    answer_text = resp.text
                elif hasattr(resp, "content"):
                    answer_text = resp.content
                elif isinstance(resp, dict) and "text" in resp:
                    answer_text = resp["text"]
                else:
                    answer_text = str(resp)

                if answer_text.strip():
                    self.waiting_for_followup = False
                    self.memory.add_qa_pair(question, answer_text.strip())
                    return answer_text.strip()
            except Exception:
                traceback.print_exc()

        try:
            fallback_resp = self.qa_llm.invoke(question)
            answer_text = fallback_resp.content if hasattr(fallback_resp, "content") else str(fallback_resp)
            self.waiting_for_followup = False
            self.memory.add_qa_pair(question, answer_text.strip())
            return answer_text.strip()
        except Exception:
            traceback.print_exc()
            return "I couldn't find an answer to that question."

# --- Global agent ---
agent = ImmersiveStoryAgent()

# --- Gradio Interface functions ---
def respond(message, chat_history):
    if not message or message.strip() == "":
        return "", chat_history

    message = message.strip()
    available_topics = [
        "Great Pyramids", "Roman Forum", "Ancient Greece", "Machu Picchu",
        "Mesopotamia", "Sangam Tamil Civilization", "Rome"
    ]

    # Check if user wants to start a story from a known topic
    matched_topic = next((t for t in available_topics if t.lower() == message.lower()), None)

    if matched_topic:
        # Generate immersive story
        story_data = agent.generate_story(matched_topic)
        story = story_data.get("story", "")
        video_refs = story_data.get("video_references", [])
        followup = story_data.get("suggested_followup", None)

        # Format the response
        response = story
        if video_refs:
            response += f"\n\n📚 References: {', '.join(video_refs)}"
        if followup:
            response += f"\n\n🤔 Follow-up: {followup}"

        chat_history.append((message, response))
        return "", chat_history
    else:
        # Treat as question to answer
        answer = agent.answer_question(message)
        chat_history.append((message, answer))
        return "", chat_history

def clear_chat():
    agent.memory.history = []
    return []

def on_play(text):
    if text:
        tts_manager.speak(text)

def on_stop():
    tts_manager.stop()

# --- Markdown list of available topics ---
available_topics_md = """
### Available Topics (type exactly to start a story):

- Great Pyramids
- Roman Forum
- Ancient Greece
- Machu Picchu
- Mesopotamia
- Sangam Tamil Civilization
- Rome

Type any of the above topic names exactly to hear an immersive story,
or ask any question about these topics or ancient history in general.
"""

# --- Build Gradio app ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏛️ Immersive Ancient History Storyteller")
    gr.Markdown(available_topics_md)

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=500, bubble_full_width=False)
            msg = gr.Textbox(label="Your message", placeholder="Type a topic or question...")

            with gr.Row():
                submit_btn = gr.Button("Submit", variant="primary")
                clear_btn = gr.Button("Clear History")

        with gr.Column(scale=1):
            gr.Markdown("### Audio Controls")
            with gr.Row():
                play_btn = gr.Button("▶ Play Last Response")
                stop_btn = gr.Button("■ Stop Audio")

    # Event handlers
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    submit_btn.click(respond, [msg, chatbot], [msg, chatbot])
    clear_btn.click(clear_chat, [], [chatbot])
    play_btn.click(on_play, chatbot, [])
    stop_btn.click(on_stop, [], [])

demo.launch()