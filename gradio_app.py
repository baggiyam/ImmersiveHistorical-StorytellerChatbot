import gradio as gr
import os
import time
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from SimpleConversationMemory import SimpleConversationMemory
import json
from pathlib import Path
import uuid
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_core.documents import Document
import re
import pyttsx3
from IPython.display import Audio, display # Note: IPython.display is for Jupyter/IPython environments. Will not work in a pure Python script without adjustments.

# --- Functions from your original script ---

def speak_text(text: str, rate: int = 150, volume: float = 0.9):
    """Speak the given text aloud (non‑blocking). Keeps original prints intact."""
    if not text:
        return
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", rate)
        engine.setProperty("volume", volume)
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print("[TTS error]", e)

def safe_parse_json(llm_output: str):
    try:
        return json.loads(llm_output)
    except json.JSONDecodeError:
        cleaned = llm_output.strip()
        cleaned = re.sub(r"(?<!\\)\\n", "\\\\\\\\n", cleaned)
        cleaned = re.sub(r"(?<!\\)\\", r"\\\\\\\\", cleaned)
        cleaned = re.sub(r"\n", "\\\\n", cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            print("❌ Still can't parse JSON after cleaning:", e)
            print("🧾 Raw output:", llm_output)
            return None

def retrieve_context(pinecone_index, query_text, openai_client, top_k=5):
    query_embedding_response = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=[query_text]
    )
    query_embedding = query_embedding_response.data[0].embedding

    results = pinecone_index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    retrieved_documents = []
    for match in results.matches:
        retrieved_documents.append(
            Document(
                page_content=match.metadata.get('text', ''),
                metadata={
                    "video_id": match.metadata.get('video_id', 'Unknown ID'),
                    "video_title": match.metadata.get('video_title', 'Unknown Video'),
                    "score": match.score
                }
            )
        )
    return retrieved_documents

def format_context(documents):
    context_string = ""
    video_references = set()
    for i, doc in enumerate(documents):
        cleaned_text = re.sub(r'^\s*[\n\s.]+', '', doc.page_content).strip()
        context_string += f"--- Document {i+1} (From Video: {doc.metadata.get('video_title', 'N/A')}, Score: {doc.metadata.get('score'):.3f}) ---\n"
        context_string += f"{cleaned_text}\n\n"
        video_references.add(doc.metadata.get('video_title', 'Unknown Video'))
    return context_string, list(video_references)

class ImmersiveStoryAgent:
    def __init__(self):
        load_dotenv()

        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            raise ValueError("PINECONE_API_KEY not set.")

        pc = Pinecone(api_key=pinecone_api_key)
        self.index_name = "preprocessed-transcripts"
        if self.index_name not in [index.name for index in pc.list_indexes()]:
            raise FileNotFoundError(f"Pinecone index '{self.index_name}' not found.")
        self.pinecone_index = pc.Index(self.index_name)

        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.85,
            api_key=os.getenv("OPENAI_API_KEY")
        )

        self.qa_llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.2,
            api_key=os.getenv("OPENAI_API_KEY")
        )

        self.story_prompt = PromptTemplate(
            input_variables=["question", "context", "video_references_list"],
            template="""
You are a master storyteller guiding a vivid, immersive journey through ancient history.

Only use information from the context. If it's not enough, say: "I can't answer that based on the available context."

Write like this:

"Close your eyes and travel back in time..."

---

Context:
{context}

Video References List: {video_references_list}

Question: {question}

---

Now write the immersive story. Respond in this JSON format:

{{
  "story": "Your immersive story goes here",
  "video_references": ["List of relevant video titles mentioned in the story"]
}}
"""
        )

        self.qa_prompt = PromptTemplate(
            input_variables=["question", "context"],
            template="""
Answer the question using only the information from the context below. If you don't know the answer, say "I can't answer that based on the available context."

Context:
{context}

Question: {question}
Answer:"""
        )

        self.story_chain = LLMChain(llm=self.llm, prompt=self.story_prompt)
        self.qa_chain = LLMChain(llm=self.qa_llm, prompt=self.qa_prompt)
        self.memory = SimpleConversationMemory(max_history=4)

    def generate_story(self, destination):
        session_id = destination.lower().strip()

        # Augment query with recent context for follow-ups
        if self.memory.is_likely_followup(destination):
            recent_context = self.memory.get_recent_context(session_id)
            augmented_query = (recent_context + "\n\n" + destination) if recent_context else destination
        else:
            augmented_query = destination

        print(f"🗺 Destination: {destination}")
        print("🔍 Retrieving context...")

        matches = retrieve_context(self.pinecone_index, augmented_query, self.openai_client, top_k=5)

        if not matches:
            return {
                "story": "I can't find enough information about that topic. Please try another destination or ask a specific question.",
                "video_references": []
            }

        context, video_titles = format_context(matches)
        print("📖 Generating immersive story...")

        try:
            raw_output = self.story_chain.run(
                question=augmented_query,
                context=context,
                video_references_list=", ".join(video_titles)
            )
            parsed = safe_parse_json(raw_output)
            if not parsed or "story" not in parsed:
                raise ValueError("Missing expected keys in story output")

            self.memory.add_qa_pair(destination, parsed['story'], session_id)

            return parsed
        except Exception as e:
            print("❌ Error during story generation:", e)
            import traceback
            traceback.print_exc()
            return {
                "story": "An error occurred while generating the story.",
                "video_references": []
            }

    def answer_question(self, topic, question):
        session_id = topic.lower().strip()

        if self.memory.is_likely_followup(question):
            recent_context = self.memory.get_recent_context(session_id)
            augmented_query = (recent_context + "\n\n" + question) if recent_context else question
        else:
            augmented_query = question

        matches = retrieve_context(self.pinecone_index, augmented_query, self.openai_client, top_k=5)
        context, _ = format_context(matches)

        try:
          answer = self.qa_chain.invoke({
           "question": augmented_query,
              "context": context
}).content
          return answer
        except Exception as e:
            print("❌ Error during answering:", e)
            return "I can't answer that based on the available context."

    def ask_follow_up(self, current_topic):
        matches = retrieve_context(self.pinecone_index, current_topic, self.openai_client, top_k=3)
        context, _ = format_context(matches)

        prompt = f"""
You are an expert historian. Based only on the context below, suggest one engaging follow-up question
that would help the user continue exploring the story about "{current_topic}".

If the context doesn't provide enough information for a good question, say "No suitable follow-up question found."

Context:
{context}
"""
        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            print("❌ Error generating follow-up:", e)
            return "No suitable follow-up question found."

def is_question(text):
    question_words = [
        "where", "what", "who", "when", "why", "how",
        "is", "are", "do", "does", "did", "can", "could", "would", "should", "which"
    ]
    text = text.strip().lower()
    return any(text.startswith(qw) for qw in question_words)

# --- Gradio Interface Code ---

# Global instance of the agent and conversation state
# This is a common pattern for Gradio apps to maintain state across interactions
agent = ImmersiveStoryAgent()
current_topic = gr.State(None)
follow_up_question = gr.State(None)
follow_up_mode = gr.State(False)

def welcome_message():
    return "🎭 Welcome to Immersive Storytelling! 🌍 Where would you like to go today? \n\n✨ Options: Great Pyramids, Roman Forum, Ancient Greece, Machu Picchu, Mesopotamia, Sangam Tamil Civilization"

def process_input(user_input, chat_history, current_topic_state, follow_up_question_state, follow_up_mode_state):
    history = chat_history or []
    output_message = ""
    new_follow_up_question = None
    new_follow_up_mode = False
    new_current_topic = current_topic_state

    if not user_input:
        return history, current_topic_state, follow_up_question_state, follow_up_mode_state

    user_input_lower = user_input.lower().strip()

    if user_input_lower in ["exit", "quit"]:
        history.append((user_input, "👋 Thanks for exploring with us!"))
        return history, None, None, False # Reset all states for exit
    elif user_input_lower == "back":
        if new_current_topic:
            agent.memory.clear_session(new_current_topic.lower().strip())
        output_message = "🔄 Let's pick a new destination!\n\n" + welcome_message()
        history.append((user_input, output_message))
        return history, None, None, False

    if user_input_lower == "yes" and follow_up_question_state and follow_up_mode_state:
        answer = agent.answer_question(new_current_topic, follow_up_question_state)
        output_message = f"\n💬 Answer:\n {answer}"
        new_follow_up_mode = False
        new_follow_up_question = None # Clear follow-up question after answering
        history.append((user_input, output_message))
        return history, new_current_topic, new_follow_up_question, new_follow_up_mode

    if new_current_topic is None or (not is_question(user_input) and not follow_up_mode_state):
        new_current_topic = user_input
        story_output = agent.generate_story(new_current_topic)
        story_text = story_output.get("story", "No story generated.")
        video_refs = ", ".join(story_output["video_references"]) if story_output.get("video_references") else "No specific video references."
        output_message = f"\n📚 Your Immersive Journey Begins:\n\n📝 Story:\n {story_text}\n\n🎬 Referenced Videos: {video_refs}"

        new_follow_up_question = agent.ask_follow_up(new_current_topic)
        if new_follow_up_question and new_follow_up_question.lower() != "no suitable follow-up question found.":
            output_message += f"\n\n🤔 Follow-up: do you want to know {new_follow_up_question}? (yes/no)"
            new_follow_up_mode = True
        else:
            new_follow_up_mode = False
            new_follow_up_question = None # Ensure it's None if no suitable question
    elif is_question(user_input):
        answer = agent.answer_question(new_current_topic, user_input)
        output_message = f"\n💬 Answer:\n {answer}"
        new_follow_up_mode = False
        new_follow_up_question = None # Clear follow-up question if a new direct question is asked
    else: # User input is not a question and not a "yes" for follow-up, implies continuation of story
        enriched_topic = f"{new_current_topic} {user_input}"
        story_output = agent.generate_story(enriched_topic)
        story_text = story_output.get("story", "No story generated.")
        video_refs = ", ".join(story_output["video_references"]) if story_output.get("video_references") else "No specific video references."
        output_message = f"\n📖 The Story Continues:\n\n📝 Story:\n {story_text}\n\n🎬 Referenced Videos: {video_refs}"

        new_follow_up_question = agent.ask_follow_up(new_current_topic)
        if new_follow_up_question and new_follow_up_question.lower() != "no suitable follow-up question found.":
            output_message += f"\n\n🤔 Follow-up: do you want to know {new_follow_up_question}? (yes/no)"
            new_follow_up_mode = True
        else:
            new_follow_up_mode = False
            new_follow_up_question = None # Ensure it's None if no suitable question

    history.append((user_input, output_message))
    return history, new_current_topic, new_follow_up_question, new_follow_up_mode

# Gradio Interface
with gr.Blocks(title="Immersive Historical Storyteller") as demo:
    gr.Markdown("# 🎭 Immersive Historical Storyteller Chatbot")
    gr.Markdown(welcome_message())

    chatbot = gr.Chatbot(label="Conversation")
    msg = gr.Textbox(label="Your Input", placeholder="Type your destination or question...")

    # Using gr.State to maintain session-specific data
    current_topic_state = gr.State(None)
    follow_up_question_state = gr.State(None)
    follow_up_mode_state = gr.State(False)

    msg.submit(
        process_input,
        [msg, chatbot, current_topic_state, follow_up_question_state, follow_up_mode_state],
        [chatbot, current_topic_state, follow_up_question_state, follow_up_mode_state]
    )
    # Clear button to reset the conversation
    clear_button = gr.Button("Clear Conversation & Reset")
    clear_button.click(lambda: ([], None, None, False), None, [chatbot, current_topic_state, follow_up_question_state, follow_up_mode_state])

demo.launch()
