import os
import time
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from context_format import retrieve_context, format_context
from SimpleConversationMemory import SimpleConversationMemory
import json
from pathlib import Path
from pinecone import Pinecone, ServerlessSpec
import uuid
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import re
import pyttsx3
import uuid
from IPython.display import Audio, display
import traceback

# === TTSManager class ===
class TTSManager:
    def __init__(self, rate=150, volume=0.9):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", rate)
        self.engine.setProperty("volume", volume)

    def speak(self, text):
        if not text:
            return
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print("[TTS error]", e)

# Instantiate a single global TTS manager
tts_manager = TTSManager()

# === Safe JSON parser ===
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
        except json.JSONDecodeError as e:
            print("❌ Still can't parse JSON after cleaning:", e)
            print("🧾 Raw output:", llm_output)
            return None

# === Conversation memory ===
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

    def get_recent(self, count=1):
        return self.history[-count:] if self.history else []

# === Context retrieval functions ===
def retrieve_context(index, query, oa_client, top_k=5):
    emb = oa_client.embeddings.create(
        model="text-embedding-ada-002", input=[query]
    ).data[0].embedding

    res = index.query(vector=emb, top_k=top_k, include_metadata=True)
    docs = [
        Document(
            page_content=m.metadata.get("text", ""),
            metadata={
                "video_id":    m.metadata.get("video_id", "Unknown ID"),
                "video_title": m.metadata.get("video_title", "Unknown Video"),
                "score":       m.score,
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
            f"--- Document {i+1} "
            f"(Video: {d.metadata.get('video_title','N/A')}, "
            f"Score: {d.metadata.get('score'):.3f}) ---\n{txt}\n\n"
        )
        vids.add(d.metadata.get("video_title", "Unknown Video"))
    return ctx, list(vids)

# === Main Agent class ===
class ImmersiveStoryAgent:
    def __init__(self):
        load_dotenv()

        self.oa = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        idx_name = "preprocessed-transcripts"
        if idx_name not in [i.name for i in pc.list_indexes()]:
            raise RuntimeError(f"Pinecone index '{idx_name}' not found.")
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

        self.current_location = None
        self.current_story = None
        self.waiting_for_followup = False
        self.followup_question = None
        self.location_options = [
            "Great Pyramids", "Roman Forum", "Ancient Greece",
            "Machu Picchu", "Mesopotamia", "Sangam Tamil Civilization", "Rome"
        ]

    def generate_story(self, topic):
        self.current_location = topic
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

    def answer_question(self, topic, question):
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
                    self.followup_question = self.ask_follow_up(answer_text)
                    self.waiting_for_followup = True if self.followup_question else False
                    return answer_text.strip()
            except Exception:
                traceback.print_exc()

        try:
            fallback_resp = self.qa_llm.invoke(question)
            answer_text = fallback_resp.content if hasattr(fallback_resp, "content") else str(fallback_resp)
            self.waiting_for_followup = False
            return answer_text.strip()
        except Exception:
            traceback.print_exc()
            return "I couldn't find an answer to that question."

    def ask_follow_up(self, snippet):
        prompt = (
            "Suggest ONE engaging follow-up question based only on this text:\n"
            f"{snippet}\n\n"
            "If no good question, reply: No suitable follow-up question found."
        )
        try:
            resp = self.llm.invoke(prompt)
            if hasattr(resp, "content"):
                resp_text = resp.content.strip()
            elif isinstance(resp, str):
                resp_text = resp.strip()
            else:
                resp_text = str(resp).strip()

            if resp_text.lower().startswith("no suitable"):
                return None
            return resp_text
        except Exception:
            traceback.print_exc()
            return None

# === Main interaction loop ===
def main():
    agent = ImmersiveStoryAgent()
    yes_answers = {'yes', 'ok', 'tell me', 'continue', 'proceed'}

    print("🎭 Welcome to Immersive Storytelling!")
    print("🌍 Type a location to start your journey.")
    print("   ✨ Options: " + ", ".join(agent.location_options))
    print("------------------------------------------------------------------")

    while True:
        if agent.waiting_for_followup:
            user_input = input(f"\n🤔 Would you like to know more? (yes/no) or ask something else: ").strip().lower()

            if user_input == 'exit':
                break

            new_location = None
            for loc in agent.location_options:
                if loc.lower() in user_input:
                    new_location = loc
                    break

            if user_input in yes_answers:
                # Continue follow-up
                answer = agent.answer_question(agent.current_location, agent.followup_question)
                print(f"\n📚 Your Immersive Journey:")
                print(f"💬 Answer: {answer}")
                if agent.followup_question:
                    tts_manager.speak(answer)

            else:
                # End follow-up immediately
                agent.waiting_for_followup = False

                if new_location:
                    # New location requested
                    response = agent.generate_story(new_location)
                    print(f"\n📚 Your Immersive Journey:")
                    print(f"📝 Story: {response['story']}")
                    if response['video_references']:
                        print(f"\n🎬 Referenced Videos: {', '.join(response['video_references'])}")
                    if response['suggested_followup']:
                        print(f"❓ Suggested Follow-up: {response['suggested_followup']}")

                    tts_manager.speak(response['story'])

                else:
                    # Handle question on current location
                    answer = agent.answer_question(agent.current_location, user_input)
                    print(f"\n💬 Answer: {answer}")
                    tts_manager.speak(answer)
        else:
            user_input = input("\n🗺️ Choose a location or ask a question (type 'exit' to quit): ").strip()
            if user_input.lower() == 'exit':
                print("👋 Goodbye!")
                break

            new_location = None
            for loc in agent.location_options:
                if loc.lower() == user_input.lower():
                    new_location = loc
                    break

            if new_location:
                response = agent.generate_story(new_location)
                print(f"\n📚 Your Immersive Journey:")
                print(f"📝 Story: {response['story']}")
                if response['video_references']:
                    print(f"\n🎬 Referenced Videos: {', '.join(response['video_references'])}")
                if response['suggested_followup']:
                    print(f"❓ Suggested Follow-up: {response['suggested_followup']}")

                tts_manager.speak(response['story'])
            else:
                if agent.current_location:
                    answer = agent.answer_question(agent.current_location, user_input)
                    print(f"\n💬 Answer: {answer}")
                    tts_manager.speak(answer)
                else:
                    print("⚠️ Please select a location first.")

if __name__ == "__main__":
    main()
