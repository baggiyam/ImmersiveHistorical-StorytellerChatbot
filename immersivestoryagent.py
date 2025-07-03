To convert the provided `.ipynb` file into a `.py` file, you can extract the Python code from the `source` fields of each cell.

Here's the content for the `RAGagent.py` file:

```python
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


import pyttsx3

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
        # If TTS fails, just continue without crashing the program
        print("[TTS error]", e)


def safe_parse_json(llm_output: str):
    try:
        return json.loads(llm_output)
    except json.JSONDecodeError:
        cleaned = llm_output.strip()

        # Escape newlines and backslashes more aggressively
        cleaned = re.sub(r'(?<!\\)\\n', '\\\\\\\\n', cleaned)
        cleaned = re.sub(r'(?<!\\)\\\\', '\\\\\\\\', cleaned)

        # Remove any control chars that are not allowed in JSON strings
        cleaned = re.sub(r'[\x00-\x1f\x7f]', '', cleaned)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            print("❌ Still can't parse JSON after cleaning:", e)
            print("🧾 Raw output:", llm_output)
            return None

def speak_text(text, rate=150, volume=0.9):
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', rate)
        engine.setProperty('volume', volume)

        # Generate unique file name
        filename = f"/mnt/data/narration_{uuid.uuid4().hex}.wav"
        engine.save_to_file(text, filename)
        engine.runAndWait()

        # Play audio in notebook
        display(Audio(filename))
    except Exception as e:
        print("Narration failed:", e)

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
        context_string += f"--- Document {i+1} (From Video: {doc.metadata.get('video_title', 'N/A')}, Score: {doc.metadata.get('score'):.3f})---\n"
        context_string += f"{cleaned_text}\n\n"
        video_references.add(doc.metadata.get('video_title', 'Unknown Video'))
    return context_string, list(video_references)


def safe_parse_json(llm_output):
    import json
    try:
        return json.loads(llm_output)
    except json.JSONDecodeError:
        # Basic cleaning and retry (if needed)
        cleaned = llm_output.strip().replace('\n', ' ').replace('\r', '')
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            print("❌ Still can't parse JSON:", e)
            print("🧾 Raw output:", llm_output)
            return None


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

Create a unique, engaging introduction that draws the listener into the location or topic: "{question}"

Use vivid sensory details — what they might see, hear, feel, or smell — to make the setting come alive.

Only use information from the context. If it's not enough, say: "I can't answer that based on the available context."

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

        # Augment query with recent context for follow-ups only if this is NOT a follow-up question (handled in main loop)
        augmented_query = destination

        print(f"🗺 Destination/Query: {destination}")
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
            # traceback.print_exc() # Removed this as traceback is not imported
            return {
                "story": "An error occurred while generating the story.",
                "video_references": []
            }

    def answer_question(self, topic, question):
        session_id = topic.lower().strip()

        # Always augment question with recent context for follow-ups (if you want)
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
            question = response.content.strip()
            if question.lower() == "no suitable follow-up question found.":
                return None
            return question
        except Exception as e:
            print("❌ Error generating follow-up:", e)
            return None


def is_question(text):
    question_words = [
        "where", "what", "who", "when", "why", "how",
        "is", "are", "do", "does", "did", "can", "could", "would", "should", "which"
    ]
    text = text.strip().lower()
    return any(text.startswith(qw) for qw in question_words)


def print_story_and_followup(story_output):
    print("\n📚 Your Immersive Journey:\n")
    print("📝 Story:\n", story_output.get("story", "No story generated."))

    if story_output.get("video_references"):
        print("\n🎬 Referenced Videos:", ", ".join(story_output["video_references"]))
    else:
        print("📼 No specific video references.")

    follow_up_question = None
    if "story" in story_output:
        # Attempt to get a follow-up question based on the current story context
        follow_up_question = agent.ask_follow_up(story_output["story"][:1000])  # Limit prompt size

    if follow_up_question:
        print(f"\n🤔 Follow-up: do you want to know {follow_up_question}? (yes/no)")
        return True, follow_up_question
    else:
        return False, None


def main():
    global agent
    agent = ImmersiveStoryAgent()

    print("🎭 Welcome to Immersive Storytelling!")
    print("🌍 Where would you like to go today?")
    print("   ✨ Options: Great Pyramids, Roman Forum, Ancient Greece, Machu Picchu, Mesopotamia, Sangam Tamil Civilization")
    print("-" * 50)

    current_topic = None
    follow_up_question = None
    follow_up_mode = False

    while True:
        user_input = input("\n🧭 Your input (type 'exit' to quit, 'back' to choose a new place): ").strip()

        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Thanks for exploring with us!")
            break
        if user_input.lower() == "back":
            print("🔄 Resetting the journey. Where would you like to go now?")
            current_topic = None
            follow_up_mode = False
            continue

        # Check if user wants to go to a new location (match locations exactly or loosely)
        known_locations = [
            "Great Pyramids", "Roman Forum", "Ancient Greece",
            "Machu Picchu", "Mesopotamia", "Sangam Tamil Civilization",
            "Rome"
        ]

        # Detect if user input mentions a known location
        detected_location = None
        for loc in known_locations:
            if loc.lower() in user_input.lower():
                detected_location = loc
                break

        # If user typed a known location, start story there
        if detected_location:
            current_topic = detected_location
            story_output = agent.generate_story(current_topic)
            follow_up_mode, follow_up_question = print_story_and_followup(story_output)
            continue

        # If we are in follow-up mode (user answering yes/no to follow-up)
        if follow_up_mode:
            if user_input.lower() == "yes":
                if follow_up_question:
                    # Strictly continue story based on the follow-up question ONLY
                    story_output = agent.generate_story(follow_up_question)
                    follow_up_mode, follow_up_question = print_story_and_followup(story_output)
                    # Update current topic to follow-up question for future context
                    current_topic = follow_up_question
                    continue
                else:
                    print("No follow-up question to continue with.")
                    follow_up_mode = False
                    continue
            elif user_input.lower() == "no":
                print("Okay, feel free to ask another question or choose a new location.")
                follow_up_mode = False
                continue
            else:
                print("Please answer with 'yes' or 'no'.")
                continue

        # If input looks like a general question and we have a current topic
        if current_topic and is_question(user_input):
            answer = agent.answer_question(current_topic, user_input)
            print("\n💬 Answer:\n", answer)
            print(f"\n🤔 Do you want to continue with the story of {current_topic}? (yes/no)")
            follow_up_mode = True
            follow_up_question = None  # Reset follow-up question until new one generated
            continue

        # If no current topic, prompt user to pick location or ask question
        if not current_topic:
            print("I didn't catch a location. Please choose a location from the options or ask a question.")
            continue

        # If input is none of above, just remind user
        print("I didn't understand that. Please select a location or ask a relevant question.")

if __name__ == "__main__":
    main()