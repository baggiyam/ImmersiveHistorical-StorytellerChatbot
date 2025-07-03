import os
import time
import json
import re
import uuid
from dotenv import load_dotenv
from openai import OpenAI
import pinecone  # NEW correct import for Pinecone client
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_core.documents import Document
from SimpleConversationMemory import SimpleConversationMemory
import pyttsx3
from IPython.display import Audio, display

# Initialize Pinecone with API key
load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENVIRONMENT")  # If you have environment variable

pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

INDEX_NAME = "preprocessed-transcripts"

if INDEX_NAME not in pinecone.list_indexes():
    raise FileNotFoundError(f"Pinecone index '{INDEX_NAME}' not found.")

pinecone_index = pinecone.Index(INDEX_NAME)

def speak_text(text: str, rate: int = 150, volume: float = 0.9):
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
        cleaned = re.sub(r'(?<!\\)\\n', '\\\\\\\\n', cleaned)
        cleaned = re.sub(r'(?<!\\)\\\\', r'\\\\\\\\', cleaned)
        cleaned = re.sub(r'\\n', '\\\\n', cleaned)
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

        self.pinecone_index = pinecone_index

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

def main():
    agent = ImmersiveStoryAgent()
    print("🎭 Welcome to Immersive Storytelling!")
    print("🌍 Where would you like to go today?")
    print("   ✨ Options: Great Pyramids, Roman Forum, Athens Acropolis, or ask a history question.")

    while True:
        user_input = input("\nYour input (or 'exit' to quit): ").strip()
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye! Hope you enjoyed the journey.")
            break

        if is_question(user_input):
            print("🤔 Answering your question...")
            answer = agent.answer_question("history", user_input)
            print(f"🗣️ Answer:\n{answer}")
            speak_text(answer)
        else:
            print("📜 Generating your immersive story...")
            result = agent.generate_story(user_input)
            story_text = result.get("story", "Sorry, no story available.")
            print(f"\n📖 Immersive Story:\n{story_text}")
            speak_text(story_text)

            follow_up = agent.ask_follow_up(user_input)
            if follow_up and follow_up.lower() != "no suitable follow-up question found.":
                print(f"\n💡 Suggested follow-up question: {follow_up}")

if __name__ == "__main__":
    main()
