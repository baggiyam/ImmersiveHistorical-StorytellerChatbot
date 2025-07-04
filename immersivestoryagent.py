import os
import re
import json
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_core.documents import Document
from pinecone import Pinecone
from SimpleConversationMemory import SimpleConversationMemory  # your custom memory
import traceback

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

        self.external_knowledge_llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.3,
            api_key=os.getenv("OPENAI_API_KEY")
        )

        self.story_prompt = PromptTemplate(
            input_variables=["question", "context", "video_references_list"],
            template="""
You are a master storyteller guiding a vivid, immersive journey through ancient history.

Use your creativity to write an engaging, immersive story based ONLY on the context below.

---

Context:
{context}

Video References List: {video_references_list}

Question: {question}

---

Respond in this JSON format:

{{
  "story": "Your immersive story goes here",
  "video_references": ["List of relevant video titles mentioned in the story"]
}}
"""
        )

        self.qa_prompt = PromptTemplate(
            input_variables=["question", "context"],
            template="""
Answer the question using ONLY the information from the context below.
If the answer cannot be found in the context, reply: "I can't answer that based on the available context."

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

        print("🔍 Retrieving context for question...")
        matches = retrieve_context(self.pinecone_index, augmented_query, self.openai_client, top_k=5)
        context, _ = format_context(matches)

        try:
            answer_response = self.qa_chain.invoke({
                "question": augmented_query,
                "context": context
            })
            answer = answer_response.content.strip()
        except Exception as e:
            print("❌ Error during context-based answering:", e)
            answer = None

        if not answer or "I can't answer that based on the available context" in answer:
            try:
                prompt = f"Answer the question truthfully and informatively:\n\nQuestion: {question}\nAnswer:"
                fallback_response = self.external_knowledge_llm.invoke(prompt)
                answer = fallback_response.content.strip()
                if not answer:
                    answer = "Sorry, I am not trained on this topic to give an answer."
            except Exception as e:
                print("❌ Error during external knowledge answering:", e)
                answer = "Sorry, I am not trained on this topic to give an answer."

        return answer

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
    print("   ✨ Options: Great Pyramids, Roman Forum, Ancient Greece, Machu Picchu, Mesopotamia, Sangam Tamil Civilization")
    print("-" * 50)

    current_topic = None
    story_output = None
    follow_up_question = None
    follow_up_mode = False

    while True:
        user_input = input("\n🧭 Your input (type 'exit' to quit, 'back' to choose a new place): ").strip()

        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Thanks for exploring with us!")
            break
        elif user_input.lower() == "back":
            if current_topic:
                agent.memory.clear_session(current_topic.lower().strip())
            current_topic = None
            story_output = None
            follow_up_question = None
            follow_up_mode = False
            print("🔄 Let's pick a new destination!")
            continue

        # Accept "yes", "ok", "continue" to continue with follow-up question
        if user_input.lower() in ["yes", "ok", "continue"] and follow_up_question and follow_up_mode:
            answer = agent.answer_question(current_topic, follow_up_question)
            print("\n💬 Answer:\n", answer)
            follow_up_mode = False
            continue

        if current_topic is None or (not is_question(user_input) and not follow_up_mode):
            current_topic = user_input
            story_output = agent.generate_story(current_topic)

            print("\n📚 Your Immersive Journey Begins:\n")
            print("📝 Story:\n", story_output.get("story", "No story generated."))

            if story_output.get("video_references"):
                print("\n🎬 Referenced Videos:", ", ".join(story_output["video_references"]))
            else:
                print("📼 No specific video references.")

            follow_up_question = agent.ask_follow_up(current_topic)

            if follow_up_question and follow_up_question.lower() != "no suitable follow-up question found.":
                print(f"\n❓ Follow-up Question: {follow_up_question}")
                print("💡 To answer, type 'yes', 'ok', or 'continue'. To ask your own question, just type it.")
                follow_up_mode = True
            else:
                follow_up_mode = False
                follow_up_question = None
            continue

        if is_question(user_input) or follow_up_mode:
            answer = agent.answer_question(current_topic, user_input)
            print("\n💬 Answer:\n", answer)
            follow_up_mode = False
            follow_up_question = None
            continue

        # If none of the above, assume new topic input
        current_topic = user_input
        story_output = agent.generate_story(current_topic)

        print("\n📚 Your Immersive Journey Begins:\n")
        print("📝 Story:\n", story_output.get("story", "No story generated."))

        if story_output.get("video_references"):
            print("\n🎬 Referenced Videos:", ", ".join(story_output["video_references"]))
        else:
            print("📼 No specific video references.")

        follow_up_question = agent.ask_follow_up(current_topic)

        if follow_up_question and follow_up_question.lower() != "no suitable follow-up question found.":
            print(f"\n❓ Follow-up Question: {follow_up_question}")
            print("💡 To answer, type 'yes', 'ok', or 'continue'. To ask your own question, just type it.")
            follow_up_mode = True
        else:
            follow_up_mode = False
            follow_up_question = None

if __name__ == "__main__":
    main()
