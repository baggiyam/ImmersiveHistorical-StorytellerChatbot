import gradio as gr
from immersive_story_agent import ImmersiveStoryAgent  # import your class

agent = ImmersiveStoryAgent()
current_topic = None

def chat(user_input):
    global current_topic
    if user_input.lower() in ["exit", "quit"]:
        return "Thanks for exploring history!"
    elif current_topic is None or not agent.memory.is_likely_followup(user_input):
        current_topic = user_input
        story = agent.generate_story(current_topic)
        return story.get("story", "No story generated.")
    else:
        answer = agent.answer_question(current_topic, user_input)
        return answer

iface = gr.Interface(fn=chat,
                     inputs=gr.Textbox(placeholder="Ask about the Roman Forum..."),
                     outputs="text",
                     title="Immersive History Bot",
                     description="Ask historical questions and explore immersive stories!")

if __name__ == "__main__":
    iface.launch()
