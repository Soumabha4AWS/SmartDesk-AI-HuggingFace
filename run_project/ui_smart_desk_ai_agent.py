import sys
from pathlib import Path
from uuid import uuid4
from dotenv import load_dotenv

# Load this repository's .env so the app works from any current directory.
load_dotenv(
    dotenv_path=Path(__file__).resolve().parents[1] / ".env",
    override=True,
)

# Make project/ importable from one level up
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "project"))
from cache.answer_cache import cache_answer, get_cached_answer  # noqa: E402
from smart_desk_ai_agent import build_support_agent, ask  # noqa: E402

import gradio as gr  # UI

THEME = gr.themes.Soft()
CUSTOM_CSS = """
body, .gradio-container { background-color: #0b0f19 !important; color: #e5e7eb !important; }
.gradio-container .block, .gradio-container .panel, .gradio-container .tabs, .gradio-container .form { background-color: #0b0f19 !important; }
.gr-markdown, .gr-markdown * { color: #e5e7eb !important; }
.prose, .prose :where(p, li, strong, em, code) { color: #e5e7eb !important; }
.prose h1, .prose h2, .prose h3, .prose h4 { color: #f9fafb !important; }
.prose a { color: #60a5fa !important; }
.prose code { background-color: #0b1220 !important; color: #f59e0b !important; }
.prose pre { background-color: #0b1220 !important; color: #e5e7eb !important; border: 1px solid #374151 !important; }
.gradio-container textarea,
.gradio-container input {
  background-color: #111827 !important;
  color: #f9fafb !important;
  -webkit-text-fill-color: #f9fafb !important;
  caret-color: #f9fafb !important;
  border-color: #4b5563 !important;
}
.gradio-container textarea::placeholder,
.gradio-container input::placeholder {
  color: #9ca3af !important;
  -webkit-text-fill-color: #9ca3af !important;
  opacity: 1 !important;
}
.gradio-container textarea:focus,
.gradio-container input:focus {
  border-color: #60a5fa !important;
  box-shadow: 0 0 0 1px #60a5fa !important;
}
.gr-button { background-color: #1f2937 !important; color: #e5e7eb !important; border-color: #374151 !important; }
.gr-button:hover { background-color: #374151 !important; }
"""

# Build the agent pipeline once at startup
print("Building Smart Desk Ai Agent UI...")
agent = build_support_agent(collection_name="smartdesk_demo")
app = agent["app"]
print("Pipeline ready. Open the UI to interact.\n")


def run_query(message, history, conversation_id, conversations):
    if not message or not str(message).strip():
        raise gr.Error("Please enter a message.")
    if not conversation_id:
        raise gr.Error("Select or create a conversation before sending a message.")

    result = get_cached_answer(conversation_id, message)
    if result is None:
        result = ask(app, message, thread_id=conversation_id)
        response = result.get("response", "")
        intent = result.get("intent", "")
        retrieved_sources = result.get("retrieved_sources", []) or []
        cache_answer(
            conversation_id,
            message,
            response,
            intent,
            retrieved_sources,
        )
    else:
        response = result["response"]
        intent = result["intent"]
        retrieved_sources = result["retrieved_sources"]

    sources = ", ".join(retrieved_sources)

    updated_conversations = dict(conversations or {})
    updated_conversations[conversation_id] = [
        *history,
        {"role": "user", "content": message},
        {"role": "assistant", "content": response},
    ]
    return response, intent, sources, updated_conversations


def initialize_conversation(conversation_ids):
    ids = list(conversation_ids or [])
    if not ids:
        ids.append(f"chat-{uuid4().hex}")
    return gr.update(choices=ids, value=ids[0]), ids


def create_conversation(conversation_ids):
    ids = list(conversation_ids or [])
    conversation_id = f"chat-{uuid4().hex}"
    ids.append(conversation_id)
    return gr.update(choices=ids, value=conversation_id), ids


def load_conversation(conversation_id, conversations):
    history = (conversations or {}).get(conversation_id, [])
    return history, history, history, "", ""


def build_ui():
    with gr.Blocks(title="Smart Desk AI Agent") as demo:
        gr.Markdown("# Smart Desk AI Agent\nAsk questions about HR/IT or tickets.")
        conversation_ids = gr.State([])
        conversations = gr.State({})

        with gr.Row():
            conversation = gr.Dropdown(
                label="Conversation",
                choices=[],
                interactive=True,
                scale=4,
            )
            new_conversation = gr.Button("New conversation")

        intent = gr.Textbox(label="Detected intent", interactive=False)
        sources = gr.Textbox(label="Retrieved sources", interactive=False)
        chat = gr.ChatInterface(
            run_query,
            chatbot=gr.Chatbot(),
            additional_inputs=[conversation, conversations],
            additional_outputs=[intent, sources, conversations],
            title="Smart Desk AI",
            description="Choose a conversation to continue it, or create a new one. Each conversation has independent memory.",
        )

        demo.load(
            initialize_conversation,
            inputs=[conversation_ids],
            outputs=[conversation, conversation_ids],
        )
        new_conversation.click(
            create_conversation,
            inputs=[conversation_ids],
            outputs=[conversation, conversation_ids],
        ).then(
            lambda: ([], [], [], "", ""),
            outputs=[
                chat.chatbot,
                chat.chatbot_state,
                chat.chatbot_value,
                intent,
                sources,
            ],
        )
        conversation.change(
            load_conversation,
            inputs=[conversation, conversations],
            outputs=[
                chat.chatbot,
                chat.chatbot_state,
                chat.chatbot_value,
                intent,
                sources,
            ],
        )
    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(theme=THEME, css=CUSTOM_CSS)
