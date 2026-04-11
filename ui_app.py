"""Gradio UI for DocuSense Pro — wraps rag_core.py pipeline."""

import os
import re
import signal
import sys
import tempfile
import hashlib
import threading
import queue
import atexit
from typing import Optional, List, Tuple, Any

import sounddevice as sd
import soundfile as sf
import speech_recognition as sr
import pyttsx3

import gradio as gr
from dotenv import load_dotenv

from rag_core import (
    get_openai_model,
    get_ollama_model,
    build_context_and_answer,
    build_rag_index,
    stream_context_and_answer,
)

load_dotenv()


# ── Voice controller ──────────────────────────────────────────────────────────

class VoiceController:
    """Thread-safe TTS controller with sentence-level interruption."""

    def __init__(self):
        self.engine = None
        self.stop_flag = False
        self.lock = threading.Lock()
        self._q: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._init_engine()

    def _init_engine(self) -> bool:
        if self.engine is not None:
            return True
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", 150)
            return True
        except Exception:
            self.engine = None
            return False

    def stop(self):
        with self.lock:
            self.stop_flag = True
            if self.engine:
                try:
                    self.engine.stop()
                except Exception:
                    pass
                self.engine = None

    def _worker(self):
        while True:
            try:
                text = self._q.get(timeout=1.0)
            except queue.Empty:
                continue
            if text == "__STOP__":
                break
            self._say(text)
            self._q.task_done()

    def _say(self, text: str):
        sentences = re.split(r"(?<=[.!?])\s+", text)
        for s in sentences:
            with self.lock:
                if self.stop_flag or self.engine is None:
                    return
            if s.strip():
                try:
                    self.engine.say(s)
                    self.engine.runAndWait()
                except Exception:
                    pass

    def speak(self, text: str):
        if not text.strip():
            return
        with self.lock:
            self.stop_flag = False
            if not self._init_engine():
                return
        self._q.put(text)
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

    def cleanup(self):
        self.stop()
        self._q.put("__STOP__")


voice_controller = VoiceController()


def _stop_voice():
    voice_controller.stop()


def _speak_async(text: str):
    voice_controller.speak(text)


def cleanup_resources():
    voice_controller.cleanup()


# ── Model init (runs once at startup) ────────────────────────────────────────

openai_model, openai_model_name = get_openai_model()
ollama_model, ollama_model_name = get_ollama_model()


# ── Utilities ─────────────────────────────────────────────────────────────────

def _file_signature(path: str) -> str:
    """MD5 fingerprint of first + last 8 KB — used as cache key."""
    h = hashlib.md5()
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        h.update(f.read(8192))
        if size > 8192:
            f.seek(max(0, size - 8192))
            h.update(f.read(8192))
    return h.hexdigest()


def _extract_qa_history(chatbot_history: list) -> List[Tuple[str, str]]:
    """Convert Gradio messages list → (user, assistant) pairs for rag_core."""
    pairs: List[Tuple[str, str]] = []
    pending: Optional[str] = None
    for msg in (chatbot_history or []):
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            pending = content
        elif role == "assistant" and pending is not None:
            pairs.append((pending, content))
            pending = None
    return pairs[-3:]


# ── Event handlers ────────────────────────────────────────────────────────────

def handle_file_upload(file) -> Tuple[Any, str]:
    """Build FAISS + BM25 index and return (bundle, status_html)."""
    if file is None:
        return None, _status_html("idle", "No document loaded yet.")
    try:
        faiss_store, bm25_tuple, parent_store, info = build_rag_index(file.name)
        name = os.path.basename(file.name)
        ext = os.path.splitext(name)[1].upper().lstrip(".")
        return (
            {"faiss": faiss_store, "bm25": bm25_tuple, "parents": parent_store},
            _status_html("ok", f"<strong>{name}</strong> &nbsp;·&nbsp; {ext} &nbsp;·&nbsp; Ready"),
        )
    except Exception as e:
        return None, _status_html("error", f"Upload failed: {e}")


def _status_html(kind: str, text: str) -> str:
    """Return a styled status pill as an HTML string."""
    styles = {
        "ok":    ("background:#dcfce7;border:1px solid #86efac;color:#166534;", "✔"),
        "error": ("background:#fee2e2;border:1px solid #fca5a5;color:#991b1b;", "✖"),
        "idle":  ("background:#f1f5f9;border:1px solid #e2e8f0;color:#64748b;",  "·"),
    }
    style, icon = styles.get(kind, styles["idle"])
    return (
        f'<div style="{style}border-radius:8px;padding:7px 12px;'
        f'font-size:0.83em;line-height:1.4;display:flex;align-items:center;gap:7px;">'
        f'<span style="font-size:0.95em">{icon}</span>{text}</div>'
    )


def on_ask_click(
    question: str,
    chatbot_history: list,
    vs: Any,
    model_choice_val: str,
    temp_val: float,
    polish_val: bool,
    k_val: int,
    voice_val: bool,
):
    """Streaming generator: yields (cleared_msg, updated_chatbot, status)."""
    chatbot_history = list(chatbot_history or [])

    if not question.strip():
        yield "", chatbot_history, "⚠ Please enter a question."
        return

    if vs is None:
        yield "", chatbot_history, "⚠ Please upload a document first."
        return

    use_openai = openai_model if model_choice_val == "OpenAI" else None
    use_openai_name = openai_model_name if use_openai else None
    use_ollama = ollama_model if model_choice_val != "OpenAI" else None
    use_ollama_name = ollama_model_name if use_ollama else None

    if use_openai is None and use_ollama is None:
        yield "", chatbot_history, (
            "❌ Selected model is not available. "
            "Set OPENAI_API_KEY in .env or start an Ollama model."
        )
        return

    faiss_store = vs.get("faiss") if isinstance(vs, dict) else vs
    bm25_index = vs.get("bm25") if isinstance(vs, dict) else None
    parent_store = vs.get("parents") if isinstance(vs, dict) else None
    qa_history = _extract_qa_history(chatbot_history)

    chatbot_history.append({"role": "user", "content": question})
    chatbot_history.append({"role": "assistant", "content": ""})
    yield "", chatbot_history, "Retrieving context…"

    partial = ""
    try:
        for token in stream_context_and_answer(
            vector_store=faiss_store,
            question=question,
            openai_model=use_openai,
            openai_model_name=use_openai_name,
            ollama_model=use_ollama,
            ollama_model_name=use_ollama_name,
            history=qa_history,
            top_k=int(k_val) if k_val else 4,
            temperature=float(temp_val) if temp_val is not None else None,
            bm25_index=bm25_index,
            parent_store=parent_store,
        ):
            partial += token
            chatbot_history[-1]["content"] = partial
            yield "", chatbot_history, None
    except Exception as e:
        chatbot_history[-1]["content"] = f"Error: {e}"
        yield "", chatbot_history, None
        return

    if polish_val and partial:
        try:
            polished = build_context_and_answer(
                vector_store=faiss_store,
                question=question,
                openai_model=use_openai,
                openai_model_name=use_openai_name,
                ollama_model=use_ollama,
                ollama_model_name=use_ollama_name,
                history=qa_history,
                top_k=int(k_val) if k_val else 4,
                temperature=float(temp_val) if temp_val is not None else None,
                polish=True,
                bm25_index=bm25_index,
                parent_store=parent_store,
            )
            if polished and len(polished.strip()) >= 20:
                partial = polished
                chatbot_history[-1]["content"] = polished
                yield "", chatbot_history, None
        except Exception:
            pass

    if voice_val and partial:
        _stop_voice()
        threading.Thread(target=_speak_async, args=(partial,), daemon=True).start()

    yield "", chatbot_history, ""


def on_clear() -> Tuple[list, str]:
    return [], ""


def on_voice_input() -> str:
    """Record 5 s of audio and return transcribed text."""
    try:
        default_in = sd.default.device[0]
        recording = sd.rec(
            int(5 * 44100), samplerate=44100, channels=1,
            dtype="float32", device=default_in,
        )
        sd.wait()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        sf.write(tmp_path, recording, 44100)
        recognizer = sr.Recognizer()
        with sr.AudioFile(tmp_path) as src:
            audio = recognizer.record(src)
        try:
            return recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            return "Could not understand audio."
        except sr.RequestError as e:
            return f"Recognition error: {e}"
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
    except Exception as e:
        return f"Voice input error: {e}"


# ── Styles ────────────────────────────────────────────────────────────────────

_CSS = """
/* ── Reset ──────────────────────────────────────────── */
footer { display: none !important; }
.gradio-container { padding-top: 0 !important; }

/* ── App header ─────────────────────────────────────── */
#app-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 55%, #1d4ed8 100%);
    border-radius: 14px;
    padding: 26px 32px 22px;
    margin-bottom: 18px;
    position: relative;
    overflow: hidden;
}
#app-header::after {
    content: '';
    position: absolute;
    top: -60px; right: -40px;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(96,165,250,0.18) 0%, transparent 65%);
    pointer-events: none;
}
#app-title {
    color: #fff;
    font-size: 1.75em;
    font-weight: 700;
    letter-spacing: -0.3px;
    margin: 0 0 5px;
}
#app-subtitle {
    color: rgba(255,255,255,0.7);
    font-size: 0.875em;
    margin: 0;
    line-height: 1.55;
}
.header-tags { margin-top: 14px; display: flex; gap: 7px; flex-wrap: wrap; }
.htag {
    background: rgba(255,255,255,0.11);
    border: 1px solid rgba(255,255,255,0.22);
    color: rgba(255,255,255,0.82);
    border-radius: 20px;
    padding: 2px 11px;
    font-size: 0.71em;
    font-weight: 500;
    letter-spacing: 0.04em;
}

/* ── LLM warning ─────────────────────────────────────── */
#llm-warning {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-left: 3px solid #f59e0b;
    border-radius: 9px;
    padding: 10px 15px;
    margin-bottom: 14px;
    font-size: 0.86em;
    color: #92400e;
    line-height: 1.5;
}

/* ── Sidebar section labels ──────────────────────────── */
.sec-label {
    font-size: 0.69em !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #94a3b8 !important;
    margin: 16px 0 6px !important;
    padding: 0 !important;
    line-height: 1 !important;
}
.first-label { margin-top: 4px !important; }

/* ── Upload button ───────────────────────────────────── */
#upload-btn button {
    width: 100%;
    border-radius: 9px !important;
    border: 1.5px dashed #cbd5e1 !important;
    background: #f8fafc !important;
    color: #475569 !important;
    font-weight: 500 !important;
    font-size: 0.88em !important;
    padding: 10px !important;
    transition: border-color 0.18s, background 0.18s, color 0.18s !important;
}
#upload-btn button:hover {
    border-color: #60a5fa !important;
    background: #eff6ff !important;
    color: #1d4ed8 !important;
}
#file-types-hint {
    font-size: 0.74em;
    color: #94a3b8;
    text-align: center;
    margin-top: 4px;
    margin-bottom: 2px;
}

/* ── Upload status ───────────────────────────────────── */
#upload-status { margin-top: 6px; }
#upload-status > div { margin: 0; padding: 0; }

/* ── Model radio ─────────────────────────────────────── */
#model-radio .wrap { gap: 6px; }
#model-radio label span { font-size: 0.88em !important; }
#model-radio .info-text { font-size: 0.76em !important; color: #94a3b8 !important; }

/* ── Advanced accordion ──────────────────────────────── */
#adv-accordion > .label-wrap button {
    font-size: 0.8em !important;
    font-weight: 600 !important;
    color: #475569 !important;
}
#adv-accordion .block { padding: 10px 0 0 !important; }
#adv-accordion label { font-size: 0.85em !important; color: #475569 !important; }
#adv-accordion .info { font-size: 0.74em !important; color: #94a3b8 !important; }

/* ── Voice section ───────────────────────────────────── */
#voice-check label { font-size: 0.88em !important; color: #475569 !important; }
#voice-row { margin-top: 6px !important; gap: 6px !important; }
#voice-row button {
    font-size: 0.8em !important;
    border-radius: 7px !important;
    padding: 6px 10px !important;
}

/* ── Chatbot ─────────────────────────────────────────── */
#chatbot {
    border-radius: 12px !important;
    border: 1px solid #e2e8f0 !important;
    box-shadow: 0 2px 8px rgba(15,23,42,0.06) !important;
}

/* ── Message input wrapper ───────────────────────────── */
#input-wrap {
    margin-top: 10px;
    background: #fff;
    border: 1.5px solid #e2e8f0;
    border-radius: 13px;
    padding: 10px 12px 8px;
    box-shadow: 0 1px 4px rgba(15,23,42,0.05);
    transition: border-color 0.15s, box-shadow 0.15s;
}
#input-wrap:focus-within {
    border-color: #93c5fd;
    box-shadow: 0 0 0 3px rgba(147,197,253,0.2);
}
#msg-input { background: transparent !important; }
#msg-input label, #msg-input .wrap-inner { padding: 0 !important; }
#msg-input textarea {
    border: none !important;
    box-shadow: none !important;
    padding: 2px 0 !important;
    font-size: 0.93em !important;
    background: transparent !important;
    resize: none !important;
}

/* ── Action buttons ──────────────────────────────────── */
#btn-row { margin-top: 6px !important; gap: 7px !important; }
#send-btn {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.88em !important;
    letter-spacing: 0.01em !important;
}
#clear-btn {
    border-radius: 8px !important;
    font-size: 0.85em !important;
}
#voice-input-btn {
    border-radius: 8px !important;
    font-size: 0.85em !important;
    min-width: 42px !important;
    padding: 0 10px !important;
}

/* ── Status bar ──────────────────────────────────────── */
#status-bar {
    text-align: center;
    min-height: 1.5em;
    margin-top: 4px;
}
#status-bar p { font-size: 0.79em !important; color: #94a3b8 !important; margin: 0 !important; }

/* ── Example chips ───────────────────────────────────── */
#example-chips .examples-holder { display: flex !important; flex-wrap: wrap !important; gap: 6px !important; }
#example-chips .examples-holder button {
    border-radius: 20px !important;
    font-size: 0.78em !important;
    padding: 4px 13px !important;
    background: #f1f5f9 !important;
    border: 1px solid #e2e8f0 !important;
    color: #475569 !important;
    transition: all 0.15s !important;
    white-space: nowrap !important;
}
#example-chips .examples-holder button:hover {
    background: #eff6ff !important;
    border-color: #93c5fd !important;
    color: #1d4ed8 !important;
}
#example-chips .label-wrap { display: none !important; }

/* ── Sidebar divider ─────────────────────────────────── */
.sidebar-divider {
    border: none;
    border-top: 1px solid #f1f5f9;
    margin: 4px 0 0;
}
"""

# ── HTML fragments ────────────────────────────────────────────────────────────

_HEADER_HTML = """
<div id="app-header">
  <div id="app-title">📄 DocuSense Pro</div>
  <p id="app-subtitle">
    Upload a document and ask questions in plain English.<br>
    Answers are grounded in your file with source citations.
  </p>
  <div class="header-tags">
    <span class="htag">PDF</span>
    <span class="htag">CSV / Excel</span>
    <span class="htag">TXT</span>
    <span class="htag">Semantic Search</span>
    <span class="htag">Hybrid Retrieval</span>
    <span class="htag">Voice I/O</span>
  </div>
</div>
"""

_LLM_WARNING_HTML = """
<div id="llm-warning">
  ⚠ <strong>No LLM configured.</strong>
  Add <code>OPENAI_API_KEY</code> to a <code>.env</code> file,
  or start Ollama (<code>ollama run llama3.1</code>).
  Until then, responses show raw retrieved context only.
</div>
"""

_FILE_HINT_HTML = '<p id="file-types-hint">PDF · CSV · Excel · TXT</p>'


# ── Interface ─────────────────────────────────────────────────────────────────

def build_interface() -> gr.Blocks:
    with gr.Blocks(title="DocuSense Pro") as demo:

        gr.HTML(_HEADER_HTML)

        if openai_model is None and ollama_model is None:
            gr.HTML(_LLM_WARNING_HTML)

        with gr.Row(equal_height=False):

            # ── Sidebar ───────────────────────────────────────────────────────
            with gr.Column(scale=3, min_width=240):

                gr.HTML('<p class="sec-label first-label">Document</p>')
                file_upload = gr.UploadButton(
                    "⬆  Upload Document",
                    file_types=[".txt", ".pdf", ".csv", ".xls", ".xlsx"],
                    variant="secondary",
                    size="sm",
                    elem_id="upload-btn",
                )
                gr.HTML(_FILE_HINT_HTML)
                upload_status = gr.HTML(
                    _status_html("idle", "No document loaded yet."),
                    elem_id="upload-status",
                )

                gr.HTML('<hr class="sidebar-divider"><p class="sec-label">Model</p>')
                model_choice = gr.Radio(
                    choices=["OpenAI", "Ollama (local)"],
                    value="Ollama (local)",
                    label="",
                    info="OpenAI requires OPENAI_API_KEY in .env",
                    elem_id="model-radio",
                )

                gr.HTML('<hr class="sidebar-divider">')
                with gr.Accordion(
                    "⚙  Advanced Settings",
                    open=False,
                    elem_id="adv-accordion",
                ):
                    temperature = gr.Slider(
                        0.0, 1.0, value=0.7, step=0.1,
                        label="Temperature",
                        info="Higher = creative · Lower = factual",
                    )
                    k_slider = gr.Slider(
                        1, 10, value=4, step=1,
                        label="Retrieved chunks (K)",
                        info="More chunks = richer context",
                    )
                    polish_checkbox = gr.Checkbox(
                        label="Polish answer  (second LLM pass, ~2× slower)",
                        value=False,
                    )

                gr.HTML('<hr class="sidebar-divider"><p class="sec-label">Voice</p>')
                voice_output = gr.Checkbox(
                    label="Read responses aloud",
                    value=False,
                    elem_id="voice-check",
                )
                with gr.Row(elem_id="voice-row"):
                    voice_btn  = gr.Button("🎤", size="sm", elem_id="voice-input-btn", scale=1)
                    stop_btn   = gr.Button("⏹ Stop", size="sm", variant="stop", scale=2)

            # ── Chat area ─────────────────────────────────────────────────────
            with gr.Column(scale=7):
                chatbot = gr.Chatbot(
                    height=520,
                    layout="bubble",
                    buttons=["copy", "copy_all"],
                    placeholder=(
                        "### Welcome to DocuSense Pro\n\n"
                        "**Step 1 —** Upload a document using the panel on the left.\n\n"
                        "**Step 2 —** Type your question below and press **Send**.\n\n"
                        "Supported formats: PDF · CSV · Excel · TXT"
                    ),
                    show_label=False,
                    elem_id="chatbot",
                )

                with gr.Group(elem_id="input-wrap"):
                    msg = gr.Textbox(
                        placeholder="Ask a question about your document…",
                        lines=2,
                        max_lines=6,
                        show_label=False,
                        container=False,
                        elem_id="msg-input",
                    )
                    with gr.Row(elem_id="btn-row"):
                        submit_btn = gr.Button(
                            "Send  ➤", variant="primary", scale=5,
                            elem_id="send-btn",
                        )
                        clear_btn = gr.Button(
                            "🗑  Clear", scale=1,
                            elem_id="clear-btn",
                        )

                gr.Examples(
                    examples=[
                        "Summarize the key points from this document.",
                        "What is the main topic or purpose of this document?",
                        "What are the most important findings or conclusions?",
                        "List all names, dates, or figures mentioned.",
                    ],
                    inputs=msg,
                    label="",
                    elem_id="example-chips",
                )

                status = gr.Markdown("", elem_id="status-bar")

        # ── State ─────────────────────────────────────────────────────────────
        vector_store = gr.State()

        # ── Wiring ────────────────────────────────────────────────────────────
        _ask_in  = [msg, chatbot, vector_store,
                    model_choice, temperature, polish_checkbox, k_slider, voice_output]
        _ask_out = [msg, chatbot, status]

        submit_btn.click(fn=on_ask_click, inputs=_ask_in, outputs=_ask_out)
        msg.submit   (fn=on_ask_click, inputs=_ask_in, outputs=_ask_out)

        file_upload.upload(
            fn=handle_file_upload,
            inputs=[file_upload],
            outputs=[vector_store, upload_status],
        )

        clear_btn.click(fn=on_clear, outputs=[chatbot, status])
        voice_btn.click(fn=on_voice_input, outputs=[msg])
        stop_btn.click (fn=_stop_voice, outputs=[], queue=False)

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────

def _handle_signal(signum, frame):
    cleanup_resources()
    sys.exit(0)


if __name__ == "__main__":
    atexit.register(cleanup_resources)
    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        demo = build_interface()
        _user = os.getenv("GRADIO_USERNAME")
        _pass = os.getenv("GRADIO_PASSWORD")
        auth  = (_user, _pass) if _user and _pass else None
        demo.launch(
            share=False,
            debug=False,
            show_error=True,
            auth=auth,
            max_threads=4,
            theme=gr.themes.Soft(
                primary_hue="blue",
                secondary_hue="slate",
                neutral_hue="slate",
                font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"],
            ),
            css=_CSS,
        )
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        print(f"Error launching app: {e}")
        sys.exit(1)
