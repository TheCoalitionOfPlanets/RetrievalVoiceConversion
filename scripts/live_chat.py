"""Live chat-style web UI: type text, hear it back in the trained voice.

Loads the model once at startup, so each turn after the first is fast.

    uv run python scripts/live_chat.py --port 7866
"""

import argparse
import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.services.conversion_service import ConversionService  # noqa: E402
from src.utils.config import load_inference_config  # noqa: E402
from src.utils.config import load_model_config  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("live_chat")

LIGHT_CSS = """
:root, .dark {
    --body-background-fill: #ffffff !important;
    --background-fill-primary: #ffffff !important;
    --background-fill-secondary: #f7f7f8 !important;
    --block-background-fill: #ffffff !important;
    --body-text-color: #111111 !important;
    --block-title-text-color: #111111 !important;
    --block-label-text-color: #111111 !important;
    --input-background-fill: #ffffff !important;
    --border-color-primary: #e5e5e5 !important;
}
body, .gradio-container { background: #ffffff !important; color: #111111 !important; }
.gradio-container code, .gradio-container pre {
    background: #f0f0f2 !important;
    color: #111111 !important;
    border: 1px solid #e0e0e0 !important;
}
.gradio-container input[type="text"],
.gradio-container textarea,
.gradio-container .input-box,
.gradio-container select {
    background: #ffffff !important;
    color: #111111 !important;
    border-color: #d0d0d0 !important;
}
.gradio-container .message,
.gradio-container .message-bubble-border,
.gradio-container .bubble,
.gradio-container .user,
.gradio-container .bot,
.gradio-container [class*="message"],
.gradio-container [class*="bubble"] {
    background: #f0f0f2 !important;
    color: #111111 !important;
}
.gradio-container .message *, .gradio-container .bubble * {
    color: #111111 !important;
}
"""

TTS_VOICES = [
    "en-US-AndrewNeural",
    "en-US-GuyNeural",
    "en-US-AriaNeural",
    "en-US-JennyNeural",
    "en-GB-RyanNeural",
    "en-GB-SoniaNeural",
]


def synthesize(text: str, voice: str) -> str:
    import edge_tts

    fd, path = tempfile.mkstemp(suffix=".wav", prefix="tts_")
    import os

    os.close(fd)

    async def run() -> None:
        await edge_tts.Communicate(text, voice).save(path)

    asyncio.run(run())

    return path


def build_app(service: ConversionService):
    import gradio as gr

    def respond(message, history, voice, pitch, index_rate):
        message = (message or "").strip()
        if not message:
            return history, "", None

        history = history + [{"role": "user", "content": message}]

        try:
            tts_path = synthesize(message, voice)
            result = service.convert_file(
                tts_path,
                tempfile.mktemp(suffix=".wav", prefix="chat_out_"),
                speaker_id=0,
                pitch_shift=int(pitch),
                index_rate=index_rate,
            )
        except Exception as error:
            logger.exception("Conversion failed")
            history = history + [
                {"role": "assistant", "content": f"Error: {error}"}
            ]
            return history, "", None

        history = history + [
            {"role": "assistant", "content": {"path": result.output_path}}
        ]

        return history, "", result.output_path

    with gr.Blocks(title="RetrievalVoiceConversion - Live Chat") as app:
        gr.Markdown("## Live chat with your trained voice")
        gr.Markdown(
            "**This voice was trained on jack's recordings** "
            "(RVC v2 generator, fine-tuned from the pretrained base — see "
            "`data/raw/jack/` and `data/checkpoints/G_latest.pth`)."
        )
        gr.Markdown(
            "Type a message, hit Enter — it's synthesized with TTS, then "
            "converted through your trained model."
        )

        chatbot = gr.Chatbot(height=500)
        now_playing = gr.Audio(label="Now playing", autoplay=True)

        with gr.Row():
            msg = gr.Textbox(
                label="Message",
                placeholder="Type something and press Enter...",
                scale=4,
            )
            send = gr.Button("Send", variant="primary", scale=1)

        with gr.Accordion("Settings", open=False):
            voice = gr.Dropdown(
                label="TTS voice", choices=TTS_VOICES, value=TTS_VOICES[0]
            )
            pitch = gr.Slider(
                label="Pitch shift (semitones)", minimum=-12, maximum=12, value=0, step=1
            )
            index_rate = gr.Slider(
                label="Index rate (timbre strength)",
                minimum=0,
                maximum=1,
                value=0.75,
                step=0.05,
            )

        clear = gr.Button("Clear chat")

        msg.submit(
            respond,
            [msg, chatbot, voice, pitch, index_rate],
            [chatbot, msg, now_playing],
        )
        send.click(
            respond,
            [msg, chatbot, voice, pitch, index_rate],
            [chatbot, msg, now_playing],
        )
        clear.click(lambda: [], None, chatbot)

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="Live chat voice conversion UI")
    parser.add_argument("--port", type=int, default=7866)
    parser.add_argument("--model-config", default="config/model.yaml")
    parser.add_argument("--inference-config", default="config/inference.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--index", default=None)
    parser.add_argument("--device", default=None)

    args = parser.parse_args()

    inference_config = load_inference_config(args.inference_config)

    if args.checkpoint:
        inference_config.checkpoint = args.checkpoint
    if args.index:
        inference_config.index.path = args.index
    if args.device:
        inference_config.device = args.device

    logger.info("Loading model (this happens once)...")
    service = ConversionService(
        load_model_config(args.model_config), inference_config
    )
    service.load()
    logger.info("Model ready.")

    import gradio as gr

    app = build_app(service)
    app.queue().launch(
        server_name="0.0.0.0",
        server_port=args.port,
        theme=gr.themes.Default(),
        css=LIGHT_CSS,
        js="() => { document.documentElement.classList.remove('dark'); }",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
