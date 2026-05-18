import os
import requests
import pyperclip
import gradio as gr
from dotenv import load_dotenv
from langdetect import detect as detect_lang


# Load the HuggingFace token from the .env file
load_dotenv("HF_TOKEN.env")
HF_TOKEN = os.getenv("HF_TOKEN")

# Headers sent with every API request for authentication
HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

# Base URL for Helsinki-NLP translation models on HuggingFace
BASE_URL = "https://router.huggingface.co/hf-inference/models/Helsinki-NLP"

# Translation models: any language → English
TO_EN = {
    "ar": "opus-mt-ar-en",
    "fr": "opus-mt-fr-en",
    "es": "opus-mt-es-en",
    "de": "opus-mt-de-en",
    "it": "opus-mt-it-en",
    "pt": "opus-mt-pt-en",
    "ru": "opus-mt-ru-en",
    "zh": "opus-mt-zh-en",
    "ja": "opus-mt-ja-en",
    "tr": "opus-mt-tr-en",
    "nl": "opus-mt-nl-en",
}

# Translation models: English → any language
FROM_EN = {
    "ar": "opus-mt-en-ar",
    "fr": "opus-mt-en-fr",
    "es": "opus-mt-en-es",
    "de": "opus-mt-en-de",
    "it": "opus-mt-en-it",
    "pt": "opus-mt-en-pt",
    "ru": "opus-mt-en-ru",
    "zh": "opus-mt-zh",
    "ja": "opus-mt-en-jap",
    "tr": "opus-mt-en-tr",
    "nl": "opus-mt-en-nl",
}

# Human-readable language names mapped from language codes
LANG_NAMES = {
    "ar": "Arabic", "en": "English", "fr": "French",
    "es": "Spanish", "de": "German", "it": "Italian",
    "pt": "Portuguese", "ru": "Russian", "zh": "Chinese",
    "ja": "Japanese", "tr": "Turkish", "nl": "Dutch",
}

# Language options shown in the dropdown (display name, language code)
LANG_CHOICES = [
    ("Arabic",     "ar"),
    ("English",    "en"),
    ("French",     "fr"),
    ("Spanish",    "es"),
    ("German",     "de"),
    ("Italian",    "it"),
    ("Portuguese", "pt"),
    ("Russian",    "ru"),
    ("Chinese",    "zh"),
    ("Japanese",   "ja"),
    ("Turkish",    "tr"),
    ("Dutch",      "nl"),
]


def hf_translate(text, model_name):
    """Send text to HuggingFace API and return the translated result."""
    url = f"{BASE_URL}/{model_name}"
    res = requests.post(url, headers=HEADERS, json={"inputs": text})
    data = res.json()
    # If API returns an error, raise it
    if isinstance(data, dict) and "error" in data:
        raise Exception(data["error"])
    return data[0]["translation_text"]


def make_detected_html(lang="—"):
    """Build the HTML string that displays the detected language inline."""
    return f'<p style="margin:0; padding:4px 0; font-size:14px;"><b>Detected Language:</b> {lang}</p>'


def translate(text, target_lang):
    """
    Main translation logic:
    1. Detect the source language
    2. If source == target, return as-is
    3. If target is English, translate directly
    4. If source is English, translate directly
    5. Otherwise, translate through English as a pivot (src → en → target)
    """
    if not text.strip():
        return "", "", make_detected_html(), ""

    try:
        text = text.strip()
        # Capitalize the first letter
        text = text[0].upper() + text[1:]

        # Detect the language of the input text
        src_lang = detect_lang(text)
        if src_lang not in LANG_NAMES:
            src_lang = "en"  # Fallback to English if language is unsupported

        detected_name = LANG_NAMES.get(src_lang, src_lang)

        # No translation needed if source and target are the same
        if src_lang == target_lang:
            return text, text, make_detected_html(detected_name), f"✅ Text is already in {detected_name}"

        if target_lang == "en":
            # Translate from source language directly to English
            model = TO_EN.get(src_lang)
            if not model:
                return "", "", make_detected_html(detected_name), f"❌ No model found to translate from {detected_name} to English"
            result = hf_translate(text, model)

        elif src_lang == "en":
            # Translate from English directly to target language
            model = FROM_EN.get(target_lang)
            if not model:
                return "", "", make_detected_html(detected_name), "❌ No model found for this target language"
            result = hf_translate(text, model)

        else:
            # Pivot through English: source → English → target
            model_to_en = TO_EN.get(src_lang)
            model_to_target = FROM_EN.get(target_lang)
            if not model_to_en or not model_to_target:
                return "", "", make_detected_html(detected_name), "❌ No model found for this language pair"
            en_text = hf_translate(text, model_to_en)       # Step 1: src → English
            result = hf_translate(en_text, model_to_target) # Step 2: English → target

        target_name = LANG_NAMES.get(target_lang, target_lang)
        status = f"✅ Detected: {detected_name} → Translated to: {target_name}"
        return result, result, make_detected_html(detected_name), status

    except Exception as e:
        return "", "", make_detected_html(), f"❌ Error: {str(e)}"


def clear_all():
    """Reset all fields to their default empty state."""
    return "", "", make_detected_html(), "", "Cleared"


def copy_translation(text):
    """Copy the translated text to the clipboard."""
    if not text:
        return "⚠️ Nothing to copy"
    pyperclip.copy(text)
    return "✅ Copied to clipboard!"


# CSS to force equal height on both textboxes
css = """
.textbox-equal textarea {
    height: 200px !important;
    min-height: 200px !important;
}
"""

with gr.Blocks(title="Smart Translator", css=css) as app:
    gr.Markdown("# 🌐 Smart Translator\nAuto-detects input language and translates to your chosen language")

    # Stores the latest translation for the copy button
    translation_state = gr.State("")

    with gr.Row(equal_height=True):
        with gr.Column():
            # Shows the auto-detected language above the input box
            detected_lang_html = gr.HTML(make_detected_html())
            input_text = gr.Textbox(
                label="Input Text",
                placeholder="Type here...",
                lines=7,
                elem_classes="textbox-equal"
            )
        with gr.Column():
            # Dropdown to select the target language, placed above the output box
            target_lang = gr.Dropdown(
                choices=LANG_CHOICES,
                value="ar",
                label="Output Language"
            )
            output_text = gr.Textbox(
                label="Translation",
                placeholder="Translation will appear here...",
                lines=7,
                interactive=False,
                elem_classes="textbox-equal"
            )

    with gr.Row():
        translate_btn = gr.Button("Translate", variant="primary")

    with gr.Row():
        clear_btn = gr.Button("Clear", variant="secondary")
        copy_btn = gr.Button("Copy Translation", variant="secondary")

    status_box = gr.Textbox(label="Status", interactive=False)

    # Trigger translation automatically when input text changes
    input_text.change(
        fn=translate,
        inputs=[input_text, target_lang],
        outputs=[output_text, translation_state, detected_lang_html, status_box]
    )

    # Trigger translation automatically when target language changes
    target_lang.change(
        fn=translate,
        inputs=[input_text, target_lang],
        outputs=[output_text, translation_state, detected_lang_html, status_box]
    )

    # Trigger translation on button click
    translate_btn.click(
        fn=translate,
        inputs=[input_text, target_lang],
        outputs=[output_text, translation_state, detected_lang_html, status_box]
    )

    # Clear all fields
    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[input_text, output_text, detected_lang_html, translation_state, status_box]
    )

    # Copy translation to clipboard
    copy_btn.click(
        fn=copy_translation,
        inputs=[translation_state],
        outputs=[status_box]
    )

app.launch(theme=gr.themes.Soft())
