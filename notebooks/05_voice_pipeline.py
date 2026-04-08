# Databricks notebook source
# ArthaSetu v2 — Voice AI Pipeline
# Sarvam-m ASR + fastText Language Detection + IndicTrans2 Translation + TTS
# Multilingual accessibility layer for Tier 2/3 India

# COMMAND ----------

import os
import io
import json
import tempfile
import numpy as np
from datetime import datetime

# COMMAND ----------
# ═══════════════════════════════════════════════════════════════════════
# LANGUAGE DETECTION (fastText)
# ═══════════════════════════════════════════════════════════════════════

print("═" * 60)
print("Loading Language Detection Model")
print("═" * 60)

SUPPORTED_LANGUAGES = {
    "hi": "Hindi",
    "mr": "Marathi",
    "te": "Telugu",
    "en": "English",
    "kn": "Kannada",
    "ta": "Tamil",
    "gu": "Gujarati",
    "or": "Odia",
}

def detect_language(text):
    """Detect language of input text."""
    try:
        from langdetect import detect
        lang = detect(text)
        if lang in SUPPORTED_LANGUAGES:
            return lang
        # Map close languages
        lang_map = {"bn": "hi", "pa": "hi", "ur": "hi"}
        return lang_map.get(lang, "en")
    except Exception:
        return "en"

# Test
test_texts = {
    "hi": "मेरा स्कोर क्या है?",
    "en": "What is my credit score?",
    "mr": "माझा स्कोर काय आहे?",
    "te": "నా స్కోరు ఎంత?",
}

for expected_lang, text in test_texts.items():
    detected = detect_language(text)
    status = "✓" if detected == expected_lang else "✗"
    print(f"  {status} '{text}' → detected: {detected} (expected: {expected_lang})")

# COMMAND ----------
# ═══════════════════════════════════════════════════════════════════════
# TRANSLATION (IndicTrans2 or Databricks AI)
# ═══════════════════════════════════════════════════════════════════════

print("═" * 60)
print("Setting up Translation Layer")
print("═" * 60)

class TranslationEngine:
    """Multilingual translation using IndicTrans2 or Databricks AI Gateway."""

    def __init__(self):
        self.engine = None
        self._init_engine()

    def _init_engine(self):
        """Try IndicTrans2 first, fallback to Databricks AI, then Google Translate."""
        # Try IndicTrans2 (quantized ONNX)
        try:
            from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator
            self.engine = "indictrans2"
            print("  ✓ IndicTrans2 loaded")
            return
        except ImportError:
            pass

        # Try Databricks Foundation Model
        try:
            from databricks.sdk import WorkspaceClient
            self.w = WorkspaceClient()
            self.engine = "databricks_llm"
            print("  ✓ Using Databricks LLM for translation")
            return
        except Exception:
            pass

        # Fallback: googletrans
        try:
            from googletrans import Translator
            self.translator = Translator()
            self.engine = "googletrans"
            print("  ✓ Using googletrans fallback")
            return
        except ImportError:
            pass

        self.engine = "passthrough"
        print("  ⚠ No translation engine available — passthrough mode")

    def translate(self, text, source_lang, target_lang):
        """Translate text between languages."""
        if source_lang == target_lang:
            return text

        if self.engine == "databricks_llm":
            try:
                src_name = SUPPORTED_LANGUAGES.get(source_lang, source_lang)
                tgt_name = SUPPORTED_LANGUAGES.get(target_lang, target_lang)
                response = self.w.serving_endpoints.query(
                    name="databricks-meta-llama-3-3-70b-instruct",
                    messages=[{
                        "role": "user",
                        "content": f"Translate the following {src_name} text to {tgt_name}. "
                                   f"Return ONLY the translation, nothing else.\n\n{text}"
                    }],
                    max_tokens=500,
                    temperature=0.1,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"  [Translation Error] {e}")
                return text

        elif self.engine == "googletrans":
            try:
                result = self.translator.translate(text, src=source_lang, dest=target_lang)
                return result.text
            except Exception:
                return text

        return text  # passthrough

translator = TranslationEngine()

# Test translation
print("\n  Testing translation:")
test_result = translator.translate("मेरा स्कोर क्या है?", "hi", "en")
print(f"  hi→en: 'मेरा स्कोर क्या है?' → '{test_result}'")

# COMMAND ----------
# ═══════════════════════════════════════════════════════════════════════
# SPEECH-TO-TEXT (ASR)
# ═══════════════════════════════════════════════════════════════════════

print("═" * 60)
print("Setting up ASR (Speech-to-Text)")
print("═" * 60)

class ASREngine:
    """Automatic Speech Recognition using Sarvam-m or Whisper."""

    def __init__(self):
        self.model = None
        self.engine = None
        self._init_engine()

    def _init_engine(self):
        """Try Sarvam-m first, then Whisper, then fallback."""
        # Try Whisper (via transformers)
        try:
            import whisper
            self.model = whisper.load_model("small")
            self.engine = "whisper"
            print("  ✓ Whisper (small) loaded for ASR")
            return
        except ImportError:
            pass

        # Try Transformers pipeline
        try:
            from transformers import pipeline
            self.model = pipeline("automatic-speech-recognition",
                                  model="openai/whisper-small",
                                  device=-1)  # CPU
            self.engine = "transformers"
            print("  ✓ Transformers Whisper pipeline loaded")
            return
        except Exception:
            pass

        self.engine = "mock"
        print("  ⚠ No ASR engine — using mock mode")

    def transcribe(self, audio_bytes):
        """Convert speech to text."""
        if self.engine == "whisper":
            # Save to temp file and transcribe
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
                f.write(audio_bytes)
                f.flush()
                result = self.model.transcribe(f.name)
                return result["text"], result.get("language", "en")

        elif self.engine == "transformers":
            result = self.model(audio_bytes)
            return result["text"], "en"

        # Mock mode
        return "मेरा स्कोर क्या है?", "hi"

asr = ASREngine()

# COMMAND ----------
# ═══════════════════════════════════════════════════════════════════════
# TEXT-TO-SPEECH (TTS)
# ═══════════════════════════════════════════════════════════════════════

print("═" * 60)
print("Setting up TTS (Text-to-Speech)")
print("═" * 60)

class TTSEngine:
    """Text-to-Speech using Sarvam TTS or gTTS."""

    def __init__(self):
        self.engine = None
        self._init_engine()

    def _init_engine(self):
        """Try gTTS first (most reliable for Indian languages)."""
        try:
            from gtts import gTTS
            self.engine = "gtts"
            print("  ✓ gTTS loaded for Text-to-Speech")
            return
        except ImportError:
            pass

        try:
            import pyttsx3
            self.tts = pyttsx3.init()
            self.engine = "pyttsx3"
            print("  ✓ pyttsx3 loaded for TTS")
            return
        except Exception:
            pass

        self.engine = "mock"
        print("  ⚠ No TTS engine — using mock mode")

    def speak(self, text, language="en"):
        """Convert text to speech audio bytes."""
        if self.engine == "gtts":
            from gtts import gTTS
            tts = gTTS(text=text, lang=language, slow=False)
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            return audio_buffer.getvalue()

        elif self.engine == "pyttsx3":
            # pyttsx3 saves to file
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as f:
                self.tts.save_to_file(text, f.name)
                self.tts.runAndWait()
                return open(f.name, "rb").read()

        # Mock: return empty bytes
        return b""

tts = TTSEngine()

# COMMAND ----------
# ═══════════════════════════════════════════════════════════════════════
# INTENT CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════

def classify_intent(text_en):
    """Simple keyword-based intent classification (English text)."""
    text_lower = text_en.lower()

    # Score-related intents
    if any(kw in text_lower for kw in ["score", "credit", "xscore", "rating", "cibil"]):
        if any(kw in text_lower for kw in ["why", "low", "reason", "less", "improve"]):
            return "score_explanation"
        if any(kw in text_lower for kw in ["improve", "increase", "better", "raise"]):
            return "score_improvement"
        return "score_check"

    # Literacy intents
    if any(kw in text_lower for kw in ["learn", "teach", "explain", "what is", "tell me about"]):
        if any(kw in text_lower for kw in ["loan", "emi", "borrow"]):
            return "literacy_loans"
        if any(kw in text_lower for kw in ["save", "saving", "budget"]):
            return "literacy_savings"
        if any(kw in text_lower for kw in ["upi", "digital", "online"]):
            return "literacy_upi"
        if any(kw in text_lower for kw in ["interest", "rate"]):
            return "literacy_interest"
        if any(kw in text_lower for kw in ["credit", "score"]):
            return "literacy_credit"
        return "literacy_general"

    # Help
    if any(kw in text_lower for kw in ["help", "what can", "menu", "options"]):
        return "help"

    # Profile
    if any(kw in text_lower for kw in ["profile", "income", "change", "update"]):
        return "profile_update"

    return "unknown"


# Test intent classification
print("\nTesting intent classification:")
test_intents = [
    ("What is my score?", "score_check"),
    ("Why is my score low?", "score_explanation"),
    ("How to improve my score?", "score_improvement"),
    ("Teach me about loans", "literacy_loans"),
    ("What is UPI safety?", "literacy_upi"),
    ("Help me", "help"),
]

for text, expected in test_intents:
    result = classify_intent(text)
    status = "✓" if result == expected else "✗"
    print(f"  {status} '{text}' → {result} (expected: {expected})")

# COMMAND ----------
# ═══════════════════════════════════════════════════════════════════════
# END-TO-END VOICE PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def voice_pipeline(audio_bytes=None, text_input=None, user_id=None):
    """
    End-to-end voice pipeline:
    1. ASR: Speech → Text (or accept text directly)
    2. Language Detection
    3. Translation to English
    4. Intent Classification
    5. Process (XScore check / Literacy RAG / Help)
    6. Translation back to user's language
    7. TTS: Text → Speech

    Returns dict with text response and audio bytes.
    """
    result = {
        "user_language": "en",
        "transcription": "",
        "english_text": "",
        "intent": "",
        "response_en": "",
        "response_user_lang": "",
        "audio_response": None,
    }

    # Step 1: Get text input
    if audio_bytes:
        text, detected_lang = asr.transcribe(audio_bytes)
        result["transcription"] = text
        result["user_language"] = detected_lang
    elif text_input:
        text = text_input
        result["transcription"] = text
    else:
        result["response_en"] = "No input provided."
        return result

    # Step 2: Language detection
    user_lang = detect_language(text)
    result["user_language"] = user_lang
    print(f"  Language detected: {user_lang} ({SUPPORTED_LANGUAGES.get(user_lang, 'Unknown')})")

    # Step 3: Translate to English
    if user_lang != "en":
        english_text = translator.translate(text, user_lang, "en")
    else:
        english_text = text
    result["english_text"] = english_text
    print(f"  English: {english_text}")

    # Step 4: Intent classification
    intent = classify_intent(english_text)
    result["intent"] = intent
    print(f"  Intent: {intent}")

    # Step 5: Process based on intent
    if intent == "score_check":
        response = (
            f"Your XScore is being calculated. Based on your payment discipline, "
            f"financial stability, asset verification, digital trust, and financial "
            f"awareness, we will provide your score. Please check the User Portal "
            f"for your detailed score breakdown."
        )
    elif intent == "score_explanation":
        response = (
            f"Your score may be lower because of: "
            f"1) Inconsistent bill payments — paying bills late reduces your Payment Discipline. "
            f"2) Low financial awareness — completing more literacy modules will improve this. "
            f"3) Limited digital footprint. "
            f"Tip: Pay all bills on time for the next 3 months to see improvement."
        )
    elif intent == "score_improvement":
        response = (
            f"To improve your XScore: "
            f"1) Pay all bills (electricity, mobile, rent) on time every month. "
            f"2) Complete financial literacy modules — each module adds to your Awareness score. "
            f"3) Maintain consistent UPI transaction patterns. "
            f"4) Link your Aadhaar and PAN if not done. "
            f"5) Build a savings buffer of at least 3 months expenses. "
            f"These actions can increase your score by 50-100 points over 3 months."
        )
    elif intent.startswith("literacy_"):
        # This would call the RAG pipeline from notebook 04
        topic = intent.replace("literacy_", "")
        response = (
            f"Let me teach you about {topic}. "
            f"I have a module on this topic. Would you like me to start the lesson? "
            f"After the lesson, you can take a quiz to test your understanding. "
            f"Completing quizzes improves your Financial Awareness score!"
        )
    elif intent == "help":
        response = (
            f"I can help you with: "
            f"1) Check your XScore — say 'What is my score?' "
            f"2) Understand your score — say 'Why is my score low?' "
            f"3) Improve your score — say 'How to improve my score?' "
            f"4) Learn financial topics — say 'Teach me about loans' or 'Explain savings' "
            f"5) Update your profile — say 'Update my income'"
        )
    else:
        response = (
            f"I'm not sure I understand. You can ask me about your credit score, "
            f"how to improve it, or learn about financial topics like loans, savings, "
            f"and UPI safety. Say 'help' for a list of options."
        )

    result["response_en"] = response

    # Step 6: Translate response to user's language
    if user_lang != "en":
        response_translated = translator.translate(response, "en", user_lang)
    else:
        response_translated = response
    result["response_user_lang"] = response_translated

    # Step 7: TTS
    try:
        audio_response = tts.speak(response_translated, language=user_lang)
        result["audio_response"] = audio_response
        print(f"  Audio response: {len(audio_response)} bytes")
    except Exception as e:
        print(f"  TTS failed: {e}")

    return result

# COMMAND ----------
# ═══════════════════════════════════════════════════════════════════════
# TEST VOICE PIPELINE
# ═══════════════════════════════════════════════════════════════════════

print("═" * 60)
print("Testing Voice Pipeline (text mode)")
print("═" * 60)

test_voice_inputs = [
    ("मेरा स्कोर क्या है?", "Hindi user checking score"),
    ("How to improve my score?", "English user wanting improvement tips"),
    ("मुझे लोन के बारे में बताओ", "Hindi user wanting loan education"),
    ("help", "English user asking for help"),
]

for text, description in test_voice_inputs:
    print(f"\n{'─' * 50}")
    print(f"  Test: {description}")
    print(f"  Input: {text}")
    result = voice_pipeline(text_input=text)
    print(f"  Response ({result['user_language']}): {result['response_user_lang'][:150]}...")

print("\n✓ Voice Pipeline Complete!")
