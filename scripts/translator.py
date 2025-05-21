# Create in scripts/translator_agent.py
from smolagents import Tool
import re
import requests
from translate import Translator

class TranslatorTool(Tool):
    name = "Translator_Tool"
    description = "Translate text to different languages, supporting multiple translation services"
    inputs = {
        "text": {
            "type": "string",
            "description": "Text to be translated",
            "nullable": True
        },
        "target": {
            "type": "string",
            "description": "Target language code (e.g., 'zh' for Chinese)",
            "enum": ["en", "zh", "ja", "ko", "hy", "sa", "ar"],
            "default": "en",
            "nullable": True
        }
    }
    output_type = "string"

    def __init__(self, translation_url="http://127.0.0.1:5000/translate"):
        super().__init__()
        self.translation_url = translation_url
        self.language_patterns = {
            'hy': re.compile(r'[\u0530-\u058F]'),  # Armenian character range
            'sa': re.compile(r'[\u0900-\u097F]'),  # Sanskrit character range
            'ar': re.compile(r'[\u0600-\u06FF]'),  # Arabic character range
            'ja': re.compile(r'[\u3040-\u309F\u30A0-\u30FF]'),  # Japanese kana range
        }

    # Retain original _detect_language and _translate_with_local methods
    # and forward method implementation...
    def _detect_language(self, text: str) -> str:
        """Detect text language, prioritizing specific language patterns"""
        # First check if it matches specific language patterns
        for lang, pattern in self.language_patterns.items():
            if pattern.search(text):
                print(f"Detected language pattern: {lang}")
                return lang
                
        # Check if it contains Japanese features (sentence-ending particles, etc.)
        if any(marker in text for marker in ["です", "ます", "だ", "の", "ね", "よ", "か"]):
            print(f"Detected Japanese based on sentence-ending particles")
            return 'ja'
        
        # Specifically check for Japanese proper nouns and vocabulary
        japanese_terms = ["東京", "日本", "京都", "大阪", "慶應義塾", "天皇", "幕府", "侍", "将軍"]
        if any(term in text for term in japanese_terms):
            print(f"Detected Japanese based on proper nouns")
            return 'ja'
        
        return 'other'

    def _translate_with_local(self, text: str, source_lang: str, target: str) -> str:
        """Translate using local translation library"""
        try:
            print(f"Using local translation: {source_lang} -> {target}")
            
            # Create translator instance
            translator = Translator(from_lang=source_lang, to_lang=target)
            
            # Translate text
            translated_text = translator.translate(text)
            
            # Build result
            return f"""### Translation Result
Source Text ({source_lang}):
{text}

Translated Text ({target}):
{translated_text}
"""
        except Exception as e:
            print(f"Local translation error: {e}")
            return f"Local translation failed: {str(e)}"

    def forward(self, text: str = None, target: str = "zh") -> str:
        """
        Translate text to target language
        
        Parameters:
            text: Text to be translated
            target: Target language code
            
        Returns:
            Translation result
        """
        try:
            if not text:
                return "Error: Empty text"
                
            # Detect source language
            source_lang = self._detect_language(text)
            print(f"Detected source language: {source_lang}")
            
            # For Japanese and other special languages, use specific translation methods
            if source_lang in ['hy', 'sa', 'ar', 'ja']:
                return self._translate_with_local(text, source_lang, target)
                
            # For other languages, use original translation service
            response = requests.post(
                self.translation_url,
                json={
                    "q": text,
                    "source": "auto",
                    "target": target
                },
                headers={"Content-Type": "application/json"},
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                return f"""### Translation Result
Source Text:
{text}

Translated Text ({target}):
{result}
"""
            else:
                return f"Translation failed with status code: {response.status_code}"

        except requests.Timeout:
            return "Error: Translation service timeout"
        except Exception as e:
            return f"Translation error: {str(e)}"

