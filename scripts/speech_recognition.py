from openai import OpenAI
import io
import math
import os
from smolagents import Tool
from pydub import AudioSegment
import time
from smolagents.models import MessageRole, Model

class SpeechRecognitionTool(Tool):
    name = "Speech_Recognition_Tool"
    description = "Convert speech to text and optimize with LLM using OpenAI Whisper and GPT"
    inputs = {
        "audio_file": {
            "type": "string",
            "description": "Path to the audio file",
            "nullable": True
        }
    }
    output_type = "string"

    def __init__(self, model=None):
        super().__init__()
        try:
            print("Loading Whisper model...")
            print("Whisper model loaded successfully")
            self.TEMP_ROOT = "temp_audio"
            self.OUTPUT_ROOT = "speech_recognition_output"
            os.makedirs(self.OUTPUT_ROOT, exist_ok=True)
            
            # Store the LLM model
            self.model = model
            if not self.model:
                raise ValueError("LLM model is required for text optimization")
            
        except Exception as e:
            raise Exception(f"Failed to initialize models: {str(e)}")

    def _optimize_text_with_llm(self, text: str) -> str:
        """Use LLM to optimize the transcribed text"""
        try:
            print("Optimizing text with LLM...")
            
            messages = [
                {
                    "role": MessageRole.SYSTEM,
                    "content": [
                        {
                            "type": "text",
                            "text": f"""You are a speech-to-text optimization expert. 
                            Your task is to improve the transcribed text while maintaining its original meaning. You cannot omit any information and you cannot add any information.
                            Provide the output in the following format:

                            === Optimized Transcription ===
                            [optimized text]

                            === Summary ===
                            [brief summary]

                            === Key Points ===
                            - [key point 1]
                            - [key point 2]
                            - [etc.]"""
                        }
                    ],
                },
                {
                    "role": MessageRole.USER,
                    "content": [
                        {
                            "type": "text",
                            "text": f"Here is the transcribed text to optimize:\n\n{text}"
                        }
                    ],
                }
            ]
            
            # Get response from LLM
            response = self.model(messages).content
            return text, response 
            
        except Exception as e:
            print(f"LLM optimization failed: {str(e)}")
            return text, text 

    def _process_audio(self, audio_file: str) -> str:
        """Process audio file and return transcribed text"""
        try:
            print(f"Processing audio file: {audio_file}")
            
            file_path = audio_file
            chunk_size_mb = 24
            max_file_size_mb = 25
            model_name = "gpt-4o-transcribe"

            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            client = OpenAI()
            full_transcript = ""

            if file_size_mb <= max_file_size_mb:
                with open(file_path, "rb") as audio_file:
                    response = client.audio.transcriptions.create(
                        model=model_name,
                        file=audio_file
                    )
                    full_transcript = response.text.strip()
            else:
                print(f"[INFO] File size {file_size_mb:.2f}MB > {max_file_size_mb}MB, splitting into chunks...")
                file_ext = os.path.splitext(file_path)[1].lower()
                if file_ext == ".mp3":
                    audio = AudioSegment.from_mp3(file_path)
                else:
                    audio = AudioSegment.from_file(file_path)

                target_chunk_size_mb = chunk_size_mb
                num_chunks = math.ceil(file_size_mb / target_chunk_size_mb)
                chunk_length_ms = math.ceil(len(audio) / num_chunks)

                print(f"[INFO] Total chunks: {num_chunks}")

                for i in range(num_chunks):
                    chunk = audio[i * chunk_length_ms : (i + 1) * chunk_length_ms]

                    audio_bytes_io = io.BytesIO()
                    chunk.export(audio_bytes_io, format="mp3")
                    audio_bytes_io.name = "temp.mp3"
                    audio_bytes_io.seek(0)

                    print(f"  → Transcribing chunk {i + 1}/{num_chunks}...")

                    response = client.audio.transcriptions.create(
                        model=model_name,
                        file=audio_bytes_io
                    )
                    chunk_text = response.text.strip()
                    full_transcript += chunk_text + " "

            original_text, optimized_text = self._optimize_text_with_llm(full_transcript)
            
            return original_text, optimized_text
            
        except Exception as e:
            raise Exception(f"Audio processing failed: {str(e)}")

    def forward(self, audio_file: str = None) -> str:
        """Main processing function"""
        try:
            if audio_file is None:
                return "Error: No audio file provided"

            if not os.path.exists(audio_file):
                return f"Error: File not found: {audio_file}"
            
            # Process audio file
            start_time = time.time()
            original_text, optimized_text = self._process_audio(audio_file)
            processing_time = time.time() - start_time
            
            if original_text and optimized_text:
                output_filename = os.path.join(
                    self.OUTPUT_ROOT,
                    f"transcript_{os.path.basename(audio_file)}_{int(time.time())}.txt"
                )
                with open(output_filename, "w", encoding="utf-8") as f:
                    f.write("=== Original Transcription ===\n")
                    f.write(original_text)
                    f.write("\n\n")
                    f.write("=== LLM Optimized Result ===\n")
                    f.write(optimized_text)
                print(f"✓ Results saved to: {output_filename}")
            
            print(f"✓ Processing completed in {processing_time:.2f} seconds")
            
            return f"""=== Original Transcription ===
{original_text}

=== LLM Optimized Result ===
{optimized_text}"""
                
        except Exception as e:
            return f"Speech recognition error: {str(e)}"