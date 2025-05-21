# Create in scripts/ocr_agent.py
from smolagents import Tool
from smolagents.models import MessageRole, Model
import os
import base64
import json
import requests
from typing import List, Dict, Any, Optional, Union
import requests
import json
import base64
import urllib
import requests
import os
import time
from smolagents.models import MessageRole, Model
from smolagents import (
    # HfApiModel,
    LiteLLMModel,
    Model,
)
import os
from .transkribus_ocr import transkribus_ocr

class OCRTool(Tool):
    name = "OCR_Tool"
    description = "OCR Tool to extract text from images"
    inputs = {
        "image_path": {
            "type": "string",
            "description": "Path to the image file to be processed",
            "nullable": True
        }
    }
    output_type = "string"

    def __init__(self, imgbb_api_key, openrouter_api_key, model):
        super().__init__()
        self.cn_model  = LiteLLMModel(
            model_id="openrouter/deepseek/deepseek-r1",
            api_key=openrouter_api_key,
            api_base="https://openrouter.ai/api/v1",
            custom_role_conversions={"tool-call": "assistant", "tool-response": "user"},
            max_completion_tokens=8192,
            drop_params=True,
        ) # LLM model for result optimization
        self.imgbb_api_key = imgbb_api_key
        self.model = model
        # TextIn OCR API credentials
        self._textin_app_id = os.getenv("textin_app_id")
        self._textin_secret_code = os.getenv("textin_secret_code")
        self._textin_url = 'https://api.textin.com/ai/service/v2/recognize/multipage'

    def _get_file_content(self, file_path):
        """Get file content"""
        with open(file_path, 'rb') as fp:
            return fp.read()

    def _textin_ocr(self, image_path: str) -> str:
        """Use TextIn OCR API for text recognition"""
        try:
            print(f"Processing image with TextIn OCR: {image_path}")
            headers = {
                'x-ti-app-id': self._textin_app_id,
                'x-ti-secret-code': self._textin_secret_code,
                'Content-Type': 'application/octet-stream'
            }
            
            image_data = self._get_file_content(image_path)
            response = requests.post(self._textin_url, data=image_data, headers=headers)
            
            if response.status_code != 200:
                print(f"TextIn OCR API call failed: {response.status_code} - {response.text}")
                return None
                
            return response.text
        except Exception as e:
            print(f"TextIn OCR processing failed: {str(e)}")
            return None

        
    def _optimize_with_llm(self, ocr_result) -> str:
        """Optimize OCR results using LLM"""
        try:
            messages = [
                {
                    "role": MessageRole.SYSTEM,
                    "content": [
                        {
                            "type": "text",
                            "text": "You are an OCR text optimization expert. Your task is to fix OCR recognition errors and provide accurate text. If it's an ancient book or historical document, please maintain the original language and style while making it semantically coherent."
                        }
                    ],
                },
                {
                    "role": MessageRole.USER,
                    "content": [
                        {
                            "type": "text",
                            "text": f"Below is the original text recognized by OCR, which may contain errors:\n\n{ocr_result}\n\nPlease provide only the optimized plain text without any additional content:"
                        }
                    ],
                }
            ]
            
            print("Optimizing OCR results with LLM")
            
            # Call the model
            response = self.cn_model(messages).content
            
            # Ensure non-empty response
            if response and response.strip():
                return response.strip()
            else:
                print("LLM response is empty, returning original text")
                return ocr_result
            
        except Exception as e:
            print(f"LLM optimization failed: {str(e)}")
            return ocr_result

    def _describe_image_with_llm(self, image_path: str) -> str:
        """Describe image content using LLM when OCR fails"""
        try:
            # Read image file and convert to base64
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Prepare messages
            messages = [
                {
                    "role": MessageRole.SYSTEM,
                    "content": [
                        {
                            "type": "text",
                            "text": "You are an image analysis expert. Please describe the content of the image in detail, paying special attention to any text, symbols, charts, or important visual elements."
                        }
                    ],
                },
                {
                    "role": MessageRole.USER,
                    "content": [
                        {
                            "type": "text",
                            "text": "Please describe the content of this image in detail, especially any visible text:"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            }
                        }
                    ],
                }
            ]
            
            # Call the model
            response = self.model(messages).content
            
            if response and response.strip():
                return f"[Image Description] {response.strip()}"
            else:
                return "Unable to analyze image content"
            
        except Exception as e:
            print(f"Image description failed: {str(e)}")
            return f"Unable to process image: {str(e)}"
        
    def _judge_language(self, image_path: str) -> str:
        """Determine image language"""
        try:
            # Read image file and convert to base64
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Prepare messages
            messages = [
                {
                    "role": MessageRole.SYSTEM,
                    "content": [
                        {
                            "type": "text",
                            "text": "You are a language identification expert. Please identify the language of the image. If the language is Chinese or Japanese, return 'T'. Otherwise, return 'F'."
                        }
                    ],
                },
                {
                    "role": MessageRole.USER,
                    "content": [
                        {
                            "type": "text",
                            "text": "Please identify the language of the image. If the language is Chinese or Japanese, return 'T'. Otherwise, return 'F'. Only return 'T' or 'F'."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            }
                        }
                    ],
                }
            ]
            
            response = self.model(messages).content.strip()
            print(f"Language detection result: {response}")
            if response.upper() == "T":
                return True
            else:
                return False
        except Exception as e:
            print(f"Language detection failed: {str(e)}")
            return False
        
        
    
    def forward(self, image_path: str = None) -> str:
        """Perform OCR recognition and return result text"""
        output_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ocr_results")
        os.makedirs(output_folder, exist_ok=True)
        try:
            if image_path is None:
                return "Error: No image file path provided"
                
            if not os.path.exists(image_path):
                return f"Error: File path does not exist: {image_path}"
                
            # Check file extension
            ext = os.path.splitext(image_path)[1].lower()
            if ext not in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
                return "Error: Unsupported image format. Supported formats include: JPG, JPEG, PNG, BMP, TIFF, WEBP"
            
            base_filename = os.path.basename(image_path)
            txt_filename = os.path.splitext(base_filename)[0] + ".txt"
            txt_filepath = os.path.join(output_folder, txt_filename)
            print(f"Starting language detection")
            if not self._judge_language(image_path):
                print(f"Starting Transkribus OCR")
                ocr_result = transkribus_ocr(image_path, self.imgbb_api_key, self.model)
                return ocr_result
            else:
                print(f"Starting TextIn OCR")
                ocr_result = self._textin_ocr(image_path)
            
            if not ocr_result:
                if self.model:
                    return self._describe_image_with_llm(image_path)
                return "OCR recognition failed, unable to extract text"
            
            # If model is available, try to optimize OCR results
            if self.cn_model:
                optimized_ocr_result = self._optimize_with_llm(ocr_result)
                with open(txt_filepath, "w", encoding="utf-8") as txt_file:
                    txt_file.write("original ocr result: " + ocr_result + "\n" + "optimized ocr result: " + optimized_ocr_result)
                return "original ocr result: " + ocr_result + "\n" + "optimized ocr result: " + optimized_ocr_result
            
            # Otherwise return OCR results directly
            return ocr_result
            
        except Exception as e:
            print(f"OCR processing failed: {str(e)}")
            # If model is available, try to describe the image using LLM
            if self.model:
                return self._describe_image_with_llm(image_path)
            return f"OCR processing failed: {str(e)}"
