import requests
import time
import os
from smolagents.models import MessageRole, Model
from smolagents import (
    # HfApiModel,
    LiteLLMModel,
    Model,
)
import base64
import re
from dotenv import load_dotenv
import xml.etree.ElementTree as ET

load_dotenv(override=True)

DEFAULT_HTR_ID = 39995 

# === 配置参数 ===
USERNAME = os.getenv("TRANSKRIBUS_USERNAME")
PASSWORD = os.getenv("TRANSKRIBUS_PASSWORD")
HTR_ID = 37646
LINE_MODEL_ID = 49272
IMGBB_API_KEY = os.getenv("IMGBB_API_KEY")

AUTH_URL = "https://account.readcoop.eu/auth/realms/readcoop/protocol/openid-connect/token"
PROCESS_URL = "https://transkribus.eu/processing/v1/processes"

language_to_htr_ids = {
    "Danish": [38364, 47113, 49362, 51398, 48496, 40982, 48721],
    "Dutch": [50786, 37851, 16804, 48329],
    "English": [48327, 54749, 37646],
    "Estonian": [48361],
    "ethiopic": [48371],
    "Finnish": [48294, 48363],
    "French": [37758, 37839],
    "German": [36508, 50870, 26068, 19584],
    "Greek": [42105],
    "Italian": [38440],
    "Latin": [48237, 27337, 37839],
    "Russian": [45595, 46147],
    "Swedish": [20686],
    "Church Slavonic": [24509],
    "Portuguese": [44949],
    "Spanish": [47702],
    "Croatian": [48702],
    "Low German": [27337],
    "Polish": [44976],
    "Scottish Gaelic": [48445],
    "Norwegian": [49721],
    "Yiddish": [59324],
    "Hungarian": [46058],
    "Hindi": [45909],
    "Braj Bhasha": [45909],
    "Sanskrit": [45909],
    "Awadhi": [45909],
    "Flemish": [42143],
    "Czech": [66429],
    "Slovak": [39995],
    "Slovenian": [51128],
    "Indonesian": [48252],
    "Hebrew": [59324],
    "Ukrainian": [51906],
    "Old Occitan": [52822],
    "Old Provençal": [52822],
    "Romanized Hebrew": [52754],
    "Icelandic": [51788],
    "Tibetan": [54935, 54525]
}



# === Token 管理类 ===
class TokenManager:
    def __init__(self, username, password):
        self.username = username
        self.password = password
        self.client_id = "processing-api-client"
        self.token = None
        self.refresh_token = None
        self.expire_time = 0

    def _get_new_token(self):
        response = requests.post(AUTH_URL, data={
            'grant_type': 'password',
            'client_id': self.client_id,
            'username': self.username,
            'password': self.password
        })
        response.raise_for_status()
        data = response.json()
        self.token = data['access_token']
        self.refresh_token = data['refresh_token']
        self.expire_time = time.time() + data['expires_in'] - 10
        return self.token

    def _refresh_token(self):
        response = requests.post(AUTH_URL, data={
            'grant_type': 'refresh_token',
            'client_id': self.client_id,
            'refresh_token': self.refresh_token
        })
        response.raise_for_status()
        data = response.json()
        self.token = data['access_token']
        self.refresh_token = data['refresh_token']
        self.expire_time = time.time() + data['expires_in'] - 10
        return self.token

    def get_token(self):
        if self.token is None or time.time() >= self.expire_time:
            try:
                return self._refresh_token()
            except Exception:
                return self._get_new_token()
        return self.token


# === 上传本地图片到 ImgBB ===
def upload_to_imgbb(image_path, imgbb_api_key):
    url = "https://api.imgbb.com/1/upload"
    with open(image_path, "rb") as file:
        image_data = file.read()
    payload = {
        "key": imgbb_api_key
    }
    files = {
        "image": image_data
    }
    response = requests.post(url, data=payload, files=files)
    response.raise_for_status()
    result = response.json()
    return result["data"]["url"]

        
def detect_language_and_htr_id_from_image_path(image_path: str, model) -> int:
    """
    Use LLM to determine the language from an image and return the corresponding HTR model ID.
    If the model cannot determine the language or returns an abnormal format, use the default HTR ID.
    
    Parameters:
        image_path (str): Path to the image file
        model: LLM calling function

    Returns:
        int: Selected HTR model ID
    """

    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

    mapping_str = "\n".join(
        f"{lang}: {'; '.join(map(str, ids))}" for lang, ids in language_to_htr_ids.items()
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are an OCR model selection assistant for historical handwriting recognition. "
                "You will be shown an image of a historical manuscript. "
                "Your task is to identify the most likely language and choose the best matching HTR model ID "
                "from the provided list. You must return exactly one model ID (as a number). "
                "If none of the languages seem appropriate, return: 39995. Do not explain your answer."
            )
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"This is the list of languages and their corresponding HTR model IDs:\n\n"
                        f"{mapping_str}\n\n"
                        "Below is a historical document image. Please determine the most likely language "
                        "and select the best matching model ID.\n"
                        "Return only the model ID as a number, nothing else. "
                        "If nothing matches well, return 39995."
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}"
                    }
                }
            ]
        }
    ]

    response = model(messages).content.strip()
    
    match = re.search(r"\b\d{5,6}\b", response)
    if match:
        return int(match.group(0))
    else:
        print(f"[Warning] LLM cannot determine model ID, returning default value 39995; response content: {response}")
        return DEFAULT_HTR_ID


def submit_image(token_manager, image_url, htr_id):
    token = token_manager.get_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    payload = {
        "config": {
            "textRecognition": {
                "htrId": htr_id
            },
            "lineDetection": {
                "modelId": LINE_MODEL_ID
            }
        },
        "image": {
            "imageUrl": image_url
        }
    }
    response = requests.post(PROCESS_URL, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["processId"]

def wait_for_result(token_manager, process_id):
    while True:
        token = token_manager.get_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json"
        }
        url = f"{PROCESS_URL}/{process_id}"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        status = response.json()["status"]
        print(f"[Status] process {process_id}: {status}")
        if status == "FINISHED":
            return
        elif status in ["FAILED", "ERROR"]:
            raise RuntimeError(f"OCR failed for process {process_id}")
        time.sleep(3)

def get_page_result(token_manager, process_id):
    token = token_manager.get_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/xml"
    }
    url = f"{PROCESS_URL}/{process_id}/page"
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.text



def extract_text_from_page_xml(xml_string):
    root = ET.fromstring(xml_string)
    ns = {'ns': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}

    lines = []
    for text_line in root.findall(".//ns:TextLine", ns):
        line_text = ""
        for word in text_line.findall(".//ns:Word", ns):
            text_equiv = word.find("ns:TextEquiv/ns:Unicode", ns)
            if text_equiv is not None and text_equiv.text:
                line_text += text_equiv.text + " "
        if line_text.strip():
            lines.append(line_text.strip())
    return "\n".join(lines)

def optimize_translate_text_with_llm(text, model):
    response = model(
        [
            {
                "role": "system",
                "content": (
                    "You are a historical text restoration and translation assistant. "
                    "You will receive OCR-extracted text from damaged, ancient, or ritual manuscripts. "
                    "Your task has **three steps**:\n\n"
                    "1. **Optimize the Original Text**: Repair likely OCR errors, reconstruct broken words or phrases, "
                    "and make the text readable and coherent in its original language (likely Old Church Slavonic, Old Russian, Classical Hebrew, Tibetan, or another ancient script). "
                    "Keep the original poetic, religious, or ceremonial tone where applicable.\n\n"
                    "2. **Translate to English**: Provide a modern English translation of the optimized text. "
                    "Preserve the meaning and tone. If some parts are unclear, annotate them as uncertain.\n\n"
                    "3. **Confidence and Context**: For each of the above, assign a confidence score (0–100%) "
                    "and briefly explain the possible cultural, religious, or historical context based on your interpretation.\n\n"
                    "**If the text is too corrupted or fragmented to optimize meaningfully, you should instead provide a brief summary of what the text is likely discussing, "
                    "based on any recognizable structure, terminology, or thematic patterns.**\n\n"
                    "Return your output in this format:\n\n"
                    "🔹 Optimized Text (Confidence: XX%):\n<Optimized version or [Unrecoverable]>\n\n"
                    "🔹 English Translation (Confidence: XX%):\n<Translation or Summary>\n\n"
                    "🔹 Contextual Interpretation:\n<Explanation of possible cultural or religious significance or origin>"
                )
            },
            {
                "role": "user",
                "content": (
                    "Here is the OCR text I want you to process. Even if it is fragmented, do your best to optimize, translate, "
                    "and provide interpretive context. If the text is unrecoverable, give a summary of what it might be discussing:\n\n"
                    f"{text}"
                )
            }
        ]
    )
    optimized_text = response.content.strip()
    return optimized_text


def save_text_to_file(text, output_file_path):
    """
    Save extracted text to a TXT file
    
    Parameters:
        text (str): Text content to save
        output_file_path (str): Output file path
    """
    try:
        with open(output_file_path, 'w', encoding='utf-8') as file:
            file.write(text)
        print(f"Text successfully saved to: {output_file_path}")
    except Exception as e:
        print(f"Error saving file: {e}")

def transkribus_ocr(image_path, imgbb_api_key, model):
    token_manager = TokenManager(USERNAME, PASSWORD)
    print("[Step 1] Uploading to ImgBB...")
    image_url = upload_to_imgbb(image_path, imgbb_api_key)
    print(f"[Step 1] Success: {image_url}")
    print("[Step 2] LLM detecting language and mapping model...")
    htr_id = detect_language_and_htr_id_from_image_path(image_path, model)
    print(f"[Step 2] LLM returned HTR ID: {htr_id}")
    print("[Step 3] Submitting recognition task to Transkribus...")
    pid = submit_image(token_manager, image_url, htr_id=htr_id)
    print(f"[Step 3] Success, Process ID: {pid}")
    print("[Step 4] Waiting for recognition to complete...")
    wait_for_result(token_manager, pid)
    print("[Step 5] Downloading PAGE XML...")
    page_xml = get_page_result(token_manager, pid)
    print("[Step 6] Extracting text content...")
    plain_text = extract_text_from_page_xml(page_xml)
    optimized_text = optimize_translate_text_with_llm(plain_text, model)
    output_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ocr_results")
    base_filename = os.path.basename(image_path)
    txt_filename = os.path.splitext(base_filename)[0] + ".txt"
    txt_filepath = os.path.join(output_folder, txt_filename)
    os.makedirs(output_folder, exist_ok=True)  # Create directory automatically
    text_to_save = "Original: \n" + plain_text + "\n\n Optimized: \n" + optimized_text
    save_text_to_file(text_to_save, txt_filepath)
    return text_to_save
