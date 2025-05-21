import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
import time
import sys
import logging
from scripts.reverse_image import GoogleLensSearchTool
import re
import traceback
import numpy as np
import pandas as pd
import json
from pydantic import BaseModel, ValidationError
from typing import Literal
import os
import json
import copy
import math
import argparse
import asyncio
import numpy as np
from typing import Literal
from pydantic import BaseModel
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from datasets import load_dataset
from rich.console import Console
import smolagents.monitoring
print(smolagents.monitoring.AgentLogger)
from smolagents.monitoring import AgentLogger
from smolagents.monitoring import LogLevel



import datasets
from datasets import Dataset
from dotenv import load_dotenv
from huggingface_hub import login
from smolagents import CodeAgent
from smolagents.agents import ToolCallingAgent
from scripts.reformulator import prepare_response
from scripts.run_agents import (
    get_single_file_description,
    get_zip_description,
)
from scripts.text_inspector_tool import TextInspectorTool
from scripts.text_web_browser import (
    ArchiveSearchTool,
    FinderTool,
    FindNextTool,
    PageDownTool,
    PageUpTool,
    SearchInformationTool,
    SimpleTextBrowser,
    VisitTool,
)
from scripts.image_web_browser import (
    SimpleImageBrowser,
    SearchInformationTool_Image,
    VisitTool_Image,
    ArchiveSearchTool_Image,
    PageUpTool_Image,
    PageDownTool_Image,
    FinderTool_Image,
    FindNextTool_Image,
    SaveHTMLTool,
)
from scripts.file_processing import (
    FileProcessor,
    OCRTool,
    PDFTool,
    DOCXTool,
    XLSXTool,
    PPTXTool,
    ImageAnalysisTool,
)

from scripts.web_tools import (
    LiteratureSearchingTool,
    GeneralBrowserTool,
    RelevantLiteratureFinderTool,
    BookMatchExtractorTool,
    DirectGoogleBooksCrawlerTool,
    SpringerSearchTool,
    SpringerStructuredSearchTool,
    SpringerDownloadAndParseTool,
)
from scripts.LocalGoogleSearchTool import LocalGoogleSearchTool
from scripts.visual_qa import visualizer
from tqdm import tqdm

from smolagents import (
    LiteLLMModel,
    Model,
)

import openai
from smolagents.models import MessageRole
from dataset_loader import load_custom_dataset
from scripts.translator import TranslatorTool
from scripts.speech_recognition import SpeechRecognitionTool
from scripts.ocr import OCRTool
from scripts.frame_extract import VideoFrameExtractorTool
import os
import json
import copy
import math
import argparse
import asyncio
import numpy as np
from typing import Literal
from pydantic import BaseModel
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from datasets import load_dataset

client = AsyncOpenAI(timeout=300.0, max_retries=1, api_key=os.getenv("OPENAI_API_KEY"))

JUDGE_PROMPT = """You are a fair evaluator. Judge whether the following [response] to [question] is semantically consistent with the [correct_answer] below.  

[question]: {question}  

[response]: {response}  

[correct_answer]: {correct_answer}  

When you judge, consider only whether the core meaning and all necessary key points in the response match the correct answer.  Even if wording or format differs, treat equivalent semantics as correct. Treat missing key points or any substantive error or omission as incorrect. For numerical answers, a small rounding difference is acceptable. Tolerate substantive deviations from the correct answer. If the extracted_final_answer is a more specific instance of the correct_answer (for example, "Pieter Schenk II" vs "Pieter Schenk"), and it still contains the core string of the correct_answer, treat it as correct.

Please output exactly in the format and criteria specified below:  

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0|\%| and 100|\%| from [response]. Put 100 if there is no confidence score available."""

class ExtractedAnswer(BaseModel):
    extracted_final_answer: str
    reasoning: str
    correct: Literal["yes", "no"]
    confidence: int
    strict: Literal[True] # 100% reliability

AUTHORIZED_IMPORTS = [
    "requests",
    "zipfile",
    "os",
    "pandas",
    "numpy",
    "sympy",
    "json",
    "bs4",
    "pubchempy",
    "xml",
    "yahoo_finance",
    "Bio",
    "sklearn",
    "scipy",
    "pydub",
    "io",
    "PIL",
    "chess",
    "PyPDF2",
    "pptx",
    "torch",
    "datetime",
    "fractions",
    "csv",
]
load_dotenv(override=True)
login(os.getenv("HF_TOKEN"))

append_answer_lock = threading.Lock()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--model-id", type=str, default="gpt-4o")
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--api-key", type=str, help="OpenAI API key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--use-image-agent", action="store_true", help="Enable image information agent")
    parser.add_argument("--use-file-agent", action="store_true", help="Enable file processor agent")
    parser.add_argument("--use-literature-agent", action="store_true", help="Enable literature search agent")
    parser.add_argument("--no-text-webbrowser-agent", action="store_true", help="Disable text webbrowser agent (enabled by default)")
    parser.add_argument("--use-springer", action="store_true", default=True, help="Enable Springer tools (enabled by default)")
    parser.add_argument("--no-springer", action="store_false", dest="use_springer", help="Disable Springer tools")
    parser.add_argument("--use-browser", action="store_true", help="Enable interactive browser functionality for literature search tools")
    parser.add_argument("--use-ocr-agent", action="store_true", help="Enable OCR agent")
    parser.add_argument("--use-translator-agent", action="store_true", help="Enable translator agent")
    parser.add_argument("--use-speech-recognition-agent", action="store_true", help="Enable speech recognition agent")
    parser.add_argument("--results-json-path", type=str, default=None, help="Path to previous results JSON file for filtering already correct answers")
    parser.add_argument("--baseline", action="store_true", help="Use baseline agent instead of agent hierarchy")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory for results")
    parser.add_argument("--level", type=str, default="level2", choices=["level1", "level2", "level3"], help="Specify which level of questions to test")
    parser.add_argument("--question-ids", type=str, help="Comma-separated list of specific question IDs to run (e.g., '16,24,35')")
    parser.add_argument("--start-id", type=int, help="Starting question ID for a range of questions to run")
    parser.add_argument("--end-id", type=int, help="Ending question ID for a range of questions to run")
    parser.add_argument("--springer-api-key", type=str, help="Springer Nature API key", default=os.getenv("SPRINGER_API_KEY"))
    parser.add_argument("--llama-api-key", type=str, help="LlamaParse API key", default=os.getenv("LLAMA_API_KEY"))
    parser.add_argument("--use-Chinese-agent", action="store_true",default=False, help="Enable Chinese agent")
    return parser.parse_args()

print("Make sure you deactivated Tailscale VPN, else some URLs will be blocked!")

USE_OPEN_MODELS = False

SET = None

custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}

### LOAD EVALUATION DATASET
# eval_ds = datasets.load_dataset("gaia-benchmark/GAIA", "2023_all")[SET]
# eval_ds = eval_ds.rename_columns({"Question": "question", "Final answer": "true_answer", "Level": "task"})

RELATIVE_EXCEL_PATH = "Historical/HistBench/HistBench.xlsx"
# RELATIVE_EXCEL_PATH = "Historical/Historical/Historical Q&A collections(100).xlsx"

EXCEL_PATH = os.path.abspath(RELATIVE_EXCEL_PATH)

user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"

SHARED_CONFIG = {
    "downloads_folder": "downloads",
    "ocr_languages": ["en", "ch_sim"],
    "speech_model": "google",
    "translation_url": "http://127.0.0.1:5000/translate",
    "imgbb_api_key": os.getenv("IMGBB_API_KEY"),
    "serpapi_api_key": os.getenv("SERPAPI_API_KEY")
}

BROWSER_CONFIG = {
    "viewport_size": 1024 * 5,
    "downloads_folder": "downloads_folder",
    "request_kwargs": {
        "headers": {"User-Agent": user_agent},
        "timeout": 300,
    },
    "serpapi_key": os.getenv("SERPAPI_API_KEY"),
}

BROWSER_CONFIG_IMAGE = {
    "viewport_size": 1024 * 5,
    "downloads_folder": "image_downloads_folder",
    "request_kwargs": {
        "headers": {"User-Agent": user_agent},
        "timeout": 300,
    },
    "serpapi_key": os.getenv("SERPAPI_API_KEY"),
}

GOOGLE_LENS_CONFIG = {
    "imgbb_api_key": os.getenv("IMGBB_API_KEY"),
    "serpapi_api_key": os.getenv("SERPAPI_API_KEY"),
    "search_api_key": os.getenv("SEARCH_API_KEY")
}

OCR_CONFIG = {
    "imgbb_api_key": os.getenv("IMGBB_API_KEY"),
    "openrouter_api_key": os.getenv("OPENROUTER_API_KEY")
}

SPRINGER_CONFIG = {
    "springer_api_key": os.getenv("SPRINGER_API_KEY"),
    "llama_api_key": os.getenv("LLAMA_API_KEY"),
    "downloads_folder": "springer_downloads"
}

os.makedirs(f"./{BROWSER_CONFIG['downloads_folder']}", exist_ok=True)
os.makedirs(f"./{BROWSER_CONFIG_IMAGE['downloads_folder']}", exist_ok=True)
os.makedirs(f"./{SPRINGER_CONFIG['downloads_folder']}", exist_ok=True)


def create_agent_hierarchy(model: Model, use_image_agent=False, use_file_agent=False, use_literature_agent=False, use_text_webbrowser_agent=True, baseline=False, springer_api_key=None, llama_api_key=None, use_springer=True, use_browser=False, use_ocr_agent=False, use_translator_agent=False, use_speech_recognition_agent=False, use_Chinese_agent=False, logger=None):
    """
    Create agent hierarchy or baseline agent
    
    Parameters:
        model: Language model used
        use_image_agent: Whether to use image agent
        use_file_agent: Whether to use file processing agent
        use_literature_agent: Whether to use literature search agent
        use_text_webbrowser_agent: Whether to use text browser agent (enabled by default)
        baseline: Whether to use baseline agent instead of agent hierarchy
        springer_api_key: Springer Nature API key
        llama_api_key: LlamaParse API key
        use_springer: Whether to use Springer related tools (enabled by default)
        use_browser: Whether to enable browser functionality for literature search tools
        logger: Agent logger
    
    Returns:
        Agent: Created agent instance
    """
    
    text_limit = 100000
    ti_tool = TextInspectorTool(model, text_limit)

    browser = SimpleTextBrowser(**BROWSER_CONFIG)
    browser_image = SimpleImageBrowser(**BROWSER_CONFIG_IMAGE)

    Image_Reverse_Search_Tool = GoogleLensSearchTool(
        imgbb_api_key=GOOGLE_LENS_CONFIG["imgbb_api_key"],
        serpapi_api_key=GOOGLE_LENS_CONFIG["serpapi_api_key"],
        search_api_key= GOOGLE_LENS_CONFIG["search_api_key"]
    )
    Image_Reverse_Search_Tool.name = "Image_Reverse_Search_Tool"


    file_processor = FileProcessor(
        ocr_languages=SHARED_CONFIG["ocr_languages"],
        model=model
    )

    pdf_tool = PDFTool(file_processor)
    xlsx_tool = XLSXTool(file_processor)
    docx_tool = DOCXTool(file_processor)
    pptx_tool = PPTXTool(file_processor)

    WEB_TOOLS = [
        SearchInformationTool(browser),
        VisitTool(browser),
        PageUpTool(browser),
        PageDownTool(browser),
        FinderTool(browser),
        FindNextTool(browser),
        ArchiveSearchTool(browser),
        TextInspectorTool(model, text_limit),
    ]
    
    if baseline:
        text_webbrowser_agent = ToolCallingAgent(
            model=model,
            tools=WEB_TOOLS,
            max_steps=20,
            verbosity_level=2,
            planning_interval=4,
            name="search_agent",
            description="""A team member that will search the internet to answer your question.
            Ask him for all your questions that require browsing the web.
            Provide him as much context as possible, in particular if you need to search on a specific timeframe!
            And don't hesitate to provide him with a complex search task, like finding a difference between two webpages.
            Your request must be a real sentence, not a google search! Like "Find me this information (...)" rather than a few keywords.
            """,
            provide_run_summary=True,
            logger=logger
        )
        text_webbrowser_agent.prompt_templates["managed_agent"]["task"] += """You can navigate to .txt online files.
        If a non-html page is in another format, especially .pdf or a Youtube video, use tool 'inspect_file_as_text' to inspect it.
        Additionally, if after some searching you find out that you need more information to answer the question, you can use `final_answer` with your request for clarification as argument to request for more information."""

        manager_agent = CodeAgent(
            model=model,
            tools=[visualizer, ti_tool],
            max_steps=12,
            verbosity_level=2,
            additional_authorized_imports=AUTHORIZED_IMPORTS,
            planning_interval=4,
            managed_agents=[text_webbrowser_agent],
            logger=logger
        )
        return manager_agent
    
    
    LITERATURE_SEARCH_TOOLS = [
        LiteratureSearchingTool(api_key=os.getenv("OPENAI_API_KEY"), download_path="downloads_folder", use_browser=use_browser),
        GeneralBrowserTool(api_key=os.getenv("OPENAI_API_KEY"), download_path="downloads_folder", use_browser=use_browser),
        RelevantLiteratureFinderTool(api_key=os.getenv("OPENAI_API_KEY"), download_path="downloads_folder", use_browser=use_browser),
        BookMatchExtractorTool(api_key=os.getenv("OPENAI_API_KEY"), download_path="downloads_folder", use_browser=use_browser),
        DirectGoogleBooksCrawlerTool(api_key=os.getenv("OPENAI_API_KEY"), download_path="downloads_folder", use_browser=use_browser),
    ]
    
    if use_springer:
        springer_tools = [
            SpringerSearchTool(springer_api_key=springer_api_key, llama_api_key=llama_api_key, download_path="springer_downloads"),
            SpringerStructuredSearchTool(springer_api_key=springer_api_key, llama_api_key=llama_api_key, download_path="springer_downloads"),
            SpringerDownloadAndParseTool(springer_api_key=springer_api_key, llama_api_key=llama_api_key, download_path="springer_downloads"),
        ]
        LITERATURE_SEARCH_TOOLS.extend(springer_tools)
        print(LITERATURE_SEARCH_TOOLS)

    IMAGE_SEARCH_TOOLS = [
        SearchInformationTool_Image(browser_image),
        VisitTool_Image(browser_image),
        PageUpTool_Image(browser_image),
        PageDownTool_Image(browser_image),
        FinderTool_Image(browser_image),
        FindNextTool_Image(browser_image),
        ArchiveSearchTool_Image(browser_image),
        TextInspectorTool(model, text_limit),
        SaveHTMLTool(browser_image),
    ]
    
    FILE_TOOLS = [
        ImageAnalysisTool(file_processor, model),
        pdf_tool,
        docx_tool,
        xlsx_tool,
        pptx_tool
    ]

    ocr_tool = OCRTool(
        imgbb_api_key=OCR_CONFIG["imgbb_api_key"],
        openrouter_api_key=OCR_CONFIG["openrouter_api_key"],
        model=model
    )
    ocr_agent = ToolCallingAgent(
        model=model,
        tools=[ocr_tool],
        max_steps=5,
        verbosity_level=2,
        planning_interval=2,
        name="ocr_agent",
        description="""Agent specialized in image text recognition.
            
            Features:
            1. Extract text content from images
            2. Automatically detect languages in images
            3. Support multi-language OCR processing
            4. Provide image content description when OCR fails

            Use cases:
            - Extract text from screenshots, scanned documents, or photos
            - Process charts, images, or documents containing text
            - Recognize mixed multi-language content in images
        """,
        provide_run_summary=True,
        logger=logger
    )

    speech_tool = SpeechRecognitionTool(model)
    speech_recognition_agent = ToolCallingAgent(
        model=model,
        tools=[speech_tool],
        max_steps=3,
        verbosity_level=2,
        planning_interval=1,
        name="speech_recognition_agent",
        description="""Agent specialized in speech recognition.
            
            Features:
            1. Convert speech in audio files to text
            2. Support processing of multiple audio formats
            3. Use Google Speech Recognition API for transcription

            Use cases:
            - Transcribe recordings, voice notes, or audio meetings
            - Process voice commands or voice messages
            - Analyze audio content
        """,
        provide_run_summary=True,
        logger=logger
    )
    frame_extractor_tool = VideoFrameExtractorTool()
    frame_extractor_agent = ToolCallingAgent(
        model=model,
        tools=[frame_extractor_tool, visualizer],
        max_steps=5,
        verbosity_level=2,
        planning_interval=1,
        name="frame_extractor_agent",
        description="""Agent specialized in video frame extraction.
            
            Features:
            1. Extract frames from videos
            2. Support processing of multiple video formats
        """,
        provide_run_summary=True,
        logger=logger
    )

    translator_tool = TranslatorTool()
    translator_agent = ToolCallingAgent(
        model=model,
        tools=[translator_tool],
        max_steps=3,
        verbosity_level=2,
        planning_interval=1,
        name="translator_agent",
        description="""Agent specialized in text translation.
            
            Features:
            1. Translate text to different languages
            2. Support conversion between multiple languages
            3. Use specialized translation methods for special languages

            Use cases:
            - Translate foreign language text
            - Process multilingual content
            - Cross-language communication and understanding
        """,
        provide_run_summary=True,
        logger=logger
    )

    WEB_TOOLS.append(LocalGoogleSearchTool(model, browser))

    text_webbrowser_agent = ToolCallingAgent(
        model=model,
        tools=WEB_TOOLS,
        max_steps=20,
        verbosity_level=2,
        planning_interval=4,
        name="search_agent",
        description="""A team member that will search the internet to answer your question.
            Ask him for all your questions that require browsing the web or searching for academic literature.
            A general-purpose web search agent capable of retrieving both academic and non-academic information quickly across web pages, files, and multilingual content.
            Use this agent when:
            - Fast or broad coverage is needed
            - The task involves web-based sources (news, blogs, Wikipedia, videos, PDFs, etc.)
            - You want preliminary academic content from sources like Google Scholar or publisher pages
            - The language of the query is not English (e.g., Chinese, German)
            This agent supports academic search, but it does not specialize in scholarly database crawling.
        """,
        provide_run_summary=True,
        logger=logger
    )

    text_webbrowser_agent.prompt_templates["managed_agent"]["task"] += """You can navigate to .txt online files.
        If a non-html page is in another format, especially .pdf or a Youtube video, use tool 'inspect_file_as_text' to inspect it.
        
        Additionally, if after some searching you find out that you need more information to answer the question, you can use `final_answer` with your request for clarification as argument to request for more information.
        
        You are the `text_webbrowser_agent`, a fast and flexible search agent for retrieving both academic and non-academic information from the open web.

        **Your Strengths**:
        - Fast search with broad coverage
        - Access to web pages, PDFs, videos, and multilingual content
        - Can return preliminary academic sources from open-access sites (e.g., Google Scholar, publisher homepages)

        **Use Cases**:
        - When a quick answer is needed
        - When the query is not strictly academic (e.g., involves media, practical info, non-peer-reviewed knowledge)
        - When the query is in Chinese, German, or another language requiring multilingual search
        - When scholarly precision is not the top priority (e.g., exploring relevant context or background first)

        **Important Functions**:
        1. Start with `LocalGoogleSearchTool` to gather search results.
        2. Use `VisitTool` to read high-potential pages.
        3. Use `TextInspectorTool` to analyze special files (e.g., .pdf, .docx, .pptx).
        4. Use `final_answer()` to return clarification requests if needed.

        **Fallback Recommendation**:
        If the query clearly requires peer-reviewed precision (e.g., academic definition, citation, exact phrase matching), consider passing the task to `literature_search_agent`.

        **!!!Attention!!!** 
        ALL Numbers in the task (such as year, quantity, etc.) and the corresponding context of the numbers MUST be retained as the input, including background information.

        Additionally, if after some searching you find out that you need more information to answer the question, you can use `final_answer` with your request for clarification as argument to request for more information.
        Start now.
        ---
        Task:
        {{task}}
    """

    image_information_agent = ToolCallingAgent(
        model=model,
        tools=[*IMAGE_SEARCH_TOOLS, Image_Reverse_Search_Tool, visualizer], 
        max_steps=15,
        verbosity_level=2,
        planning_interval=4,
        name="image_information_agent",
        description="""You are the image_information_agent. You are always the first to process any task that involves an image.

            Your responsibility is to extract, search, and summarize all relevant information from images and their online appearances. You perform reverse image search and follow up with targeted page visits to uncover the origin, meaning, and associated context of any image.

            Use this agent when:
            - The task contains an image file path (e.g., .jpg, .png, .webp)
            - The user asks what is shown, written, or represented in an image
            - The goal is to identify the image's source, creator, or related knowledge online

            Your core capabilities:
            - Perform reverse image search using `Image_Reverse_Search_Tool`
            - Visit related pages using `VisitTool` to find detailed metadata
            - Separate information found **in the image** (e.g., symbols, people, writing) from what exists **about the image online**

            Execution rules:
            - You must run even if the image contains non-English text (e.g., Chinese calligraphy)
            - Only process images that are explicitly mentioned in the task — ignore all illustrative examples

            You are the default entry point for all image-related tasks.
        """,
        provide_run_summary=True,
        logger=logger
    )

    image_information_agent.prompt_templates["managed_agent"]["task"] += """
        You are the `image_information_agent`, responsible for extracting and analyzing information from images. You process both the **visual content** and its **online context** by using reverse image search and web tools.
        You should give the highest importance and priority to the first result of the `Image_Reverse_Search_Tool`, which includes the website title, the image source link, and the image url.
        1. **Image_Reverse_Search_Tool**:
        - Purpose: Find where an image appears online and discover related information.
        - When to use: This should be your first step when analyzing any image.
        - Output: Provides links to web pages where the image or similar images appear.
        - You should give the first result of the `Image_Reverse_Search_Tool`, which includes the website title, the image source link, and the image url, the highest importance and priority.

        2. **VisitTool**:
        - Purpose: Visit a specific web page to gather detailed information.
        - When to use: When you need to examine a particular web page in detail.
        - What to look for: Detailed information such as:
            * Product descriptions and specifications
            * Historical context and background information
            * Auction details and provenance information
            * Artist or creator information
            * Dating and authentication details
            * Any other relevant contextual information
        - Advantage: Allows focused analysis of a single important page.

        **Recommended Functions**:
        1. Start with `Image_Reverse_Search_Tool` to find where the image appears online.
        2. Use `VisitTool` to visit all the pages you found in the `Image_Reverse_Search_Tool` including the "link" and "image URL".
        3. Use `visualizer` to visualize the image by the "image URL" returned by `Image_Reverse_Search_Tool`.
        3. Integrate all findings into a comprehensive report about the image.

        **IMPORTANT: DISTINGUISHING EXAMPLES FROM ACTUAL TASKS**
        The following is just an EXAMPLE to illustrate the workflow. DO NOT process 'historical_document.png' unless it's specifically mentioned in the actual task:

        - *Example Task*: Analyze 'historical_document.png'.
        - *Example Process*:
            - Use `Image_Reverse_Search_Tool: historical_document.png` to find online sources
            - Use `VisitTool: https://specific-page.com` for any specific page that needs detailed examination
            - Integrate findings into a report

        Your objective is to process only the actual images mentioned in the current task, not any examples used for illustration.

        Your task is:
        {{task}}

        Begin by identifying any image file paths in this task and using Image_Reverse_Search_Tool. You'd better visit the top five results returned by `Image_Reverse_Search_Tool` first.
    """

    literature_search_agent = ToolCallingAgent(
        model=model,
        tools=LITERATURE_SEARCH_TOOLS,
        max_steps=8,
        verbosity_level=2,
        planning_interval=4,
        name="literature_search_agent",
        description="""A specialized literature research agent skilled in finding authoritative academic sources for historical questions:
        
        Use this agent when:
        - The task demands peer-reviewed articles, citations, or scholarly books
        - Precision and source quality are more important than response speed
        - You need to locate exact phrases, match historical facts, or verify academic claims

        This agent is slower than general search but significantly more reliable for formal academic tasks.
        1. LiteratureSearchingTool: Search for scholarly articles and books on a specific topic
        2. RelevantLiteratureFinderTool: Find and filter the most relevant literature sources
        3. GeneralBrowserTool: Perform general web searches for academic information
        4. BookMatchExtractorTool: Extract book match snippets from Google Books search
        5. DirectGoogleBooksCrawlerTool: Directly analyze Google Books search results
        6. SpringerSearchTool: Search academic papers on Springer Nature's open access platform
        7. SpringerStructuredSearchTool: Perform structured searches using categorized research concepts
        8. SpringerDownloadParseTool: Download and parse PDFs from Springer Nature using LlamaParse
        
        For "exactMatch" questions, search for the exact original wording that exists in scholarly literature.
        For other history questions, locate and verify facts using credible academic sources.
        """,
        provide_run_summary=True,
        logger=logger
    )

    literature_search_agent.prompt_templates["managed_agent"]["task"] = """You are the `literature_search_agent`, a specialized agent for high-quality academic literature retrieval.

**Primary Role**:
- Handle academic and historical questions where **source credibility**, **precision**, and **citation** are critical.

**Use Cases**:
- "exactMatch" type questions where answers must appear verbatim in books or papers
- Verification of scientific, medical, or historical facts
- Retrieval of scholarly articles, citations, or excerpts from books

For 'exactMatch' type questions: 
- The EXACT original wording can be found in scholarly literature
- Your primary task is to locate this exact text
- The answer exists verbatim in academic sources
- CRITICAL REQUIREMENT: You MUST input the ENTIRE question text as your search query
- IMPORTANT: If the question contains blanks (like "____", "___", or "[BLANK]"), remove these blanks before searching
- Example: "The Battle of _____ was fought in 1815" → search for "The Battle of was fought in 1815"
- Do NOT break down the question into keywords - use the complete text

For all other question types:
- Relevant supporting content must be found in academic sources
- Prioritize high-quality, well-cited scholarly papers

You have five powerful tools at your disposal:

1. **LiteratureSearchingTool**:
   - Purpose: Search for high-impact, recent scholarly articles on a specific topic
   - Usage: `LiteratureSearchingTool: [research topic/query]`
   - Output: Returns 5 relevant scholarly articles with citation counts, publication years, and key findings
   - When to use: For initial broad search of authoritative academic sources

2. **RelevantLiteratureFinderTool**:
   - Purpose: Filter and rank the most relevant literature sources for a specific query
   - Usage: `RelevantLiteratureFinderTool: [specific research question]`
   - Output: Returns the 3 most relevant sources with relevance scores and key information
   - When to use: To pinpoint the most directly relevant sources for your question
   - For exactMatch questions, use this to find the exact original wording

3. **GeneralBrowserTool**:
   - Purpose: Perform general web searches beyond academic databases
   - Usage: `GeneralBrowserTool: [search query]`
   - Output: Returns general web search results
   - When to use: Only after exhausting academic sources, for supplementary information

4. **BookMatchExtractorTool**:
   - Purpose: Extract exact book match snippets from Google Books with highlighted matching terms
   - Usage: `BookMatchExtractorTool: [exact phrase to search]`
   - Output: Returns book match snippets with highlighted terms that match the query
   - When to use: BEST TOOL for exactMatch questions - use this FIRST with the entire question (blanks removed)
   - Example: For "The Battle of _____ was fought in 1815"
   - Do this: `BookMatchExtractorTool: The Battle of was fought in 1815`

5. **DirectGoogleBooksCrawlerTool**:
   - Purpose: Extract book match snippets directly from a Google Books search URL
   - Usage: `DirectGoogleBooksCrawlerTool: [google books search URL]`
   - Output: Returns book match snippets from the URL with highlighted terms
   - When to use: When you already have a Google Books search URL and need to extract match snippets"""

    if use_springer:
        literature_search_agent.prompt_templates["managed_agent"]["task"] += """
    6. SpringerSearchTool: Search academic papers on Springer Nature's open access platform
    - Usage: `SpringerSearchTool: [research topic/query]`
    - Output: Returns 5 relevant scholarly articles with citation counts, publication years, and key findings
    - When to use: For initial broad search of authoritative academic sources

7. SpringerStructuredSearchTool: Perform structured searches using categorized research concepts  
    - Usage: `SpringerStructuredSearchTool: [research topic/query]`
    - Output: Returns 5 relevant scholarly articles with citation counts, publication years, and key findings
    - When to use: For initial broad search of authoritative academic sources

8. SpringerDownloadParseTool: Download and parse PDFs from Springer Nature using LlamaParse
    - Usage: `SpringerDownloadParseTool: [url]`
    - Output: Returns 5 relevant scholarly articles with citation counts, publication years, and key findings
    - When to use: For initial broad search of authoritative academic sources
"""

    literature_search_agent.prompt_templates["managed_agent"]["task"] += """

**Mandatory Functions for exactMatch questions**:
1. FIRST use `BookMatchExtractorTool` with the ENTIRE question text (with blanks removed)
   - Example: For "The Battle of _____ was fought in 1815"
   - Do this: `BookMatchExtractorTool: The Battle of was fought in 1815`

2. If no exact match is found, use `RelevantLiteratureFinderTool` with the same query
   - Example: `RelevantLiteratureFinderTool: The Battle of was fought in 1815`

3. If still no exact match, use traditional literature search tools

For all other questions:
- Start with `LiteratureSearchingTool` to get a broad overview of scholarly articles
- Then use `RelevantLiteratureFinderTool` with precise query terms to find the most relevant sources
- Only after exhausting academic sources, use `GeneralBrowserTool` if needed

Always integrate findings into a comprehensive answer with proper academic citations

You have been submitted this task by your manager.
---
Task:
{{task}}
---

Begin by determining if this is an exactMatch question. If it is, use BookMatchExtractorTool with the entire question text (blanks removed) FIRST. If not, proceed with the standard workflow starting with LiteratureSearchingTool.
"""

    file_processor_agent = ToolCallingAgent(
        model=model,
        tools=FILE_TOOLS,
        max_steps=20,
        verbosity_level=2,
        planning_interval=4,
        name="file_processor",
        description="""A specialized team member for processing various types of files:
1. Automatic File Type Detection: 
   - Files are automatically analyzed to determine their type
   - No need to specify file type in your requests
   - Just provide the file path and the appropriate tool will be selected

2. OCR: Extract ONLY the plain text from images using EasyOCR
   - Returns EXACTLY the text content with no analysis or additional information
   - Input: image file path
   - Output: extracted text only
   
3. Image Analysis: Analyze and describe image content in detail
   - Provides detailed descriptions of what appears in the image
   - Input: image file path
   - Output: comprehensive description of the image content
   
4. Speech Recognition: Convert speech to text
   - Input: audio file path (.wav, .mp3, etc.)
   - Output: transcribed text
   
5. Translation: Translate text between languages
   - Input: text and target language code (e.g., 'en', 'zh')
   - Output: translated text
""",
        provide_run_summary=True,
        logger=logger
    )


    file_processor_agent.prompt_templates["managed_agent"]["task"] += """
File Type Detection:
- The system automatically detects file types based on file extension or content analysis
- Simply provide the file path without specifying the file type
- Example: "Extract content from this file: /path/to/file.ext" instead of "Extract text from this image: /path/to/image.png"

For image files (detected automatically):
- Supported formats: .png, .jpg, .jpeg, .bmp
- Two processing options:
  1. Text extraction using OCR - when you need to extract text from the image
  2. Image analysis - when you need to understand the image content and get a detailed description
- Example: "Extract text from this image: /path/to/image.jpg" for OCR
- Example: "Analyze this image: /path/to/image.jpg" for visual description

For audio files (detected automatically):
- Supported formats: .wav, .mp3, .m4a
- Speech recognition is applied automatically
- For non-English audio, transcribe first then translate

For document files (detected automatically):
- Supported formats: .pdf, .docx, .xlsx, .pptx
- Text extraction is applied based on document type

For text translation:
- Use TranslatorTool with appropriate language codes
- Common codes: 'en' (English), 'zh' (Chinese), 'ja' (Japanese), 'ko' (Korean)

If you encounter any issues:
- Check if file exists
- Verify file path is correct
- Use `final_answer` with error description if file is inaccessible or format unsupported
"""
    cn_model  = LiteLLMModel(
            model_id="openrouter/deepseek/deepseek-r1",
            api_key=OCR_CONFIG["openrouter_api_key"],
            api_base="https://openrouter.ai/api/v1",
            max_completion_tokens=8192,
            drop_params=True,
        )
    Chinese_agent = ToolCallingAgent(
        model=cn_model,
        tools=[visualizer, ti_tool],
        verbosity_level=2,
        planning_interval=4,
        name="Chinese_agent",
        description="""You are an experienced history expert, familiar with important events, figures, and intellectual trends across all historical periods. Please complete the following history multiple-choice question with a rigorous approach:
        Please analyze all options one by one, pointing out the basis for or errors in each option;
        Finally, clearly indicate the correct answer that best aligns with historical facts or the intent of the question, and briefly explain the reason;
        Your answer should demonstrate logical thinking and professionalism, avoiding intuition-based responses.""",
        logger=logger
    )

    managed_agents = []
    
    if use_text_webbrowser_agent:
        managed_agents.append(text_webbrowser_agent)
    
    if use_image_agent:
        managed_agents.append(image_information_agent)
    
    if use_literature_agent:
        managed_agents.append(literature_search_agent)
    
    if use_file_agent:
        managed_agents.append(file_processor_agent)
    
    if use_ocr_agent:
        managed_agents.append(ocr_agent)

    if use_translator_agent:
        managed_agents.append(translator_agent)

    if use_speech_recognition_agent:
        managed_agents.append(speech_recognition_agent)

    if use_Chinese_agent:
        managed_agents.append(Chinese_agent)
    
    managed_agents.append(frame_extractor_agent)

    manager_agent = CodeAgent(
        model=model,
        tools=[visualizer,ti_tool],
        max_steps=20,
        verbosity_level=2,
        additional_authorized_imports=AUTHORIZED_IMPORTS,
        planning_interval=4,
        managed_agents=managed_agents,
        name="manager",
        description="""Team manager responsible for coordinating work between different agents.
        You can use following agents to solve the task:
        1. text_webbrowser_agent - For web searches and browsing
        2. image_information_agent - For image analysis
        3. literature_search_agent - Specialized agent for academic literature searches
        4. ocr_agent - For OCR text extraction from images
        5. translator_agent - For text translation
        6. speech_recognition_agent - For speech recognition
        7. Chinese_agent - For Chinese text analysis
        
        Remember:
        - For any image file, you must use image_information_agent and visualizer to analyze the image!
        - For image with text, you must use ocr_agent to extract the text first!
        - For any question without image file, you must use text_webbrowser_agent to obtain factual information!
        """,
        logger=logger
    )
    manager_agent.prompt_templates["system"] = """You are a team manager, responsible for coordinating the work of specialized agents to solve complex tasks.

You have access to the following agents:
1. text_webbrowser_agent - For web searches and browsing
2. image_information_agent - For image analysis
3. literature_search_agent - Specialized agent for academic literature searches
4. ocr_agent - For OCR text extraction from images
5. translator_agent - For text translation
6. speech_recognition_agent - For speech recognition
7. Chinese_agent - For Chinese text analysis
8. frame_extractor_agent - For video frame extraction
"""

    manager_agent.prompt_templates["task"] = """You are the manager of a team of specialized agents. Your job is to coordinate their work to solve complex tasks.
    You MUST use the text_webbrowser_agent to search for the information you need.
Remember, image_information_agent is not the visualizer.
you must use the ocr_agent to extract the text first.
"""

    return manager_agent


def append_answer(entry: dict, jsonl_file: str) -> None:
    jsonl_file = Path(jsonl_file)
    jsonl_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Get main logger
    logger = logging.getLogger("main")
    
    # Get task ID to create task-specific logger
    task_id = str(entry.get("task_id", "unknown"))
    task_logger = get_task_logger(LOG_DIR, task_id)
    
    # Write to JSONL file
    with append_answer_lock, open(jsonl_file, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(entry) + "\n")
    
    # Get Excel file path
    excel_file = jsonl_file.with_suffix('.xlsx')
    
    # Convert entry to DataFrame
    entry_df = pd.DataFrame([entry])
    
    # If Excel file exists, append data
    if os.path.exists(excel_file):
        try:
            existing_df = pd.read_excel(excel_file)
            # Merge old and new data
            combined_df = pd.concat([existing_df, entry_df], ignore_index=True)
            # Write to Excel file
            combined_df.to_excel(excel_file, index=False)
        except Exception as e:
            task_logger.error(f"Error updating Excel file: {e}, creating new file")
            entry_df.to_excel(excel_file, index=False)
    else:
        # If Excel file doesn't exist, create new file
        entry_df.to_excel(excel_file, index=False)
    
    assert os.path.exists(jsonl_file), "JSONL file doesn't exist!"
    assert os.path.exists(excel_file), "Excel file doesn't exist!"
    task_logger.info(f"Answer exported to file: {jsonl_file.resolve()}")
    task_logger.info(f"Answer exported to Excel file: {excel_file.resolve()}")
    logger.info(f"Answer exported to file: {jsonl_file.resolve()} and Excel file: {excel_file.resolve()}")


def answer_single_question(example, model_id, answers_file, visualizer, args):
    """Answer a single question and save the result, including answer evaluation and summary generation"""
    # Get task ID, ensure it's a string
    task_id = str(example["task_id"])
    
    # Create task-specific logger
    task_logger = logging.getLogger(f"task_{task_id}")
    task_logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    if task_logger.handlers:
        task_logger.handlers.clear()
    
    # Create log directory
    log_dir = os.path.join(os.path.dirname(answers_file), "output_logs", args.run_name)
    os.makedirs(log_dir, exist_ok=True)
    
    # Create file handler
    file_handler = logging.FileHandler(os.path.join(log_dir, f"{task_id}.log"), encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    task_logger.addHandler(file_handler)
    task_logger.addHandler(console_handler)
    
    # Set not to propagate to parent (avoid duplicate logs)
    task_logger.propagate = False
    
    # Log start information
    task_logger.info(f"Starting to process task ID: {task_id}")
    task_logger.info(f"Question: {example['question']}")
    task_logger.info(f"Using model: {model_id}")
    
    # Create two console instances: one for file, one for terminal
    file_console = Console(file=open(os.path.join(log_dir, f"des_{task_id}.txt"), 'wt'))
    terminal_console = Console()  # Default output to terminal
    
    # Create a combined AgentLogger that outputs to both file and terminal
    class DualOutputAgentLogger(AgentLogger):
        def __init__(self, file_console, terminal_console, level=2):
            super().__init__(level=level, console=file_console)
            self.terminal_console = terminal_console
            
        def log(self, *args, level=LogLevel.INFO, **kwargs):
            # Call parent class log method to write to file
            super().log(*args, level=level, **kwargs)
            # Also output to terminal
            if level <= self.level:
                self.terminal_console.print(*args, **kwargs)
    
    # Use combined AgentLogger
    agent_logger = DualOutputAgentLogger(file_console, terminal_console, level=2)
    
    # Create model
    model = LiteLLMModel(
        model_id,
        api_key=os.getenv("OPENAI_API_KEY"),
        custom_role_conversions=custom_role_conversions,
        max_completion_tokens=8192,
        drop_params=True,
    )
    
    document_inspection_tool = TextInspectorTool(model, 100000)
    
    agent = create_agent_hierarchy(model, 
                                  use_image_agent=args.use_image_agent, 
                                  use_file_agent=args.use_file_agent, 
                                  use_literature_agent=args.use_literature_agent, 
                                  use_text_webbrowser_agent=not args.no_text_webbrowser_agent,
                                  baseline=args.baseline,
                                  springer_api_key=args.springer_api_key or SPRINGER_CONFIG["springer_api_key"],
                                  llama_api_key=args.llama_api_key or SPRINGER_CONFIG["llama_api_key"],
                                  use_springer=args.use_springer,
                                  use_browser=not args.use_browser,
                                  use_ocr_agent=args.use_ocr_agent,
                                  use_translator_agent=args.use_translator_agent,
                                  use_speech_recognition_agent=args.use_speech_recognition_agent,
                                  use_Chinese_agent=args.use_Chinese_agent,
                                  logger=agent_logger)  # Add logger parameter

    # Get question-related information from example
    question = example["question"]
    answer_type = example.get("answer_type", "")
    data_type = example.get("data_type", "none")
    data_requirement = example.get("data_requirement", "")

    augmented_question = f"""You have one question to answer. It is paramount that you provide a correct answer.
Give it all you can: I know for a fact that you have access to all the relevant tools to solve it and find the correct answer (the answer does exist). Failure or 'I cannot answer' or 'None found' will not be tolerated, success will be rewarded.
**Important:** For Questions with Images, You must use `image_information_agent`
"""

    augmented_question += f"""
IMPORTANT: Use web search tools(firstly use text_webbrowser_agent) and literature_search_agent to verify information or to gather additional context. Relying solely on your internal knowledge without verification can lead to incorrect answers.
You must make sure you find the correct answer!
For questions requiring academic or scholarly information, you should consider using `literature_search_agent`. You can provide it with the question, and it will find the information in scholarly literature. If it doesn't work well, change to `text_webbrowser_agent` immediately.
For question with image files, you must use both the image_information_agent and visualizer to analyze the image. image_information_agent and visualizer are two independent agent and tool.
If texts are inspected from the image, you must use the ocr_agent to extract the text first.
Here is the task:
{question}
"""

    # Process data based on data type
    if data_type == "file":
        # Check if there are multiple files (determined by file_names field)
        if "file_names" in example and isinstance(example["file_names"], list) and len(example["file_names"]) > 1:
            file_names = example["file_names"]
            task_logger.info(f"Processing multiple files (total {len(file_names)})")
            
            # Create prompt for multiple files
            prompt_use_files = "\n\nTo solve the task above, you will have to analyze these attached files:\n"
            
            # Limit the number of files to process to avoid overly long prompts
            max_files_to_process = min(10, len(file_names))
            processed_files = []
            
            # Process each file
            for i, file_path in enumerate(file_names[:max_files_to_process]):
                if not isinstance(file_path, str) or not os.path.exists(file_path):
                    task_logger.warning(f"Skipping invalid file path: {file_path}")
                    continue
                    
                file_basename = os.path.basename(file_path)
                file_ext = os.path.splitext(file_path)[1].lower()
                
                task_logger.info(f"Processing file {i+1}/{max_files_to_process}: {file_basename}")
                
                try:
                    # Choose appropriate description method for different file types
                    if file_ext in ['.jpg', '.jpeg', '.png', '.gif']:
                        # Special handling for image files
                        prompt_use_files += f"\n### File {i+1}: {file_basename} (Image)\n"
                        # Only add basic information, suggest using specialized tools
                        prompt_use_files += f"- File Path: {file_path}\n"
                        
                    else:
                        # Other files use generic description method
                        prompt_use_files += f"\n### File {i+1}: {file_basename}\n"
                        file_desc = get_single_file_description(
                            file_path, example["question"], visualizer, document_inspection_tool
                        )
                        prompt_use_files += file_desc
                    
                    processed_files.append(file_basename)
                except Exception as e:
                    task_logger.error(f"Error getting description for file {file_basename}: {e}")
                    prompt_use_files += f"\n### File {i+1}: {file_basename}\n"
                    prompt_use_files += f"- Unable to get file description: {str(e)}\n"
            
            # If there are more files not processed, add a note
            if len(file_names) > max_files_to_process:
                remaining = len(file_names) - max_files_to_process
                prompt_use_files += f"\n\nThere are {remaining} other files not detailed here. As needed, you can use appropriate tools to process these files."
            
            # Add unified usage instructions
            prompt_use_files += f"""

## File Processing Guidelines:
1. Please analyze all provided files, don't just focus on the first file
2. Integrate information from all files to answer the question
"""
            
        elif ".zip" in str(example.get("file_name", "")):
            # ZIP file processing
            task_logger.info(f"Processing ZIP file: {example.get('file_name', '')}")
            prompt_use_files = "\n\nTo solve the task above, you will have to use these attached files:\n"
            prompt_use_files += get_zip_description(
                example["file_name"], example["question"], visualizer, document_inspection_tool
            )
            prompt_use_files += "\n\nNote: For different file types in the ZIP file, please choose the most appropriate specialized tool"
        else:
            # Single file processing
            file_path = example.get("file_name", "")
            task_logger.info(f"Processing single file: {file_path}")
            
            if not file_path or not os.path.exists(file_path):
                task_logger.warning(f"Warning: Specified file does not exist or is invalid: {file_path}")
                prompt_use_files = "\n\nWarning: Specified file does not exist or is invalid"
            else:
                prompt_use_files = "\n\nTo solve the task above, you will have to use this attached file:"
                try:
                    prompt_use_files += f"\n### File : {file_path}\n"
                except Exception as e:
                    task_logger.error(f"Error getting file description: {e}")
                    prompt_use_files += f"\nError: Cannot read file ({str(e)})\n"
        
        # Add to final question
        augmented_question += prompt_use_files
    
    elif data_type == "foreign_text":
        # Process foreign language text
        task_logger.info("Processing foreign language text data")
        translator = TranslatorTool()
        translated_text = translator.forward(data_requirement, target="en")
        augmented_question += f"\n\nTo solve the task above, you will need to understand this text:\n{translated_text}\n"
        augmented_question += f"\nOriginal text: {data_requirement}\n"
    
    elif data_type == "search_query":
        # Process information that needs to be searched
        task_logger.info(f"Processing search query data: {data_requirement}")
        augmented_question += f"\n\nTo solve the task above, you will need to search for information about: {data_requirement}\n"
        augmented_question += "Please use the web browsing tools to find relevant information.\n"
    
    elif data_type == "text":
        # Process regular text
        task_logger.info(f"Processing regular text data: {data_requirement[:50]}...")
        augmented_question += f"\n\nAdditional information: {data_requirement}\n"

    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_logger.info(f"Starting task execution, time: {start_time}")
    
    try:
        # Run agent to generate answer
        task_logger.info("Starting to run agent...")
        final_result = agent.run(augmented_question)

        agent_memory = agent.write_memory_to_messages(summary_mode=True)

        # Check if literature_search_agent was used (for history questions)
        literature_agent_used = False
        for step in agent.memory.steps:
            step_str = str(step)
            if "literature_search_agent" in step_str or "literature_searching_task" in step_str or "relevant_literature_finder" in step_str:
                literature_agent_used = True
                break
        
        # For history questions, if literature_search_agent wasn't used, add a warning
        if not literature_agent_used and args.use_literature_agent and "history" in example["question"].lower():
            task_logger.warning("Warning: This history question did not use literature_search_agent!")
            # Add a warning message to agent_memory
            warning_message = {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "WARNING: This history question was answered without using the mandatory literature_search_agent. The answer may not be based on proper academic sources and could be inaccurate. For better results, the literature_search_agent should have been used to find scholarly sources."
                    }
                ]
            }
            if isinstance(agent_memory, list):
                agent_memory.append(warning_message)

        task_logger.info("Preparing final answer...")
        final_result = prepare_response(augmented_question, agent_memory, reformulation_model=model)

        output = str(final_result)
        
        # For history questions, if literature_search_agent wasn't used, add a warning to the output
        if not literature_agent_used and args.use_literature_agent and "history" in example["question"].lower():
            output = "WARNING: This history question was answered without using the mandatory literature_search_agent. The answer may lack proper academic sourcing.\n\n" + output
        
        task_logger.info(f"Agent run complete, generated answer length: {len(output)}")
        
        for memory_step in agent.memory.steps:
            memory_step.model_input_messages = None
        intermediate_steps = [str(step) for step in agent.memory.steps]

        parsing_error = True if any(["AgentParsingError" in step for step in intermediate_steps]) else False
        iteration_limit_exceeded = True if "Agent stopped due to iteration limit or time limit." in output else False
        raised_exception = False

    except Exception as e:
        task_logger.error(f"Agent run error: {e}")
        output = None
        intermediate_steps = []
        parsing_error = False
        iteration_limit_exceeded = False
        raised_exception = True
        
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_logger.info(f"Task execution completed, time: {end_time}")
    
    def check_answer_internal(model_answer: str, true_answer: str, question: str, task_id: str) -> bool:
        """
        Synchronous answer evaluation: returns True for "correct", False for "incorrect".
        """
        # 1. Construct messages
        messages = [
            {"role": "system", "content": "You are a strict JSON-only responder. Always output exactly one JSON object and nothing else."},
            {"role": "user",   "content": JUDGE_PROMPT.format(
                question=question,
                response=model_answer,
                correct_answer=true_answer
            )},
        ]

        try:
            # —— Option A: If synchronous parse is supported —— 
            resp = openai.beta.chat.completions.parse(
                model="gpt-4o",
                messages=messages,
                max_tokens=1024,
                response_format=ExtractedAnswer
            )
            parsed = resp.choices[0].message.parsed
            
            # Save parsed content to file
            parsed_dir = os.path.join(os.path.dirname(answers_file), "parsed_results")
            os.makedirs(parsed_dir, exist_ok=True)
            
            # Use timestamp to create unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            parsed_file = os.path.join(parsed_dir, f"parsed_{task_id}_{timestamp}.json")
            
            # Convert parsed object to dictionary and save
            parsed_dict = parsed.dict()
            with open(parsed_file, "w", encoding="utf-8") as f:
                json.dump(parsed_dict, f, ensure_ascii=False, indent=4)
            
            task_logger.info(f"Parsed results saved to: {parsed_file}")

            # 3. Make binary judgment based on parsed.correct
            is_correct = (parsed.correct == "yes")
            task_logger.info(f"Evaluation: {'correct' if is_correct else 'incorrect'} (confidence={parsed.confidence}%)")
            return is_correct

        except json.JSONDecodeError:
            task_logger.error(f"Return content is not valid JSON:\n{raw}")
            return False
        except ValidationError as ve:
            task_logger.error(f"JSON schema validation failed: {ve}")
            return False
        except Exception as e:
            task_logger.error(f"Error during evaluation: {e}")
            return False
        
    # Use built-in function to check if answer is correct
    is_correct = check_answer_internal(output, example["true_answer"], example["question"], example["task_id"])
    
    # Built-in function: generate problem solution summary
    def generate_summary_internal(result):
        """Generate problem summary"""
        try:
            task_logger.info("Generating problem solution summary...")
            # Simply print data type, don't try to print content (may cause errors)
            task_logger.debug(f"Input data type: {type(result)}")
            
            # Prepare default values for variables
            question = "Unknown question"
            answer = "Unknown answer"
            reasoning = "No reasoning process provided"
            
            # If result is dictionary type, try to safely get values
            if isinstance(result, dict):
                # Safely get dictionary values
                question = result.get("question", question)
                answer = result.get("answer", answer)
                reasoning = result.get("reasoning", reasoning)
                # Safely print dictionary contents
                task_logger.debug("Result dictionary contains the following keys: " + str(list(result.keys())))
            # If result is string type
            elif isinstance(result, str):
                task_logger.warning("Warning: result is string type, using default structure")
                # String as answer
                answer = result
            # Handle other types
            else:
                task_logger.warning(f"Warning: result is unexpected type: {type(result)}")
                # Try to convert to string as answer
                try:
                    answer = str(result)
                except:
                    pass
            
            # Use message format consistent with text_inspector_tool.py
            messages = [
                {
                    "role": MessageRole.SYSTEM,
                    "content": [
                        {
                            "type": "text",
                            "text": """As a fair evaluator, please assess whether the model's answer is semantically consistent with the correct answer. For any search results included in the summary, extract the source of each sentence and include references, clearly indicating where each piece of information was obtained from."""
                        }
                    ]
                },
                {
                    "role": MessageRole.USER,
                    "content": [
                        {
                            "type": "text",
                            "text": f"""Question: {question}

Answer: {answer}

Please write a structured and easy-to-read summary report based on the following problem-solving process:
{reasoning}

Your report **must be written in plain language** that is easy to understand. The key requirement is:  
⚠️ **All cited content must clearly include both the specific quote and the URL**, so that the information can be verified manually without ambiguity.

Your summary must include the following four parts:

1. **Tools used and how they were used:**
   - List each tool used (e.g., web search, image analysis, OCR, translation).
   - For each tool, explain exactly what was done (e.g., search keywords, what content was translated).
   - Clearly state what result the tool returned (e.g., if OCR returned a paragraph, show that paragraph).
   - Explain why each tool was selected for this problem.
   ⚠️ **Reminder: Most problems require Web search. If it was not used, this is a serious flaw.**

2. **Detailed information sources:**
   - Provide source titles, webpage URLs, and author names (if available).
   - For each source, include **exact text excerpts** in quotation marks, along with citation and URL, for example:
     * "Maintaining proper blood sugar levels is crucial for preventing type 2 diabetes." — [Mayo Clinic](https://www.mayoclinic.org/...)
   - Assess the credibility of each source (e.g., medical institution, news agency, academic article).
   - If multiple sources were used to verify the same fact, indicate cross-verification explicitly.
   ⚠️ **Do not just give URLs—actual quoted content is required for every source.**

3. **Reasoning process and logic steps:**
   - Show how the final answer was derived step-by-step from the information found.
   - List any assumptions made and how they were verified.
   - Describe how different pieces of information were integrated and compared.
   - Explain why other possible answers were excluded, and based on what evidence.
   - Highlight key reasoning steps or decision points.

4. **Answer quality and reliability analysis:**
   - Rate the reliability (high / medium / low), and explain your reasoning.
   - Point out any assumptions, weaknesses, or uncertainties in the final answer.
   - Evaluate whether the evidence is sufficient and consistent.
   - Suggest possible improvements or further verification steps.
   - ⚠️ If Web search was not used, emphasize clearly that this reduces reliability, and suggest what keywords should have been searched.

Your report must be written clearly, sectioned by part, and all source citations must include **both quoted text and URLs**. This is the most important requirement for verification."""
                        }
                    ]
                }
            ]
            summary = model(messages)
            summary_text = summary.content if hasattr(summary, 'content') else str(summary)
            task_logger.info("Summary generation complete")
            return f"\n\n### Solution Process Summary ###\n{summary_text}\n\n"
        except Exception as e:
            # Detailed error information
            error_type = type(e).__name__
            error_msg = str(e)
            import traceback
            trace = traceback.format_exc()
            task_logger.error(f"Error generating summary: {error_type}: {error_msg}")
            task_logger.debug(f"Detailed error information:\n{trace}")
            # Return useful information even if error occurs
            return f"\n\n### Solution Process Summary ###\nUnable to generate summary: {error_type}: {error_msg}\n\n"
    
    # Create result dictionary
    result = {
        "task_id": example["task_id"],
        "task": example["task"],
        "question": example["question"],
        "answer": output,
        "true_answer": example["true_answer"],
        "is_correct": is_correct,
        "reasoning": " ".join(intermediate_steps),
        "file_name": example.get("file_name", ""),
        "file_type": example.get("file_type", ""),
        "file_tool": example.get("file_tool", ""),
        "data_type": data_type,
        "data_requirement": data_requirement,
        "answer_type": answer_type,
        "model_id": model_id,
        "timestamp": time.time()
    }
    
    # Use built-in function to generate question summary
    summary = generate_summary_internal(result)
    result["summary"] = summary
    
    # Save results (JSONL and Excel)
    task_logger.info("Saving results to JSONL and Excel files...")
    append_answer(result, answers_file)
    
    # Save TXT format log
    task_logger.info("Saving results to TXT file...")
    txt_file = answers_file.replace(".jsonl", ".txt")
    with open(txt_file, "a", encoding="utf-8") as f:
        f.write(f"Question ID: {example['task_id']}\n")
        f.write(f"Question: {example['question']}\n")
        f.write(f"Answer type: {answer_type}\n")
        f.write(f"Data requirement: {data_requirement}\n")
        f.write(f"Data type: {data_type}\n")
        f.write(f"Our answer: {output}\n")
        f.write(f"Correct answer: {example['true_answer']}\n")
        # f.write(f"Reasoning process: {' '.join(intermediate_steps)}\n")
        f.write(f"Is correct: {'✓' if is_correct else '✗'}\n")
        f.write(f"File: {example.get('file_name', '')}\n")
        f.write(f"File type: {example.get('file_type', '')}\n")
        f.write(f"Tools used: {example.get('file_tool', '')}\n")
        f.write(f"Model: {model_id}\n")
        f.write(f"Timestamp: {time.time()}\n")
        f.write(summary)
        f.write("\n" + "-"*50 + "\n\n")
    
    # Create separate answer files for each answer
    task_logger.info("Creating separate answer files...")
    # Create separate answer directory
    single_answers_dir = os.path.join(os.path.dirname(answers_file), "single_answers")
    os.makedirs(single_answers_dir, exist_ok=True)
    
    # Get question ID as filename
    task_id = str(example['task_id'])
    
    # Save separate JSON file
    json_file = os.path.join(single_answers_dir, f"{task_id}.json")
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    # Save separate TXT file
    txt_file = os.path.join(single_answers_dir, f"{task_id}.txt")
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write(f"Question ID: {example['task_id']}\n")
        f.write(f"Question: {example['question']}\n")
        f.write(f"Answer type: {answer_type}\n")
        f.write(f"Data requirement: {data_requirement}\n")
        f.write(f"Data type: {data_type}\n")
        f.write(f"Our answer: {output}\n")
        f.write(f"Correct answer: {example['true_answer']}\n")
        f.write(f"Is correct: {'✓' if is_correct else '✗'}\n")
        f.write(f"File: {example.get('file_name', '')}\n")
        f.write(f"File type: {example.get('file_type', '')}\n")
        f.write(f"Tools used: {example.get('file_tool', '')}\n")
        f.write(f"Model: {model_id}\n")
        f.write(f"Timestamp: {time.time()}\n")
        f.write(summary)
    
    task_logger.info(f"Created separate answer files: {json_file} and {txt_file}")
    
    # Update statistics - update statistics after each question
    output_dir = os.path.dirname(answers_file)
    update_statistics(answers_file, args.run_name, output_dir)
    
    task_logger.info(f"Task {task_id} completed, correct: {is_correct}")
    return result

def get_examples_to_answer(answers_file, eval_ds, args=None) -> List[dict]:
    # Get main logger
    logger = logging.getLogger("main")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    
    logger.info(f"Loading answers from {answers_file}...")
    try:
        # If file doesn't exist, raise exception, enter except block
        done_questions = pd.read_json(answers_file, lines=True)["question"].tolist()
        logger.info(f"Found {len(done_questions)} previous results!")
    except Exception as e:
        logger.error(f"Error when loading records: {e}")
        logger.info("No usable records! ▶️ Starting new.")
        # Ensure file exists, even if it's empty
        Path(answers_file).touch()
        done_questions = []
    
    # Filter out already completed questions
    examples = [line for line in eval_ds.to_list() if line["question"] not in done_questions]
    
    # If command line arguments are provided, filter questions based on ID or range
    if args:
        filtered_examples = []
        level_prefix = f"{args.level}_"  # For example, "level1_"
        
        # Process specific ID list
        if args.question_ids:
            question_ids = [id.strip() for id in args.question_ids.split(',')]
            logger.info(f"Filtering specific question IDs: {question_ids}")
            
            # Convert numeric IDs to full ID format (e.g., "16" -> "level_1_16")
            full_ids = []
            # Modify level_prefix format
            level_prefix = f"level_{args.level.replace('level', '')}_"  # For example, "level_1_"
            # print(level_prefix)
            for id in question_ids:
                if id.startswith(level_prefix):
                    full_ids.append(id)
                else:
                    full_ids.append(f"{level_prefix}{id}")
            # Filter questions
            for example in examples:
                if example.get("task_id") in full_ids:
                    filtered_examples.append(example)
        
        # Process ID range
        elif args.start_id is not None or args.end_id is not None:
            start_id = args.start_id if args.start_id is not None else 1
            end_id = args.end_id if args.end_id is not None else float('inf')
            
            logger.info(f"Filtering question IDs from {start_id} to {end_id}")
            level_prefix = f"level_{args.level.replace('level', '')}_"
            for example in examples:
                task_id = example.get("task_id", "")
                print(task_id, level_prefix)
                if task_id.startswith(level_prefix):
                    try:
                        # Extract numeric part
                        id_num = int(task_id[len(level_prefix):])
                        print(id_num)
                        if start_id <= id_num <= end_id:
                            filtered_examples.append(example)
                    except ValueError:
                        # If ID format is incorrect, skip
                        continue
        
        # If filtering is applied, use filtered list
        if args.question_ids or args.start_id is not None or args.end_id is not None:
            logger.info(f"Filtered question count: {len(filtered_examples)}/{len(examples)}")
            return filtered_examples
    
    return examples


def analyze_results(answers_file):
    """
    Analyze the results file, count the number and percentage of correct answers
    
    Parameters:
        answers_file: Path to the results file (JSONL format)
        
    Returns:
        dict: Dictionary containing statistics
    """
    # Get main logger
    logger = logging.getLogger("main")
    
    # Default return value (empty results)
    default_result = {"total": 0, "correct": 0, "accuracy": 0, "by_task": {}, "by_file_type": {}}
    
    try:
        # Check if file exists
        if not os.path.exists(answers_file):
            logger.warning(f"Results file does not exist: {answers_file}")
            return default_result
        
        # Check if file is empty
        if os.path.getsize(answers_file) == 0:
            logger.warning(f"Results file is empty: {answers_file}")
            return default_result
            
        # Read results file
        logger.debug(f"Analyzing results file: {answers_file}")
        results = []
        with open(answers_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): 
                    continue
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"Warning: Skipping invalid JSON line")
        
        if not results:
            logger.warning("Results file is empty or format is incorrect")
            return default_result
        
        # Count total accuracy
        total = len(results)
        correct = sum(1 for r in results if r.get("is_correct", False))
        accuracy = correct / total if total > 0 else 0
        
        # Count by task type
        by_task = {}
        for r in results:
            task = r.get("task", "Unknown")
            if task not in by_task:
                by_task[task] = {"total": 0, "correct": 0, "accuracy": 0}
            
            by_task[task]["total"] += 1
            if r.get("is_correct", False):
                by_task[task]["correct"] += 1
        
        # Calculate accuracy for each task
        for task in by_task:
            task_total = by_task[task]["total"]
            task_correct = by_task[task]["correct"]
            by_task[task]["accuracy"] = task_correct / task_total if task_total > 0 else 0
        
        # Count by file type
        by_file_type = {}
        for r in results:
            file_type = r.get("file_type", "No file")
            if not file_type:
                file_type = "No file"
                
            if file_type not in by_file_type:
                by_file_type[file_type] = {"total": 0, "correct": 0, "accuracy": 0}
            
            by_file_type[file_type]["total"] += 1
            if r.get("is_correct", False):
                by_file_type[file_type]["correct"] += 1
        
        # Calculate accuracy for each file type
        for file_type in by_file_type:
            type_total = by_file_type[file_type]["total"]
            type_correct = by_file_type[file_type]["correct"]
            by_file_type[file_type]["accuracy"] = type_correct / type_total if type_total > 0 else 0
        
        # Return statistics
        stats = {
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
            "by_task": by_task,
            "by_file_type": by_file_type
        }
        
        return stats
    
    except Exception as e:
        logger.error(f"Error analyzing results: {e}")
        import traceback
        logger.debug(f"Detailed error: {traceback.format_exc()}")
        return default_result

def print_statistics(stats):
    """
    Print statistics
    
    Parameters:
        stats: Statistics dictionary
    """
    # Get main logger
    logger = logging.getLogger("main")
    
    logger.info("\n" + "="*50)
    logger.info("Results statistics")
    logger.info("="*50)
    
    # Total statistics
    logger.info(f"\nTotal questions: {stats['total']}")
    logger.info(f"Correct answers: {stats['correct']}")
    logger.info(f"Total accuracy: {stats['accuracy']*100:.2f}%")
    
    # Count by task type
    logger.info("\nCount by task type:")
    logger.info("-"*40)
    logger.info(f"{'Task type':<20} {'Total':<8} {'Correct':<8} {'Accuracy':<10}")
    logger.info("-"*40)
    for task, data in sorted(stats['by_task'].items(), key=lambda x: x[1]['accuracy'], reverse=True):
        logger.info(f"{task[:20]:<20} {data['total']:<8} {data['correct']:<8} {data['accuracy']*100:.2f}%")
    
    # Count by file type
    logger.info("\nCount by file type:")
    logger.info("-"*40)
    logger.info(f"{'File type':<20} {'Total':<8} {'Correct':<8} {'Accuracy':<10}")
    logger.info("-"*40)
    for file_type, data in sorted(stats['by_file_type'].items(), key=lambda x: x[1]['accuracy'], reverse=True):
        logger.info(f"{file_type[:20]:<20} {data['total']:<8} {data['correct']:<8} {data['accuracy']*100:.2f}%")
    
    logger.info("\n" + "="*50)

def export_statistics_to_file(stats, output_file):
    """
    Export statistics to a file, including detailed run information and recent results
    
    Parameters:
        stats: Statistics dictionary
        output_file: Output file path
    """
    # Get main logger
    logger = logging.getLogger("main")
    
    # Get current time
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Try to get details of recently processed questions
    latest_results = []
    answers_file = output_file.replace("_stats.txt", ".jsonl")
    try:
        with open(answers_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # Get the last up to 5 processed results
            for line in lines[-5:] if len(lines) >= 5 else lines:
                latest_results.append(json.loads(line))
    except Exception as e:
        logger.error(f"Error reading recent results: {e}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write title and timestamp
        f.write(f"Statistics - Updated: {current_time}\n")
        f.write("="*80 + "\n\n")
        
        # Overall statistics
        f.write(f"Total questions: {stats['total']}\n")
        f.write(f"Correct answers: {stats['correct']}\n")
        f.write(f"Incorrect answers: {stats['total'] - stats['correct']}\n")
        f.write(f"Overall accuracy: {stats['accuracy']*100:.2f}%\n\n")
        
        # Recently processed questions
        if latest_results:
            f.write("Recently processed questions:\n")
            f.write("-"*80 + "\n")
            for result in reversed(latest_results):  # Show newest first
                task_id = result.get('task_id', 'unknown')
                question = result.get('question', 'unknown')[:100] + '...' if len(result.get('question', '')) > 100 else result.get('question', 'unknown')
                is_correct = "✓" if result.get('is_correct', False) else "✗"
                processed_time = datetime.fromtimestamp(result.get('timestamp', 0)).strftime("%Y-%m-%d %H:%M:%S") if result.get('timestamp') else 'unknown'
                
                f.write(f"Question ID: {task_id} | Result: {is_correct} | Time: {processed_time}\n")
                f.write(f"Question: {question}\n")
                f.write("-"*40 + "\n")
            f.write("\n")
        
        # Statistics by task type - sorted by accuracy in descending order
        f.write("Statistics by task type:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Task type':<30} {'Total':<8} {'Correct':<8} {'Accuracy':<10}\n")
        f.write("-"*80 + "\n")
        for task, data in sorted(stats['by_task'].items(), key=lambda x: x[1]['accuracy'], reverse=True):
            f.write(f"{task[:30]:<30} {data['total']:<8} {data['correct']:<8} {data['accuracy']*100:.2f}%\n")
        
        # Statistics by file type - sorted by accuracy in descending order
        f.write("\nStatistics by file type:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'File type':<20} {'Total':<8} {'Correct':<8} {'Accuracy':<10}\n")
        f.write("-"*80 + "\n")
        for file_type, data in sorted(stats['by_file_type'].items(), key=lambda x: x[1]['accuracy'], reverse=True):
            f.write(f"{file_type[:20]:<20} {data['total']:<8} {data['correct']:<8} {data['accuracy']*100:.2f}%\n")
    
    logger.info(f"Statistics exported to: {output_file}")

# Add logging system
def setup_logging(output_dir, run_name):
    """Set up logging system"""
    # Create log directory
    log_dir = os.path.join(output_dir)
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create main log file handler
    main_log_file = os.path.join(log_dir, f"main_{run_name}.log")
    file_handler = logging.FileHandler(main_log_file, encoding='utf-8')
    
    # Create console handler
    console_handler = logging.StreamHandler()
    
    # Create command line output log file handler
    cmd_log_file = os.path.join(log_dir, f"cmd_{run_name}.log")
    cmd_handler = logging.FileHandler(cmd_log_file, encoding='utf-8')
    
    # Set formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    cmd_handler.setFormatter(formatter)
    
    # Add handlers to root logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(cmd_handler)
    
    # Create a dedicated command line output logger
    cmd_logger = logging.getLogger('cmd_output')
    cmd_logger.setLevel(logging.INFO)
    cmd_logger.addHandler(cmd_handler)
    cmd_logger.propagate = False  # Prevent propagation to root logger
    
    return log_dir

def get_task_logger(log_dir, task_id):
    """Get task-specific logger"""
    # Create task-specific logger
    logger = logging.getLogger(f"task_{task_id}")
    
    # If handlers have been configured, return immediately
    if logger.handlers:
        return logger
        
    # Set level
    logger.setLevel(logging.INFO)
    
    # Create file handler
    task_log_file = os.path.join(log_dir, f"{task_id}.log")
    file_handler = logging.FileHandler(task_log_file, encoding='utf-8')
    
    # Set formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add handler
    logger.addHandler(file_handler)
    
    # Set not to propagate to parent (avoid duplicate logs)
    logger.propagate = True
    
    return logger

def update_statistics(answers_file, run_name, output_dir):
    """
    Update, display, and save statistics after each question is processed.
    This function is thread-safe to handle concurrent updates.
    
    Parameters:
        answers_file: Path to the results JSONL file
        run_name: Name of the current run 
        output_dir: Directory where to save statistics file
    """
    # Get the main logger
    logger = logging.getLogger("main")
    
    # Static variable to track the last update time for throttling
    if not hasattr(update_statistics, "last_update_time"):
        update_statistics.last_update_time = 0
    
    # Check if file exists and is not empty
    if not os.path.exists(answers_file) or os.path.getsize(answers_file) == 0:
        logger.warning(f"Statistics update skipped: {answers_file} doesn't exist or is empty")
        return
    
    # Use a lock to prevent multiple threads from updating statistics simultaneously
    with append_answer_lock:
        try:
            # Throttle updates - minimum 1 second between full updates
            current_time = time.time()
            time_since_last_update = current_time - update_statistics.last_update_time
            
            # Always update the stats file, but only log detailed stats if enough time has passed
            full_update = time_since_last_update >= 1.0
            
            # Calculate updated statistics
            stats = analyze_results(answers_file)
            
            # Skip update if no results found
            if stats['total'] == 0:
                logger.warning("Statistics update skipped: No results found")
                return
            
            # Save statistics to file (do this regardless of throttling)
            stats_file = answers_file.replace(".jsonl", "_stats.txt")
            export_statistics_to_file(stats, stats_file)
            
            # For throttled updates, just show a brief message
            if not full_update:
                logger.info(f"Stats updated: {stats['total']} qs | Acc: {stats['accuracy']*100:.2f}% | File: {stats_file}")
                return
                
            # Update last update time for full updates
            update_statistics.last_update_time = current_time
            
            # Get the most recent result (last line in the file)
            latest_result = None
            try:
                with open(answers_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if lines:
                        latest_result = json.loads(lines[-1])
            except Exception as e:
                logger.error(f"Error reading latest result: {e}")
            
            # Display progress header with timestamp
            current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.info("\n" + "="*25 + f" STATISTICS UPDATE [{current_time_str}] " + "="*25)
            logger.info(f"Run: {run_name}")
            
            # Display overall statistics
            logger.info(f"📊 PROGRESS: {stats['total']} questions processed | ✓ {stats['correct']} correct | ❌ {stats['total'] - stats['correct']} incorrect")
            logger.info(f"📈 ACCURACY: {stats['accuracy']*100:.2f}%")
            
            # Display information about the latest processed question if available
            if latest_result:
                logger.info(f"\n🔄 LAST PROCESSED: Question {latest_result.get('task_id', 'unknown')}")
                logger.info(f"   Result: {'✓ CORRECT' if latest_result.get('is_correct', False) else '❌ INCORRECT'}")
                
                # Display accuracy by task type (for the task types with at least 2 questions)
                task_stats = {task: data for task, data in stats['by_task'].items() if data['total'] >= 2}
                if task_stats:
                    logger.info("\n📋 ACCURACY BY TASK TYPE (with 2+ questions):")
                    for task, data in sorted(task_stats.items(), key=lambda x: x[1]['accuracy'], reverse=True):
                        logger.info(f"   {task[:20]:<20}: {data['accuracy']*100:.2f}% ({data['correct']}/{data['total']})")
            
            logger.info(f"\n📝 Detailed statistics saved to {stats_file}")
            logger.info("="*80)
            
        except Exception as e:
            logger.error(f"Error updating statistics: {e}")
            import traceback
            logger.debug(f"Detailed error: {traceback.format_exc()}")

def main():
    """Run the main program."""
    # Parse arguments
    args = parse_args()
    
    # Set SET based on level parameter
    global SET
    SET = f"{args.level}_final_summary"
    
    # Convert args.level to the format required by dataset_loader.py (convert "level2" to "level 2")
    sheet_name = args.level.replace("level", "level ")
    
    # Create the output directory based on baseline parameter
    if args.baseline:
        output_dir = Path(f"output_baseline/{SET}")
    else:
        output_dir = Path(args.output_dir) / SET
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create LOG_DIR
    global LOG_DIR
    LOG_DIR = output_dir / "logs"
    LOG_DIR.mkdir(exist_ok=True, parents=True)
    
    # Setup logging
    setup_logging(LOG_DIR, args.run_name)
    logger = logging.getLogger("main")
    
    os.makedirs("output_logs", exist_ok=True)
    # Execute command (no user confirmation needed)
    if hasattr(args, 'command') and args.command:
        success = run_terminal_cmd(
            command=args.command,
            output_dir= "output_logs",
            run_name=args.run_name,
            is_background=True,
            is_main_execution=True
        )
        if not success:
            logger.error("Command execution failed")
            sys.exit(1)
    
    # Log baseline mode if active
    if args.baseline:
        logger.info("Running in baseline mode")
    
    # Log level information
    logger.info(f"Testing questions from level: {args.level}")
    
    # Check if we are under a Tailscale VPN, which could block some URLs
    logger.warning("Make sure you deactivated Tailscale VPN, else some URLs will be blocked!")
    
    # Start time for the entire run
    start_time = time.time()
    
    # Log start info
    logger.info(f"Starting run with arguments: {args}")

    # Print path information for debugging
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Relative path: {RELATIVE_EXCEL_PATH}")
    logger.info(f"Absolute path: {EXCEL_PATH}")
    
    # Check if file exists
    if os.path.exists(EXCEL_PATH):
        logger.info(f"File exists, continue processing")
    else:
        logger.error(f"Error: File does not exist! Please check if the path is correct")
        # Try listing possible locations
        possible_dirs = [".", "Historical", "examples/open_deep_research", "../"]
        for dir_path in possible_dirs:
            try:
                files = os.listdir(dir_path)
                logger.info(f"Files in directory '{dir_path}': {files}")
            except Exception as e:
                logger.error(f"Cannot list files in directory '{dir_path}': {e}")
        return
    
    # Load custom Excel dataset, passing results JSON path
    eval_ds = load_custom_dataset(EXCEL_PATH, test_mode=False, results_json_path=args.results_json_path, sheet_name=sheet_name)
    
    # Define output file paths
    answers_file = f"{output_dir}/{args.run_name}.jsonl"
    txt_file = answers_file.replace(".jsonl", ".txt")
    
    # Check if results file already exists, if so analyze results
    if os.path.exists(answers_file) and os.path.getsize(answers_file) > 0:
        logger.info(f"Detected existing results file: {answers_file}")
        stats = analyze_results(answers_file)
        print_statistics(stats)
        
        # Export statistics to file
        stats_file = answers_file.replace(".jsonl", "_stats.txt")
        export_statistics_to_file(stats, stats_file)
        
        # Ask whether to continue running
        response = input("Continue running the test? (y/n): ")
        if response.lower() != 'y':
            logger.info("User chose to exit")
            return
        
        # Modification: If continuing to run, don't clear TXT file, append content
        with open(txt_file, "a", encoding="utf-8") as f:
            f.write(f"\n\nContinuing test run: {args.run_name}\n")
            f.write(f"Model: {args.model_id}\n")
            f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    else:
        # If it's a new run, create new TXT file
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write(f"Test run: {args.run_name}\n")
            f.write(f"Model: {args.model_id}\n")
            # f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    tasks_to_run = get_examples_to_answer(answers_file, eval_ds, args)

    with ThreadPoolExecutor(max_workers=args.concurrency) as exe:
        futures = [
            exe.submit(answer_single_question, example, args.model_id, answers_file, visualizer, args)
            for example in tasks_to_run
        ]
        for f in tqdm(as_completed(futures), total=len(tasks_to_run), desc="Processing tasks"):
            f.result()

    logger.info("All tasks completed.")

    # Final statistics summary
    logger.info("\nFinal statistics summary...")
    stats = analyze_results(answers_file)
    print_statistics(stats)
    
    # Final statistics are already updated by update_statistics function after each question
    # No need to export statistics results again here
    logger.info(f"Results saved to {answers_file}")

def run_terminal_cmd(command, output_dir, run_name, is_background=False, is_main_execution=False):
    """Execute terminal command and log output
    
    Parameters:
        command: Command to execute
        output_dir: Output directory path
        run_name: Run name, used to generate log filename
        is_background: Whether to run in background
        is_main_execution: Whether executed in main function (if so, no user confirmation needed)
    """
    # Get terminal output logger
    terminal_logger = setup_terminal_logging(output_dir, run_name)
    
    # Log command start execution
    terminal_logger.info(f"Executing command: {command}")
    
    try:
        # If not executed in main function, need user confirmation
        if not is_main_execution:
            print(f"Will execute command: {command}")
            response = input("Continue? (y/n): ")
            if response.lower() != 'y':
                terminal_logger.info("User canceled command execution")
                return False
        
        # Execute command
        if is_background:
            # Run in background
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            terminal_logger.info("Command started in background")
            return process
        else:
            # Synchronous run
            process = subprocess.run(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Log output
            if process.stdout:
                terminal_logger.info(f"Standard output:\n{process.stdout}")
            if process.stderr:
                terminal_logger.error(f"Error output:\n{process.stderr}")
            
            # Log execution result
            terminal_logger.info(f"Command execution completed, return code: {process.returncode}")
            return process.returncode == 0
            
    except Exception as e:
        terminal_logger.error(f"Command execution failed: {str(e)}")
        return False

if __name__ == "__main__":
    # Add test call
    main()
