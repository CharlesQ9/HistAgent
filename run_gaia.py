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
from scripts.frame_extract import VideoFrameExtractorTool
import re
import traceback
import numpy as np
import pandas as pd


import datasets
from datasets import Dataset
from dotenv import load_dotenv
from huggingface_hub import login, snapshot_download
from smolagents import CodeAgent
from smolagents.agents import ToolCallingAgent
from scripts.reformulator import prepare_response
from scripts.run_agents import (
    get_single_file_description,
    get_zip_description,
)
import smolagents.monitoring
print(smolagents.monitoring.AgentLogger)
from smolagents.monitoring import AgentLogger
from smolagents.monitoring import LogLevel
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
    DownloadTool_Image,
    ArchiveSearchTool_Image,
    PageUpTool_Image,
    PageDownTool_Image,
    FinderTool_Image,
    FindNextTool_Image,
    ImageContextExtractorTool,
    VisitImageSearchResultsTool,
    SaveHTMLTool,
)
from scripts.file_processing import (
    FileProcessor,
    # SpeechRecognitionTool,
    OCRTool,
    # TranslatorTool,
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
    # HfApiModel,
    LiteLLMModel,
    Model,
)

import openai 
import base64
from smolagents.models import MessageRole
from dataset_loader import load_custom_dataset
# from scripts.web_tools import literature_searching_task, general_browser_task, relevant_literature_finder
from scripts.translator import TranslatorTool
from scripts.speech_recognition import SpeechRecognitionTool
from scripts.ocr import OCRTool

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
    parser.add_argument("--concurrency", type=int, default=8)
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
    parser.add_argument("--set-to-run", type=str, default="validation")
    parser.add_argument("--use-open-models", type=bool, default=False)
    parser.add_argument("--use-raw-dataset", action="store_true")
    return parser.parse_args()

args = parse_args()




### IMPORTANT: EVALUATION SWITCHES

print("Make sure you deactivated Tailscale VPN, else some URLs will be blocked!")

USE_OPEN_MODELS = False

SET = None

custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}

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
"""

    return manager_agent


def load_gaia_dataset(use_raw_dataset: bool, set_to_run: str) -> datasets.Dataset:
    if not os.path.exists("data/gaia"):
        if use_raw_dataset:
            snapshot_download(
                repo_id="gaia-benchmark/GAIA",
                repo_type="dataset",
                local_dir="data/gaia",
                ignore_patterns=[".gitattributes", "README.md"],
            )
        else:
            # WARNING: this dataset is gated: make sure you visit the repo to require access.
            snapshot_download(
                repo_id="smolagents/GAIA-annotated",
                repo_type="dataset",
                local_dir="data/gaia",
                ignore_patterns=[".gitattributes", "README.md"],
            )

    def preprocess_file_paths(row):
        if len(row["file_name"]) > 0:
            row["file_name"] = f"data/gaia/{set_to_run}/" + row["file_name"]
        return row

    eval_ds = datasets.load_dataset(
        "data/gaia/GAIA.py",
        name="2023_all",
        split=set_to_run,
        # data_files={"validation": "validation/metadata.jsonl", "test": "test/metadata.jsonl"},
    )

    eval_ds = eval_ds.rename_columns({"Question": "question", "Final answer": "true_answer", "Level": "task"})
    eval_ds = eval_ds.map(preprocess_file_paths)
    print(eval_ds)
    return eval_ds


def append_answer(entry: dict, jsonl_file: str) -> None:
    jsonl_path = Path(jsonl_file)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with append_answer_lock, open(jsonl_file, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(entry) + "\n")
    assert jsonl_path.exists(), "File not found!"
    print("Answer exported to file:", jsonl_path.resolve())

def generate_summary_internal(model, question, answer, reasoning):
    """Generate problem summary"""
    try:
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
    * "Maintaining proper blood sugar levels is crucial for preventing type 2 diabetes." — [Clinic](https://www...)
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
        return f"\n\n### Solution Process Summary ###\n{summary_text}\n\n"
    except Exception as e:
        # Detailed error information
        error_type = type(e).__name__
        error_msg = str(e)
        import traceback
        trace = traceback.format_exc()
        # Return useful information even if error occurs
        return f"\n\n### Solution Process Summary ###\nUnable to generate summary: {error_type}: {error_msg}\n\n"



def answer_single_question(
    example: dict, model_id: str, answers_file: str, visual_inspection_tool: TextInspectorTool
) -> None:
    model_id = "gpt-4o"
    model_params: dict[str, Any] = {
        "model_id": model_id,
        "custom_role_conversions": custom_role_conversions,
    }
    if model_id == "o1":
        model_params["reasoning_effort"] = "high"
        model_params["max_completion_tokens"] = 8192
    else:
        model_params["max_tokens"] = 4096
    model = LiteLLMModel(**model_params)
    # model = InferenceClientModel(model_id="Qwen/Qwen3-32B", provider="novita", max_tokens=4096)
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
                                  )

    augmented_question = """You have one question to answer. It is paramount that you provide a correct answer.
Give it all you can: I know for a fact that you have access to all the relevant tools to solve it and find the correct answer (the answer does exist).
Failure or 'I cannot answer' or 'None found' will not be tolerated, success will be rewarded.
Run verification steps if that's needed, you must make sure you find the correct answer! Here is the task:

""" + example["question"]

    if example["file_name"]:
        if ".zip" in example["file_name"]:
            prompt_use_files = "\n\nTo solve the task above, you will have to use these attached files:\n"
            prompt_use_files += get_zip_description(
                example["file_name"], example["question"], visual_inspection_tool, document_inspection_tool
            )
        else:
            prompt_use_files = "\n\nTo solve the task above, you will have to use this attached file:\n"
            prompt_use_files += get_single_file_description(
                example["file_name"], example["question"], visual_inspection_tool, document_inspection_tool
            )
        augmented_question += prompt_use_files

    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        # Run agent
        final_result = agent.run(augmented_question)

        agent_memory = agent.write_memory_to_messages()

        final_result = prepare_response(augmented_question, agent_memory, reformulation_model=model)

        output = str(final_result)
        for memory_step in agent.memory.steps:
            memory_step.model_input_messages = None
        intermediate_steps = agent_memory

        summary = generate_summary_internal(model, augmented_question, output, agent_memory)

        # Check for parsing errors which indicate the LLM failed to follow the required format
        parsing_error = True if any(["AgentParsingError" in step for step in intermediate_steps]) else False

        # check if iteration limit exceeded
        iteration_limit_exceeded = True if "Agent stopped due to iteration limit or time limit." in output else False
        raised_exception = False

    except Exception as e:
        print("Error on ", augmented_question, e)
        output = None
        intermediate_steps = []
        parsing_error = False
        iteration_limit_exceeded = False
        exception = e
        raised_exception = True
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    token_counts_manager = agent.monitor.get_total_token_counts()
    token_counts_web = list(agent.managed_agents.values())[0].monitor.get_total_token_counts()
    total_token_counts = {
        "input": token_counts_manager["input"] + token_counts_web["input"],
        "output": token_counts_manager["output"] + token_counts_web["output"],
    }
    
    annotated_example = {
        "task_id": example["task_id"],
        "model_answer": output,
        "reasoning_trace": intermediate_steps
    }
    append_answer(annotated_example, answers_file)
    txt_file = answers_file.replace(".jsonl", ".txt")
    with open(txt_file, "a", encoding="utf-8") as f:
        f.write(f"Question ID: {example['task_id']}\n")
        f.write(f"Question: {example['question']}\n")
        f.write(f"Our answer: {output}\n")
        f.write(f"Correct Answer: {example['true_answer']}\n")
        f.write(f"File: {example.get('file_name', '')}\n")
        f.write(f"Model: {model_id}\n")
        f.write(summary)
        steps_str = "\n".join([str(step) for step in intermediate_steps])
        f.write(f"Reasoning: {steps_str}\n")
        f.write("\n" + "-"*50 + "\n\n")


def get_examples_to_answer(answers_file: str, eval_ds: datasets.Dataset) -> list[dict]:
    print(f"Loading answers from {answers_file}...")
    try:
        done_questions = pd.read_json(answers_file, lines=True)["question"].tolist()
        print(f"Found {len(done_questions)} previous results!")
    except Exception as e:
        print("Error when loading records: ", e)
        print("No usable records! ▶️ Starting new.")
        done_questions = []
    # return [line for line in eval_ds.to_list() if line["question"] not in done_questions]
    return eval_ds.to_list()[20:60]


def main():
    args = parse_args()
    print(f"Starting run with arguments: {args}")

    eval_ds = load_gaia_dataset(args.use_raw_dataset, args.set_to_run)
    print("Loaded evaluation dataset:")
    print(pd.DataFrame(eval_ds)["task"].value_counts())

    answers_file = f"gaia_output/{args.set_to_run}/{args.run_name}.jsonl"
    tasks_to_run = get_examples_to_answer(answers_file, eval_ds)
    print(len(tasks_to_run))

    with ThreadPoolExecutor(max_workers=args.concurrency) as exe:
        futures = [
            exe.submit(answer_single_question, example, args.model_id, answers_file, visualizer)
            for example in tasks_to_run
        ]
        for f in tqdm(as_completed(futures), total=len(tasks_to_run), desc="Processing tasks"):
            f.result()

    # for example in tasks_to_run:
    #     answer_single_question(example, args.model_id, answers_file, visualizer)
    print("All tasks processed.")


if __name__ == "__main__":
    main()
