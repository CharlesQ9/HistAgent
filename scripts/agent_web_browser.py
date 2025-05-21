# Shamelessly stolen from Microsoft Autogen team: thanks to them for this great resource!
# https://github.com/microsoft/autogen/blob/gaia_multiagent_v01_march_1st/autogen/browser_utils.py
import mimetypes
import os
import pathlib
import re
import time
import uuid
import asyncio
import json
import logging
import traceback
import random
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import unquote, urljoin, urlparse

import pathvalidate
import requests
from serpapi import GoogleSearch
from bs4 import BeautifulSoup

from smolagents import Tool
from browser_use import Agent, Browser, BrowserConfig
from langchain_openai import ChatOpenAI
# Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class BrowserAgent:
    """A unified browser agent that handles all web browsing operations."""

    def __init__(
        self,
        start_page: Optional[str] = None,
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
        use_browser: bool = True,
        download_path: Optional[str] = None
    ):
        """
        Initialize the browser agent.

        Args:
            start_page: Initial page to load (defaults to about:blank)
            api_key: OpenAI API key for the agent
            system_prompt: Custom system prompt for the agent
            use_browser: Whether to use a real browser (True) or text-only mode (False)
            download_path: Path to save downloaded files
        """
        self.start_page = start_page if start_page else "about:blank"
        self.api_key = api_key
        self.use_browser = use_browser
        self.download_path = download_path or "downloads"
        self.system_prompt = system_prompt or """
You are a web browser assistant that helps users navigate and extract content from webpages.

**Critical Operational Rules:**

1. **Visit Actual Pages:** Always navigate to and read the content on the actual webpages found in search results.
2. **Extract Directly:** Extract information directly from the webpage content to ensure accuracy.
3. **Navigate Deeply:** If required information is not on the landing page, navigate through multiple pages.
4. **Cite Sources:** Always provide the direct URL for each source.

**Browser Interaction & Timing:**

5. **Random Delays:** Before every page operation (clicking, searching, opening/closing tabs, typing, scrolling), pause for a random duration between 2 and 5 seconds.
6. **Post-Close Delay:** After closing a tab, wait another random duration before the next action.

**Task Execution Guidelines:**

7. **Focus on Content:** When accessing a page, extract its textual content and structure.
8. **Handle Errors:** If a page fails to load, provide error information and suggest alternatives.
9. **Download Files:** When requested, download files to the specified directory.
10. **Search Functionality:** Support searching within pages and Google searches.
11. **Page Navigation:** Support scrolling up/down and viewport management.

**Authentication Handling:**

12. **Avoid Barriers:** If you encounter login walls, CAPTCHA, or paywalls, exit that page immediately.
13. **Do Not Bypass:** Never attempt to bypass security measures or enter credentials.
14. **Note and Proceed:** Record "Authentication required" and try alternative sources.

**PDF Handling:**

15. **Download PDFs:** Download freely accessible PDFs to the specified directory.
16. **Avoid Restricted PDFs:** Do not attempt to download PDFs requiring authentication.
17. **Record Downloads:** Note the filename and location of each downloaded PDF.
"""
        
        # Initialize browser if needed
        if self.use_browser:
            self._init_browser()
        
        # Set initial page
        self.current_url = self.start_page
        
        # Ensure download directory exists
        if self.download_path:
            os.makedirs(self.download_path, exist_ok=True)

    def _init_browser(self):
        """Initialize the browser components."""
        config = BrowserConfig(headless=True)
        self.browser = Browser(config)
        
    async def _get_agent(self, task: str):
        """Get or create the agent as needed."""
        llm = ChatOpenAI(model="o4-mini", temperature=0, api_key=self.api_key)
        agent_kwargs = {
            "task": task,
            "llm": llm,
            "generate_gif": False,
            "extend_system_message": self.system_prompt
        }
        
        if self.use_browser and hasattr(self, 'browser'):
            agent_kwargs["browser"] = self.browser
            
        return Agent(**agent_kwargs)

    async def execute_task(self, task: str, max_steps: int = 38) -> str:
        """
        Execute a browser task using the agent.
        
        Args:
            task: The task to perform
            max_steps: Maximum number of steps the agent can take
            
        Returns:
            String containing the result of the task
        """
        logging.info(f"Executing browser task: {task}")
        
        try:
            llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=self.api_key)
            agent_kwargs = {
                "task": task,
                "llm": llm,
                "generate_gif": False,
                "extend_system_message": self.system_prompt
            }
            
            if self.use_browser and hasattr(self, 'browser'):
                agent_kwargs["browser"] = self.browser
            agent = Agent(**agent_kwargs)
            result = await agent.run(max_steps=max_steps)
            return result
            
        except Exception as e:
            error_msg = f"Error executing browser task: {str(e)}"
            logging.error(error_msg)
            return error_msg

    async def close(self):
        """Close the browser if it's open."""
        if hasattr(self, 'browser'):
            await self.browser.close()
            self.browser = None

class BrowserTool(Tool):
    """A unified tool for web browsing operations."""
    
    name = "browser_task"
    description = "Execute a web browsing task using a browser agent."
    inputs = {
        "task": {"type": "string", "description": "The task to perform (e.g., search, navigate, extract content)"},
        "max_steps": {"type": "integer", "description": "Maximum number of steps the agent can take", "default": 38, "nullable": True}
    }
    output_type = "string"

    def __init__(self, api_key: Optional[str] = None, download_path: Optional[str] = None, use_browser: bool = True):
        """
        Initialize the browser tool.
        
        Args:
            api_key: OpenAI API key
            download_path: Path to download files
            use_browser: Whether to use a real browser
        """
        super().__init__()
        self.agent = BrowserAgent(
            api_key=api_key,
            download_path=download_path,
            use_browser=use_browser
        )

    def forward(self, task: str, max_steps: int = 38) -> str:
        """
        Execute a browser task.
        
        Args:
            task: The task to perform
            max_steps: Maximum number of steps
            
        Returns:
            String containing the task result
        """
        try:
            result = asyncio.run(self.agent.execute_task(task, max_steps))
            self.agent.close()
            return result or "No information found."
        except Exception as e:
            error_msg = f"Error searching information: {str(e)}"
            logging.error(error_msg)
            return error_msg

def create_browser_tool(api_key: Optional[str] = None, download_path: Optional[str] = None) -> Dict[str, Tool]:
    """
    Create a browser tool instance.
    
    Args:
        api_key: OpenAI API key
        download_path: Path to download files
        
    Returns:
        Dictionary containing the browser tool
    """
    return {
        "browser_task": BrowserTool(api_key=api_key, download_path=download_path)
    }

if __name__ == "__main__":
    import os
    import argparse
    from dotenv import load_dotenv
    
    # Load environment variables from .env file
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Run browser agent for testing")
    parser.add_argument("--task", type=str, default=""""Visit this URL: https://siegeofjerusalem.org/SJC-description.html and extract all information about the book of manners associated with British Library MS Cotton Caligula A. ii.""", help="Task to perform with the browser agent")
    parser.add_argument("--max_steps", type=int, default=38, help="Maximum number of steps to execute")
    parser.add_argument("--download_path", type=str, default="downloads", help="Path to download files")
    
    args = parser.parse_args()
    
    # Get API key from environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        exit(1)
    
    # Create and run browser tool
    browser_tool = BrowserTool(api_key=api_key, download_path=args.download_path)
    result = browser_tool.forward(args.task, args.max_steps)
    
    print("\n=== BROWSER TASK RESULT ===")
    print(result)