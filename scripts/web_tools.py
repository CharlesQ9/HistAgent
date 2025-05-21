import asyncio
import re
import json
import os
from typing import Dict, Any, List, Optional, Tuple
import logging
import traceback
import random
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote, urlparse
from ratelimit import limits, sleep_and_retry
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

from smolagents import tool, Tool
from browser_use import Agent, Browser, BrowserConfig
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from environment or use default
API_KEY = os.getenv("OPENAI_API_KEY", "")

class LiteratureSearchBrowser:
    """Browser to search for literature."""
    
    def __init__(self, api_key=None, system_prompt=None, download_path=None, use_browser=True):
        """
        Initialize with OpenAI API key.
        
        Args:
            api_key: OpenAI API key
            download_path: Path to download PDF files
            use_browser: Whether to use a browser agent (set to False to use a text-only agent)
        """
        self.api_key = api_key
        self.use_browser = use_browser
        self.download_path = download_path or "literature_downloads"
        self.system_prompt = system_prompt or f"""
**ScholarBot Academic Research Protocol:**

1.  **Role:** You are operating as ScholarBot, an academic research assistant. Your primary goal is to find and retrieve detailed academic content based on the user's query, using web Browse.
2.  **Core Task:** Locate and extract specific information, primarily exact quotes or data, from scholarly sources like books and academic articles.

**Critical Operational Rules:**

3.  **Deep Dive Required:** Always navigate *into* the actual article or book pages found in search results. Do not rely solely on information presented in search result snippets.
4.  **Direct Extraction:** Extract information *directly* from the content of the article/book page (full text or abstract), not from intermediate search pages.
5.  **Prioritize Full Access:** Attempt to access the full text of sources whenever available. If full text is inaccessible, use the complete abstract page as the minimum source.
6.  **Verbatim Recording:** When you find text that directly matches or answers the query, record it *verbatim*, preserving the exact wording from the source.
7.  **Cite Location:** Always note the page number(s) or specific location (e.g., section, paragraph) within the source where the extracted information was found.
8.  **PDF Handling:**
    * Identify freely accessible PDF files relevant to the query.
    * Download these accessible PDFs to the designated download directory: {{self.download_path}}.
    * Do *not* attempt to download files requiring login, payment, or subscription.
    * Record the filename of each successfully downloaded PDF in your memory/final report.
9.  **Authentication Avoidance:**
    * If you encounter any login prompt, paywall, CAPTCHA, or other authentication barrier, immediately stop interacting with that source.
    * Do *not* attempt to log in, pay, or bypass these barriers.
    * Note "Authentication required" for that source in your memory and move to the next potential source. Focus only on freely accessible content.

**Search Strategy (Execute in this Order):**

10. **Google Domain Rotation:**
    * Maintain this list of Google domains: `www.google.com`, `www.google.ca`, `www.google.fr`, `www.google.co.uk`, `www.google.de`, `www.google.com.au`, `www.google.co.jp`, `www.google.co.in`, `www.google.com.br`, `www.google.ru`, `www.google.it`, `www.google.es`, `www.google.com.mx`, `www.google.co.kr`, `www.google.nl`, `www.google.pl`, `www.google.com.sg`, `www.google.co.za`, `www.google.com.tr`, `www.google.se`.
    * For *every* new search action directed at Google (Books, Scholar, or regular Search), randomly select one domain from this list to use as the base URL.
    * Use a different random domain for each subsequent Google search within the task.
    * Example: Instead of `https://books.google.com/`, use `https://books.[random-domain]/`. Instead of `https://scholar.google.com/`, use `https://scholar.[random-domain]/`. Instead of `https://www.google.com/`, use `https://www.[random-domain]/`.
    * Wait 5-7 seconds between searches initiated on different Google services/domains.

11. **FIRST Search: Google Books:**
    * Navigate to Google Books using a randomly selected domain (e.g., `https://books.google.ca/`).
    * Search using precise keywords from the user's query.
    * **Handling Redirects to Google Search:** If your Google Books search redirects to a standard Google Search page (URL starts with `https://www.google.com/search?...`):
        * Look for a section identified by the HTML: `<div class="VNSPub">Found inside</div>`.
        * If found, extract all subsequent HTML content, focusing on text within `<span>` elements, especially text highlighted with `<em>` tags. Record this extracted text (initially preserving `<em>` tags). These snippets are high-priority results from books.
        * Carefully analyze these snippets. If one provides a direct answer (especially for `exactMatch`), record it and its source, then stop searching.
    * **Check Snippets on Results:** On the Google Books results page itself, examine snippets. If a snippet provides a perfect answer (especially for `exactMatch`), record it and its source, then stop searching.
    * **Explore Previews:** Click into promising results to view book previews. Use the book's internal search function (if available, often in a side panel) to find keywords within the preview.
    * **Fallback/Filter:** If the initial Books search fails or is blocked, perform a search on regular Google (using a random domain) and immediately apply the "Books" filter found below the search bar.
    * Extract relevant quotes/information and page numbers from available previews or snippets.

12. **SECOND Search: Google Scholar (If Books Insufficient):**
    * Navigate to Google Scholar using a randomly selected domain (different from the one used for Books, e.g., `https://scholar.google.fr/`).
    * Search using precise keywords, focusing on academic articles and papers.
    * Attempt to access full text or abstracts.
    * Extract relevant quotes/information and source details (journal, year, page numbers).

13. **THIRD Search: Regular Google Search (If Books/Scholar Insufficient):**
    * Navigate to Google Search using a randomly selected domain (different from previous ones, e.g., `https://www.google.co.uk/`).
    * Search using the query keywords, potentially adding terms like "quote", "excerpt", or "full text".
    * Look for results from educational sites (.edu), institutional repositories, or other credible scholarly sources. Check for alternative versions of texts.
    * Extract relevant quotes/information and source details.

**Browser Interaction & Timing:**

14. **Mandatory Random Delays:** Before *every* page operation (clicking, searching, opening/closing tabs, typing, scrolling), pause for a *random* duration between 3 and 5 seconds. The duration *must* vary each time; do not use the same delay consecutively.
15. **Post-Close Delay:** After closing a tab, wait another random duration (3-5 seconds, different from the previous delay) before performing the next action (like opening a new tab or searching).
16. **Tab Management:** After finishing the analysis of a specific source (book preview, article page), close its tab before proceeding to the next source or search step.

**Special Handling for `exactMatch` Questions:**

17. **Identify `exactMatch`:** Recognize when the task is an `exactMatch` question type, requiring a verbatim answer found in literature.
18. **Remove Blanks Pre-Search:** CRITICAL: Before performing *any* search (Books, Scholar, Regular) for an `exactMatch` question, *always* remove blank placeholders (like "____", "___", or "[BLANK]") from the query string.
    * Example: Query "The Battle of _____ was fought in 1815" becomes search term "The Battle of was fought in 1815".
19. **Prioritize Snippets/Book Results:** For `exactMatch` questions, pay extra attention to:
    * Search result snippets on Google Books or regular Google Search.
    * The "Found inside" section if redirected from Google Books.
    * If a snippet exactly matches the modified query (blanks removed), record the snippet text and source, and consider the task potentially complete – stop searching and report.
20. **Reconstruct Final Answer:** When you find the text in a source that completes the `exactMatch` query:
    * Formulate the final answer by taking the original question format (with blanks).
    * Fill in the blank(s) with the information found in the source.
    * Clearly indicate or emphasize the filled-in information.
    * Example: Found text "The Battle of Waterloo was fought in 1815". Final Answer: "The Battle of **Waterloo** was fought in 1815".
"""
        
    async def _run_task(self, task: str, max_steps: int = 38, download_path: str = None) -> str:
        """
        Run the given task with a browser agent.
        
        Args:
            task: The task to perform
            max_steps: Maximum number of steps the agent can take
            download_path: Path to download PDF files, defaults to self.download_path
            
        Returns:
            String containing the result of the task
        """
        logging.info(f"Running browser task: {task}")
        download_path = download_path or self.download_path
        
        config = BrowserConfig(headless=True)
        browser = Browser(config) if self.use_browser else None
        
        # Ensure download directory exists
        if download_path and not os.path.exists(download_path):
            os.makedirs(download_path, exist_ok=True)
            logging.info(f"Created download directory: {download_path}")
        
        try:
            llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=self.api_key)
            
            # Create agent with or without browser based on use_browser parameter
            agent_kwargs = {
                "task": task,
                "llm": llm,
                "generate_gif": False,
                "extend_system_message": self.system_prompt,
                "save_conversation_path": "./web_tools/web_tools"
            }
            
            # Only add browser parameter if use_browser is True
            if self.use_browser:
                agent_kwargs["browser"] = browser
                
            # Create the agent
            agent = Agent(**agent_kwargs)
            
            # Run the agent
            result = await agent.run(max_steps=max_steps)

            return result
            
        except Exception as e:
            error_msg = f"Error running browser task: {str(e)}"
            logging.error(error_msg)
            traceback.print_exc()
            return error_msg

    async def extract_book_matches(self, query: str, max_steps: int = 38) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Extracts book matches from Google Books search by:
        1. Searching Google Books for the query
        2. Extracting the URL of the search results
        3. Parsing the HTML to find the book matches section
        4. Extracting text with highlighted parts (marked with <em> tags)
        
        Args:
            query: The search query
            max_steps: Maximum number of steps the agent can take
            
        Returns:
            Tuple containing:
            - the Google Books search URL
            - list of book match snippets with their details
        """
        logging.info(f"Extracting book matches for query: {query}")
        
        # Random Google domain selection for Books search
        google_domains = [
            "www.google.com", "www.google.ca", "www.google.fr", "www.google.co.uk", "www.google.de", 
            "www.google.com.au", "www.google.co.jp", "www.google.co.in", "www.google.com.br"
        ]
        
        books_domain = random.choice(google_domains).replace("www.", "books.")
        
        # Create a task for browser agent to search Google Books and return the URL
        search_task = f"""
        Search for information about "{query}" in Google Books.
        
        IMPORTANT: 
        1. Use {books_domain} instead of books.google.com
        2. Each time WAIT a RANDOM duration between 3-5 seconds between pages to avoid rate limiting (never use the same delay twice)
        3. Do NOT access login-required content
        
        SPECIFIC STEPS:
        1. Go to {books_domain}
        2. Search for: "related:{query}" - IMPORTANT: You MUST prefix the query with "related:" to comply with robots.txt allowed paths
        3. If redirected to a regular Google search page (starting with https://www.google.com/search?), that's okay
        4. Check if you see the section labeled "Found inside" (matching results found in books)
        5. COPY THE CURRENT URL and include it in your final report
        6. VERY IMPORTANT: Do not click on any book results - just get the search results URL
        Your final response should be a JSON dictionary with the following format:
        {{
            "search_url": "https://www.google.com/search?tbm=bks&q=related:your_query",
            "book_matches_found": true/false
        }}
        
        Make sure to properly format the JSON so I can parse it directly. Replace true/false with the actual boolean value based on whether the "Found inside" section was visible.
        """
        
        try:
            # Run the browser task to get the search URL
            result = await self._run_task(search_task, max_steps=max_steps)
            for action in result.action_results():
                if action.is_done:
                    result = action.extracted_content
                    print("result", result)
                    print("type", type(result))
                    result = json.loads(result)
                    print("result", result)
                    print("type", type(result))
            # Extract URL from the browser result
            # url_pattern = r'URL: (https?://[^\s]+)'
            # url_match = re.search(url_pattern, result["search_url"])
            
            if not result["book_matches_found"]:
                return "No URL found in search results", []
            
            search_url = result["search_url"]
            logging.info(f"Found search URL: {search_url}")
            
            # Now fetch and parse the HTML from the URL
            try:
                # Set headers to mimic a browser
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                    'Cache-Control': 'max-age=0'
                }
                # print("step1", search_url)
                response = requests.get(search_url, headers=headers)
                response.raise_for_status()  # Raise exception for HTTP errors
                # print("step2", response.text)
                # Parse the HTML content
                soup = BeautifulSoup(response.text, 'html.parser')
                # print("step3", soup)
                # Save soup to a test file for debugging
                # with open("./soup_test_output.html", "w", encoding="utf-8") as f:
                #     f.write(str(soup))
                
                # Find all container elements with class="bHexk Tz5Hvf" as in test.py
                containers = soup.find_all('div', class_='bHexk Tz5Hvf')
                
                book_matches = []
                for container in containers:
                    # Try to find the title and content within this container
                    title_elem = container.find('h3', class_=lambda c: c and 'LC20lb' in c)
                    vnspub_elem = container.find('div', class_='VNSPub')
                    
                    # Only proceed if we found both elements
                    if title_elem and vnspub_elem:
                        book_title = title_elem.get_text(strip=True)
                        
                        # Find the span after VNSPub that contains the content
                        content_span = vnspub_elem.find_next_sibling('span')
                        if content_span:
                            content_text = content_span.get_text(strip=False)
                            
                            # Create book match entry with the same structure as original code
                            book_match = {
                                'title': book_title,
                                'heading': vnspub_elem.get_text(strip=True),
                                'content': content_text
                            }
                            
                            book_matches.append(book_match)
                
                print(f"Found {len(book_matches)} book matches")
                
                for result in book_matches:
                    print(f"\nTitle: {result['title']}")
                    print(f"Heading: {result['heading']}")
                    print(f"Content: {result['content']}")
                    print('-' * 50)
                
                # If no results were found, provide debugging information
                if not book_matches:
                    print("No matching elements found in the HTML.")
                    
                    # Check if the container class exists at all
                    alt_containers = soup.find_all('div', class_=lambda c: c and 'bHexk' in (c.split() if c else []))
                    if alt_containers:
                        print(f"Found {len(alt_containers)} containers with 'bHexk' in class name.")
                    
                    # Show what classes actually exist for potential containers
                    common_classes = {}
                    for div in soup.find_all('div', class_=True):
                        for class_name in div.get('class', []):
                            common_classes[class_name] = common_classes.get(class_name, 0) + 1
                    
                    print("\nMost common div classes in the document:")
                    sorted_classes = sorted(common_classes.items(), key=lambda x: x[1], reverse=True)
                    for class_name, count in sorted_classes[:10]:  # Top 10 classes
                        print(f"  {class_name}: {count} occurrences")
                
                return search_url, book_matches
                
            except Exception as e:
                logging.error(f"Error parsing HTML: {str(e)}")
                return search_url, []
                
        except Exception as e:
            error_msg = f"Error extracting book matches: {str(e)}"
            logging.error(error_msg)
            traceback.print_exc()
            return "", []
    
    async def parse_google_books_url(self, url: str) -> List[Dict[str, Any]]:
        """
        Parses the HTML of a Google Books search results page to extract book match snippets.
        
        Args:
            url: The URL of the Google Books search results page
            
        Returns:
            List of book match snippets with their details
        """
        logging.info(f"Parsing Google Books URL: {url}")
        
        try:
            # Set headers to mimic a browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Cache-Control': 'max-age=0'
            }
            
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            # Parse the HTML content
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find the book matches section
            book_matches_div = soup.find('div', class_='VNSPub', string='Found inside')
            
            if not book_matches_div:
                logging.info("No book matches section found")
                return []
            
            # Get the parent div that contains both the section header and the content
            parent_div = book_matches_div.find_parent('div', class_='cmlJmd ETWPw')
            
            if not parent_div:
                logging.info("Could not find parent div for book matches")
                return []
            
            # Extract all span elements that follow the VNSPub div, which contain the snippets
            spans = parent_div.find_all('span')
            
            book_matches = []
            current_match = {}
            
            # Process each span to extract the text and highlighted parts
            for span in spans:
                if 'VNSPub' in span.get('class', []):
                    continue  # Skip the header span
                    
                # Get text content with <em> tags preserved
                snippet_html = str(span)
                snippet_text = span.get_text()
                
                # Extract highlighted parts (text inside <em> tags)
                em_tags = span.find_all('em')
                highlights = [em.get_text() for em in em_tags]
                
                # Check if this spans contains book title information
                if span.find('a'):
                    # This might be a book title with link
                    book_link = span.find('a').get('href', '')
                    book_title = span.find('a').get_text()
                    
                    # Start a new book match entry
                    if current_match and 'snippet_html' in current_match:
                        book_matches.append(current_match)
                        
                    current_match = {
                        'book_title': book_title,
                        'book_link': book_link
                    }
                elif snippet_text and snippet_text.strip():
                    # This is a text snippet
                    if not current_match:
                        current_match = {}
                    
                    current_match['snippet_html'] = snippet_html
                    current_match['snippet_text'] = snippet_text
                    current_match['highlights'] = highlights
                    
                    # Add to book matches if we have all the key components
                    if 'snippet_html' in current_match:
                        book_matches.append(current_match)
                        current_match = {}
            
            # Add the last match if it wasn't added yet
            if current_match and 'snippet_html' in current_match:
                book_matches.append(current_match)
            
            return book_matches
            
        except Exception as e:
            logging.error(f"Error parsing Google Books URL: {str(e)}")
            return []

class LiteratureSearchingTool(Tool):
    name = "literature_searching_task"
    description = "Search for literature and return the most relevant sources for the query."
    inputs = {
        "query": {"type": "string", "description": "The research query or topic to search for"},
        "max_results": {"type": "integer", "description": "Maximum number of sources to return (default 5)", "default": 5, "nullable": True},
        "max_steps": {"type": "integer", "description": "Maximum number of steps the agent can take", "default": 38, "nullable": True},
        "download_path": {"type": "string", "description": "Path to download PDF files", "default": "literature_downloads", "nullable": True}
    }
    output_type = "string"

    def __init__(self, api_key=None, download_path=None, use_browser=True):
        """
        Initialize the literature searching tool.
        
        Args:
            api_key: OpenAI API key
            download_path: Path to download PDF files
            use_browser: Whether to use a browser agent (set to False to use a text-only agent)
        """
        super().__init__()
        self.download_path = download_path or "literature_downloads"
        self.browser = LiteratureSearchBrowser(api_key=api_key, download_path=download_path, use_browser=use_browser)

    async def _literature_searching_task(self, query: str, max_results: int = 5) -> str:
        """
        Search for literature and return the most relevant sources for the query.
        
        Args:
            query: The research query or topic to search for
            max_results: Maximum number of sources to return (default 5)
            
        Returns:
            String containing the most relevant literature with explanations
        """
        google_domains = [
            "www.google.com", "www.google.ca", "www.google.fr", "www.google.co.uk", "www.google.de", 
            "www.google.com.au", "www.google.co.jp", "www.google.co.in", "www.google.com.br", "www.google.ru",
            "www.google.it", "www.google.es", "www.google.com.mx", "www.google.co.kr", "www.google.nl"
        ]
        
        scholar_domain = random.choice(google_domains).replace("www.", "scholar.")
        
        restricted_task = f"""Search for {max_results} high-impact, recent scholarly articles about: {query}. 
        
        IMPORTANT: Before accessing any website, first check its robots.txt file by visiting domain.com/robots.txt (e.g., https://books.google.com/robots.txt) and respect all directives. Do not access disallowed paths.
        
        IMPORTANT: Instead of using scholar.google.com, use {scholar_domain}
        This helps avoid rate limiting and detection by search engines.
        Wait 5-7 seconds between searches on different domains.
        
        Prioritize:
        1. Most related articles
        2. Highly cited papers (preferably in the top percentile of citations for their publication year)
        3. Recent publications (preferably within the last 3-5 years)
        4. Papers published in reputable journals or conferences
        5. Research from established institutions or authors
        6. Review papers and meta-analyses when appropriate
        
        For each recommended paper, include:
        - Full citation
        - Citation count
        - Publication year
        - Brief description of key findings
        
        Sort results by relevance and citation impact. Return exactly {max_results} articles if possible. Do NOT access login-required sites or paywalled content."""
        
        try:
            # Run the task with BrowserAgentBehavior
            result = asyncio.run(self.browser._run_task(
                restricted_task, 
                max_steps=38, 
                download_path=self.download_path
            ))
            for action in result.action_results():
                logging.info(f"Literature Searching Action: {action}")
                if action.is_done:
                    return action.extracted_content
            return "No literature found."
        except Exception as e:
            error_msg = f"Error searching literature: {str(e)}"
            logging.error(error_msg)
            return error_msg

    def forward(self, query: str, max_results: int = 5, max_steps: int = 38, download_path: str = None) -> str:
        """
        Search for literature and return the most relevant sources for the query.
        
        Args:
            query: The research query or topic to search for
            max_results: Maximum number of sources to return (default 5)
            max_steps: Maximum number of steps the agent can take
            download_path: Path to download PDF files
            
        Returns:
            String containing the most relevant literature with explanations
        """
        logging.info(f"Searching literature for query: {query}")
        actual_download_path = download_path or self.download_path
        
        google_domains = [
            "www.google.com", "www.google.ca", "www.google.fr", "www.google.co.uk", "www.google.de", 
            "www.google.com.au", "www.google.co.jp", "www.google.co.in", "www.google.com.br", "www.google.ru",
            "www.google.it", "www.google.es", "www.google.com.mx", "www.google.co.kr", "www.google.nl"
        ]
        
        # Randomly select domains for each service
        books_domain = random.choice(google_domains).replace("www.", "books.")
        scholar_domain = random.choice([d for d in google_domains if d != books_domain.replace("books.", "www.")]).replace("www.", "scholar.")
        regular_domain = random.choice([d for d in google_domains if d != books_domain.replace("books.", "www.") and d != scholar_domain.replace("scholar.", "www.")])
        
        restricted_task = f"""
**ScholarBot Supplemental Instructions (Refining Base Protocol):**

Based on the specific query '{query}', apply these supplemental instructions that refine or add to your core **ScholarBot Academic Research Protocol**:

1.  **`robots.txt` Check:** Before accessing *any* website domain for the first time (e.g., books.google.com, specific publisher sites), you MUST check its `robots.txt` file (e.g., `https://domain.com/robots.txt`). Strictly adhere to all `Disallow` directives for paths on that domain. Do not access disallowed paths.

2.  **Deep Dive Exception (`exactMatch`):** While your core protocol (Rule #3) requires deep dives into sources, note the following exceptions: For `exactMatch` questions, if you find a perfect answer directly within Google Books search result snippets (see Rule #6 below) or within the special 'book results' section on a redirected Google Search page (see Rule #5 below), you should stop searching and report that result immediately.

3.  **Domain Rotation Refinement:**
    * Your core protocol (Rule #10) requires using random Google domains. For this task, use these specific domains for the *first* search on each respective service:
        * Google Books: Start with `{books_domain}`
        * Google Scholar: Start with `{scholar_domain}`
        * Regular Google Search: Start with `{regular_domain}`
    * For any *subsequent* searches on the *same* service (e.g., a second Google Books search), or if the initial domain fails, revert to selecting a random domain from the list provided in your core protocol (Rule #10), ensuring it's different from the previous one used for that service or other services in this task. Maintain the 5-7 second wait between searches on different Google services/domains.

4.  **Search Query Formulation:**
    * **Google Books (First Search):** If it is *not* an `exactMatch` question, search for keywords directly from the query. If it *is* an `exactMatch` question, search for the exact wording *after* removing blanks (as per core protocol Rule #18).
    * **Google Scholar (Second Search):** Use precise keywords. For `exactMatch`, use the modified query (blanks removed).
    * **Regular Google (Third Search):** Use keywords, potentially adding terms like "quote", "excerpt", "full text", "study", "research", "pdf". For `exactMatch`, *always* use the modified query (blanks removed).

5.  **Google Books Redirect Handling Refinement:** (Refines core protocol Rule #11)
    * If redirected from Google Books to a standard Google Search page (`https://www.google.[tld]/search?...`), look specifically for the HTML `<div class="VNSPub">Found inside</div>`.
    * If found, extract the text within subsequent `<span>` elements, paying close attention to `<em>` tagged highlights (preserve `<em>` tags during initial extraction for analysis).
    * **`exactMatch` Stop Condition:** If this extracted text provides a direct and complete answer to an `exactMatch` query, record the text and source, and **STOP** searching immediately.

6.  **Google Books Snippet Handling Refinement:** (Refines core protocol Rule #11 & #19)
    * On the *initial* Google Books search results page, **carefully check the text snippets** displayed below each result *before* clicking into any book.
    * **`exactMatch` Stop Condition:** If a snippet perfectly answers an `exactMatch` query (comparing against the blank-removed version), record the snippet text and its source book details, and **STOP** searching immediately.

7.  **Google Books Preview Interaction:** (Refines core protocol Rule #11)
    * When exploring book previews, actively use the **search box** often found within the book preview interface (e.g., left side panel) to search for keywords *within that specific book*.
    * Also, click on relevant snippets shown in the main Google Books search results to navigate directly to that context within the preview page.

8.  **Regular Google Search Source Focus:** (Refines core protocol Rule #13)
    * When performing the regular Google search (Step 3), prioritize results from educational websites (.edu, .ac.uk, etc.), institutional repositories (archives, preprint servers like arXiv), government agencies (.gov), and reputable non-profit organizations (.org). Look for alternative versions of texts (e.g., author copies).

9.  **Detailed Documentation Requirements:** (Expands on core protocol Rules #6, #7, #8) For each source accessed, ensure you document:
    * Full citation details (authors, title, journal/book, year, volume, pages, DOI/ISBN).
    * Direct URL accessed.
    * Citation count & publication year (if readily available).
    * **EXACT QUOTES:** Verbatim text answering or directly related to the query.
    * **Full Abstract/Summary:** Copy the complete text when available.
    * **Methodology:** Note if described.
    * **Key Findings/Conclusions:** Summarize, include page numbers if possible.
    * **Downloaded PDF Filename:** If applicable.
    * Record "Authentication required" if access was blocked.

10. **Source Prioritization Criteria:** When evaluating found sources, prioritize them in this order:
    1.  Provides a direct, verifiable answer (especially for `exactMatch`).
    2.  Full text was accessible.
    3.  Content contains exact matches to key query phrases.
    4.  Source is highly cited or from a reputable publisher/journal.
    5.  Information is recently published (unless historical context is needed).

11. **Final Output Formatting:** Present your findings as a detailed research summary. Organize the information clearly by source, including all the details gathered in Rule #9 for each. For `exactMatch` queries, ensure the final answer uses the reconstructed format specified in your core protocol (Rule #20).

Remember to adhere to all other rules in your base **ScholarBot Academic Research Protocol**, including random delays (Rules #14, #15) and tab management (Rule #16).
"""
        
        try:
            # Run the task with BrowserAgentBehavior
            result = asyncio.run(self.browser._run_task(
                restricted_task, 
                max_steps=max_steps, 
                download_path=download_path
            ))
            for action in result.action_results():
                logging.info(f"Literature Searching Action: {action}")
                if action.is_done:
                    return action.extracted_content
            return "No literature found."
        except Exception as e:
            error_msg = f"Error searching literature: {str(e)}"
            logging.error(error_msg)
            return error_msg

class GeneralBrowserTool(Tool):
    name = "general_browser_task"
    description = "Run a general web search and return the results."
    inputs = {
        "query": {"type": "string", "description": "The search query"},
        "max_steps": {"type": "integer", "description": "Maximum number of steps the agent can take", "default": 38, "nullable": True},
        "download_path": {"type": "string", "description": "Path to download PDF files", "default": "general_downloads", "nullable": True}
    }
    output_type = "string"

    def __init__(self, api_key=None, download_path=None, use_browser=True):
        """
        Initialize the general browser tool.
        
        Args:
            api_key: OpenAI API key
            download_path: Path to download PDF files
            use_browser: Whether to use a browser agent (set to False to use a text-only agent)
        """
        super().__init__()
        self.download_path = download_path or "general_downloads"
        self.system_prompt = f"""
**WebSearchBot Research Protocol:**

1.  **Role:** You are operating as WebSearchBot, a web research assistant. Your primary task is to search the web to find and retrieve accurate, relevant information based on the user's query.

**Critical Operational Rules:**

2.  **Visit Actual Pages:** Always navigate to and read the content on the actual webpages found in search results; do not rely solely on information presented in search snippets.
3.  **Extract Directly:** Extract information directly from the webpage content itself to ensure accuracy and capture necessary details.
4.  **Navigate Deeply:** If required information is not on the landing page, navigate through multiple relevant pages within the same website to gather complete data.
5.  **Cite Sources:** Always provide the direct URL (link) for each source from which you extract information.

**Research Methodology:**

6.  **Initial Search:** Begin searches with broader queries related to the topic.
7.  **Refine Search:** Based on initial results, refine your search terms to focus on more specific aspects or keywords.
8.  **Verify Information:** Cross-reference information found on one source with information from other reputable sources to verify accuracy.
9.  **Prioritize Sources:** Give preference to information from authoritative sources, such as:
    * Educational institutions (.edu, .ac domains)
    * Government websites (.gov, .gov.[country code], etc.)
    * Reputable news organizations
    * Established research institutions or publications.
10. **Note Conflicts:** If you find conflicting information across sources, or if data seems uncertain, explicitly state this in your findings.
11. **Balanced View:** When researching topics with multiple perspectives or ongoing debates, aim to summarize the main viewpoints found in your sources.

**Browser Interaction & Timing:**

12. **Mandatory Random Delays:** Before *every* page operation (clicking, searching, opening/closing tabs, typing, scrolling), pause for a *random* duration between 3 and 5 seconds. The duration *must* vary each time; do not use the same delay consecutively.
13. **Post-Close Delay:** After closing a tab, wait another random duration (3-5 seconds, different from the previous delay) before performing the next action (like opening a new tab or searching).
14. **Google Domain Rotation:**
    * Maintain this list of Google domains: `www.google.com`, `www.google.ca`, `www.google.fr`, `www.google.co.uk`, `www.google.de`, `www.google.com.au`, `www.google.co.jp`, `www.google.co.in`, `www.google.com.br`, `www.google.ru`, `www.google.it`, `www.google.es`, `www.google.com.mx`, `www.google.co.kr`, `www.google.nl`, `www.google.pl`, `www.google.com.sg`, `www.google.co.za`, `www.google.com.tr`, `www.google.se`.
    * For *every* new Google search you initiate, randomly select one domain from this list to use as the base URL (e.g., use `https://www.[random-domain]/` instead of `https://www.google.com/`). Use a different domain for subsequent searches.
    * Implement a longer random delay, between 5 and 7 seconds, when switching between different Google services or initiating searches shortly after one another on potentially different domains.

**PDF Handling:**

15. **Download Useful PDFs:** If you find relevant PDF documents that are freely accessible, download them.
16. **Save Location:** Save all downloaded PDF files to the designated download directory: {{self.download_path}}.
17. **Avoid Restricted PDFs:** Do *not* attempt to download PDFs that are behind paywalls or require login credentials.
18. **Record Downloads:** After successfully downloading a PDF, record its filename and confirm its save location. Include this information in your memory or final report.

**Authentication Handling:**

19. **Avoid Barriers:** If you encounter any page requiring login, CAPTCHA verification, payment (paywall), or other forms of authentication, immediately cease interaction with that page or source.
20. **Do Not Bypass:** Do *not* attempt to bypass security measures, solve CAPTCHAs (unless explicitly supported by base agent capabilities), or enter credentials.
21. **Note and Proceed:** Record "Authentication required" for the source in your memory and proceed to find alternative, freely accessible sources.
"""
        self.browser = LiteratureSearchBrowser(api_key=api_key, system_prompt=self.system_prompt, download_path=download_path, use_browser=use_browser)

    def forward(self, query: str, max_steps: int = 38, download_path: str = None) -> str:
        """
        Run a general web search and return the results.
        
        Args:
            query: The search query
            max_steps: Maximum number of steps the agent can take
            download_path: Path to download PDF files
            
        Returns:
            String containing the search results
        """
        logging.info(f"Running general web search for query: {query}")
        actual_download_path = download_path or self.download_path
        
        restricted_task = f"""
**WebSearchBot Supplemental Instructions (Refining Base Protocol):**

Based on the specific query '{query}', apply these supplemental instructions that refine or add to your core **WebSearchBot Research Protocol**:

1.  **CRITICAL FIRST STEP: `robots.txt` Check:**
    * Before accessing *any* website domain for the first time (e.g., www.example.com, scholar.google.com), you MUST check its `robots.txt` file (e.g., `https://www.example.com/robots.txt`).
    * Strictly respect all `Disallow` directives. Do not access any disallowed paths or pages listed in the `robots.txt` file. If a required path is disallowed, seek information via alternative allowed paths or different sources.

2.  **`exactMatch` Question Handling (If Applicable):**
    * Identify if the query is an `exactMatch` question (contains blanks like "____" or "[BLANK]").
    * **CRITICAL Pre-Search Step:** If it *is* an `exactMatch` question, ALWAYS remove the blank placeholders *before* performing any search (Google Search, Scholar, Books).
        * Example: "The Battle of _____ was fought in 1815" → search for "The Battle of was fought in 1815".
    * **Final Answer Reconstruction:** When you find the text that completes an `exactMatch` query, formulate the final answer using the *original* question format (with blanks). Fill in the blank(s) with the found information and highlight or emphasize the filled-in part(s).
        * Example: Found "The Battle of Waterloo was fought in 1815". Final Answer: "The Battle of **Waterloo** was fought in 1815".

3.  **Refined Search Strategy Sequence & Conditions:**
    * **Step A: General Google Search:** Always start with this (using random domains as per core protocol Rule #14). Apply `exactMatch` pre-search modification if needed (Rule #2 above).
    * **Step B: Google Scholar (Conditional):** If the topic is academic or historical, *also* perform a search on Google Scholar (using a different random domain). Apply `exactMatch` pre-search modification if needed. Focus on accessing full text where possible and extracting precise quotes/information.
    * **Step C: Google Books (Conditional):** If the topic might benefit from book content, *also* perform a search on Google Books (using a different random domain). Apply `exactMatch` pre-search modification if needed. Follow the specific handling rules below (Rule #4).

4.  **Specific Google Books Handling Instructions:**
    * **Redirect Handling (`VNSPub` Section):** If a Google Books search redirects to a standard Google Search page (`https://www.google.[tld]/search?...`), look specifically for the HTML: `<div class="VNSPub">Found inside</div>`.
        * If found, extract text from subsequent `<span>` elements, noting `<em>` highlights (preserve `<em>` initially).
        * **`exactMatch` Stop Condition:** If this extracted text provides a direct, complete answer to an `exactMatch` query, record the text/source and **STOP** searching immediately.
    * **Snippet Handling (Results Page):** On the *initial* Google Books search results page, carefully check the text snippets *first*.
        * **`exactMatch` Stop Condition:** If a snippet perfectly answers an `exactMatch` query, record the snippet/source and **STOP** searching immediately.
    * **Preview Interaction:** Use book previews to find relevant sections. If redirected to standard Google Search *and* the `VNSPub` section isn't definitive, ensure the "Books" filter below the search bar is selected.
    * **Extraction:** Extract relevant information from accessible previews, noting page numbers when available.

5.  **Detailed Documentation Requirements:** (Expands on core protocol Rule #5) For each important source consulted, document:
    * Full title of the webpage or document.
    * Direct URL.
    * Author or Publisher/Website Name information.
    * Publication or Last Updated date (if available).
    * Key information found relevant to the query.
    * Exact quotes that directly address the query (use sparingly, prefer summarizing unless quotes are crucial).
    * Filename and confirmed save location (`{{self.download_path}}`) for any successfully downloaded PDFs.

6.  **Final Output Formatting:** (Expands on core protocol Rules #10, #11)
    * Present your findings as a comprehensive summary.
    * Where appropriate (especially if finding conflicting data per Rule #10), compare information across different sources.
    * Highlight the most relevant findings that directly answer the user's query.
    * Clearly list all cited sources (with URLs) and any downloaded files.

Remember to adhere to all other rules in your base **WebSearchBot Research Protocol**, including visiting actual pages, direct extraction, deep navigation, source prioritization, handling conflicts, random delays, PDF downloading/saving, and authentication avoidance.
"""
        
        try:
            # Run the task with BrowserAgentBehavior
            result = asyncio.run(self.browser._run_task(
                restricted_task, 
                max_steps=max_steps, 
                download_path=download_path
            ))
            for action in result.action_results():
                logging.info(f"General Browser Action: {action}")
                if action.is_done:
                    return action.extracted_content
            return "No Information Found."
        except Exception as e:
            error_msg = f"Error in general browser task: {str(e)}"
            logging.error(error_msg)
            return error_msg


class RelevantLiteratureFinderTool(Tool):
    name = "relevant_literature_finder"
    description = "Search for literature and return the most relevant sources for the query."
    inputs = {
        "query": {"type": "string", "description": "The research query or topic to search for"},
        "max_results": {"type": "integer", "description": "Maximum number of sources to return (default 3)", "default": 3, "nullable": True},
        "max_steps": {"type": "integer", "description": "Maximum number of steps the agent can take", "default": 38, "nullable": True},
        "download_path": {"type": "string", "description": "Path to download PDF files", "default": "relevant_literature_downloads", "nullable": True}
    }
    output_type = "string"
    
    def __init__(self, api_key=None, download_path=None, use_browser=True):
        """
        Initialize the relevant literature finder tool.
        
        Args:
            api_key: OpenAI API key
            download_path: Path to download PDF files
            use_browser: Whether to use a browser agent (set to False to use a text-only agent)
        """
        super().__init__()
        self.download_path = download_path or "relevant_literature_downloads"
        self.system_prompt = LiteratureSearchBrowser(api_key=api_key, download_path=download_path).system_prompt
        self.api_key = api_key
        self.use_browser = use_browser

    async def _search_and_filter_literature(self, query: str, max_results: int = 3, max_steps: int = 38, download_path: str = None) -> str:
        """
        Search for literature and then filter for the most relevant ones.
        
        Args:
            query: The search query for literature
            max_results: Maximum number of relevant results to return
            max_steps: Maximum number of steps the agent can take
            download_path: Path to download PDF files
            
        Returns:
            String containing the most relevant literature findings
        """
        logging.info(f"Searching and filtering literature for: {query}")
        actual_download_path = download_path or self.download_path
        
        try:
            # Create a search task
            search_task = f"""Search for high-impact, recent scholarly articles and relevant content about: {query}. 
            
            1. BEFORE accessing any website, ALWAYS check the site's robots.txt file by visiting domain.com/robots.txt (e.g., https://books.google.com/robots.txt)
            2. RESPECT all directives in the robots.txt file - do not access any disallowed paths or pages
            3. If a path is disallowed in robots.txt, do not attempt to access it and find an alternative path
            
            CRITICAL INSTRUCTION: You MUST click into each article to read the full text or abstract page. Do not just rely on the search results page.
            
            SPECIAL INSTRUCTIONS FOR EXACTMATCH QUESTIONS:
            - If this is an exactMatch question, you only need to find ONE most relevant literature source
            - If the source appears as a small image segment, carefully extract the text from the image and return it
            - After filling in the blanks, verify that the complete text can be found in Google Books
            - NEVER answer with "Unable to determine" for exactMatch questions - continue searching until you find a match
            
            FOLLOW THIS SEARCH STRATEGY IN ORDER:
            
            STEP 1: Begin with Google Books
            - IMPORTANT: Instead of using books.google.com, randomly select from these Google domains:
              books.google.com, books.google.ca, books.google.fr, books.google.co.uk, books.google.de, 
              books.google.com.au, books.google.co.jp, books.google.co.in, books.google.com.br, books.google.it,
              books.google.es, books.google.com.mx, books.google.co.kr, books.google.nl
            - Use a different domain for each search to avoid rate limiting
            - Wait 5-7 seconds between searches on different domains
            - If it is not an exact match question, search for keywords directly from the query. If it is an exact match question, search for the exact wording of the query.
            - For exactMatch questions, ALWAYS remove any blanks (like "____" or "[BLANK]") from the question before searching
            - Example: "The Battle of _____ was fought in 1815" → search for "The Battle of was fought in 1815"
            - This is CRITICAL because scholarly texts won't contain these blank placeholders
            - CRITICALLY IMPORTANT: If your search redirects to a regular Google page (URL starting with https://www.google.com/search?):
              * Look for a section labeled "Found inside" (matching results found in books)
              * IMPORTANT: When you see HTML with <div class="VNSPub">Found inside</div>, extract all HTML information that follows this div - this contains the matching results
              * The content will typically appear in a structure like: <div class="cmlJmd ETWPw"><div class="VNSPub">Found inside</div><span><span>... text content with <em>highlighted parts</em> ...</span></span></div>
              * Extract the text within the <span> elements that follow the VNSPub div
              * Pay special attention to text highlighted with <em> tags - these are the exact matches to your search query
              * For extraction purposes, preserve the <em> tags to identify matched content
              * For your final answer, you can remove the <em> tags while still emphasizing this matched content
              * Example: If you see "<em>In 1825</em> the Bourbon regime... In <em>Franche-Comté</em>...", extract this entire text and note that "In 1825" and "Franche-Comté" were highlighted matches
              * This section contains direct text snippets from books matching your search
              * READ THESE SNIPPETS CAREFULLY FIRST before anything else
              * For exactMatch questions, the answer is very likely to be found in these snippets
              * If you find a matching snippet, record the text and book source immediately
              * If a snippet perfectly matches the query for an exactMatch question, STOP SEARCHING and return that result
            - IMPORTANT: On the search results page, CAREFULLY CHECK THE SNIPPETS first before clicking into books:
              * Read each snippet on the search results page carefully
              * If a snippet perfectly matches the query (with blanks removed for exactMatch questions), YOU CAN STOP SEARCHING
              * Simply record that snippet along with its source book details and return it immediately
              * For exactMatch questions, once you've found an exact match in a snippet, no further searching is needed
            - Find books with preview access related to the topic
            - Use the "Search inside" or preview feature to locate relevant sections
            - Important: If search results show SNIPPETS that match the query, CLICK on them to see the full context
            - When viewing a book, use the SEARCH BOX on the left side panel to search for keywords within that book
            - If Google Books redirects your search to regular Google (URL starting with https://www.google.com/search?q=), this is acceptable - continue with the search
            - When redirected to regular Google, make sure "Books" is selected in the options below the search bar to filter results to book content
            - If the Google Books search is rejected or doesn't work, try searching on regular Google.com instead, then click on the "Books" filter option below the search bar
            - You don't need the entire book to be accessible - focus on available preview sections or snippets
            - Look for exact quotes or information that matches the query
            - Record exact page numbers whenever possible
            - Each time wait a RANDOM duration between 3-5 seconds for page operations (clicking links, opening books, performing searches), never use the same delay twice
            - After finishing with each book, CLOSE THE TAB before moving to the next source
            
            STEP 2: If insufficient results from Google Books, search on Google Scholar
            - IMPORTANT: Instead of using scholar.google.com, randomly select from these Google domains:
              scholar.google.com, scholar.google.ca, scholar.google.fr, scholar.google.co.uk, scholar.google.de, 
              scholar.google.com.au, scholar.google.co.jp, scholar.google.co.in, scholar.google.com.br, scholar.google.it,
              scholar.google.es, scholar.google.com.mx, scholar.google.co.kr, scholar.google.nl
            - Use a different domain than you used for Google Books
            - Wait 5-7 seconds between searches on different domains
            - Search for relevant articles using keywords from the query
            - For exactMatch questions, ALWAYS remember to search without any blanks
            - Example: "The Battle of _____ was fought in 1815" → search for "The Battle of was fought in 1815" 
            - This is CRITICAL because scholarly texts won't contain these blank placeholders
            - Always try to access the full text when possible
            - Extract precise quotes and information
            - Each time wait a RANDOM duration between 3-5 seconds for page operations (opening articles, clicking links), never use the same delay twice
            - After finishing with each article, CLOSE THE TAB before moving to the next source
            
            STEP 3: If full text is inaccessible from both Google Books and Google Scholar:
            - Try a regular Google Search using a random Google domain from this list:
              www.google.com, www.google.ca, www.google.fr, www.google.co.uk, www.google.de, 
              www.google.com.au, www.google.co.jp, www.google.co.in, www.google.com.br, www.google.it,
              www.google.es, www.google.com.mx, www.google.co.kr, www.google.nl
            - Use a different domain than you used in previous searches
            - Wait 5-7 seconds between searches on different domains
            - Search for the same concepts plus terms like "quote", "excerpt", or "full text"
            - For exactMatch questions, ALWAYS continue to search without any blanks
            - Example: "The Battle of _____ was fought in 1815" → search for "The Battle of was fought in 1815"
            - This is ESSENTIAL because web texts won't contain these blank placeholders
            - Look for educational websites, repositories, or other scholarly sources
            - Check if there are alternative versions of the text on different websites
            - Each time wait a RANDOM duration between 3-5 seconds for page operations, never use the same delay twice
            - After finishing with each source, CLOSE THE TAB before moving to the next source
            
            PDF DOWNLOAD INSTRUCTIONS:
            - When you find freely accessible PDFs, download them to this specific directory: {actual_download_path}
            - Do NOT attempt to download files behind paywalls or requiring login
            - After downloading, note the filename and specific location of each downloaded PDF
            - Include the PDF filename in your report for each downloaded article
            
            IMPORTANT HANDLING INSTRUCTIONS:
            - If you encounter any login walls, CAPTCHA verification, paywalls, or other authentication requirements, EXIT that page immediately and follow the alternative search steps above
            - Do NOT attempt to bypass any security measures or authentication systems
            - Simply note "Authentication required" for that source and move on to other accessible sources or alternative search methods
            - Focus your time on resources that are freely accessible without login requirements
            
            Prioritize:
            1. Sources containing EXACT text matches to the query (highest priority)
            2. Sources where you can access full text, not just abstracts
            3. Most relevant content to the query topic
            4. Highly cited papers or books from reputable sources
            
            CRITICAL FOR EXACTMATCH QUESTIONS:
            - For exactMatch type questions, finding the precise original wording is ESSENTIAL
            - The complete and precise answer exists verbatim in the literature and must be found
            - ALWAYS remove any blanks (like "____", "___", or "[BLANK]") from the question before searching
            - This is CRITICAL because texts in scholarly sources won't contain these blank placeholders
            - Example: "The Battle of _____ was fought in 1815" → search for "The Battle of was fought in 1815"
            - When you find a match, try to fill in the blank and then verify the complete sentence with the blank filled in Google Books
            - Only return ONE most relevant source for exactMatch questions
            - If the source is a small image, carefully extract the text from the image and include it in your response
            - If you find an exact match in a Google Books snippet or in the "Found inside" section, stop searching immediately and return that snippet with source details
            - IMPORTANT: To find the "Found inside" section, look for HTML with <div class="VNSPub">Found inside</div> and extract all information that follows this div
            - The content will typically appear in a structure like: <div class="cmlJmd ETWPw"><div class="VNSPub">Found inside</div><span><span>... text content with <em>highlighted parts</em> ...</span></span></div>
            - Extract the text within the <span> elements that follow the VNSPub div
            - The <em> tags highlight the exact matches to your search query - these are the most important parts to focus on
            - When presenting your findings, you can remove the <em> tags but be sure to emphasize those matched terms
            - CRITICALLY IMPORTANT FOR FINAL ANSWER: When extracting information and presenting results, ADD BACK THE BLANKS and fill them in with the correct information
            - Example: If you searched for "The Battle of was fought in 1815" and found "The Battle of Waterloo was fought in 1815", your final answer should be "The Battle of _Waterloo_ was fought in 1815"
            - Always highlight or emphasize the filled-in information that was originally a blank in the question
            - NEVER answer with "Unable to determine" - continue searching until you find a valid match
            
            For each important source, document:
            - The full title and URL
            - Author or publisher information
            - Publication date (if available)
            - Key information found
            - Exact quotes that directly address the query
            - Any downloaded files with their locations
            
            Format your response as a comprehensive summary of the information found, comparing across sources and highlighting the most relevant findings. Include all sources and downloaded files.
            """
            
            # Run the search task using the browser
            config = BrowserConfig(headless=True)
            browser = Browser(config) if self.use_browser else None
            llm = ChatOpenAI(model="o4-mini", temperature=0, api_key=self.api_key)
            if self.use_browser:
                agent = Agent(
                    task=search_task, 
                    llm=llm, 
                    browser=browser, 
                    generate_gif=False,
                    extend_system_message=self.system_prompt
                )
            else:
                agent = Agent(
                    task=search_task, 
                    llm=llm, 
                    generate_gif=False,
                    extend_system_message=self.system_prompt
                )
            result = await agent.run(max_steps=max_steps)
            await browser.close()  # 关闭浏览器
            
            # Use GPT to filter and rank the most relevant results
            filter_llm = ChatOpenAI(model="o4-mini", temperature=0, api_key=self.api_key)
            
            prompt = f"""
            As an expert research librarian specializing in history, review the following search results and identify the {max_results} most relevant sources for answering this query: "{query}"
            
            Remember:
            1. BEFORE accessing any website, ALWAYS check the site's robots.txt file by visiting domain.com/robots.txt (e.g., https://books.google.com/robots.txt)
            2. RESPECT all directives in the robots.txt file - do not access any disallowed paths or pages
            3. If a path is disallowed in robots.txt, do not attempt to access it and find an alternative path
            
            For Google Books specifically, ONLY access these allowed paths:
            - Allow: /books?*q=related:*
            - Allow: /books?*q=editions:*
            - Allow: /books?*q=subject:*
            - Allow: /books/about
            - Allow: /booksrightsholders
            - Allow: /books?*zoom=1*
            - Allow: /books?*zoom=5*
            - Allow: /books/content?*zoom=1*
            - Allow: /books/content?*zoom=5*
            
            IMPORTANT: When searching Google Books, ALWAYS prefix your query with "related:" (e.g., "related:your search term") to comply with robots.txt directives and ensure your access is allowed.
            
            Search Results:
            {result.final_result}
            
            CRITICAL FOR EXACT MATCH QUESTIONS:
            If this is an exactMatch type question, you MUST find and preserve the EXACT original wording from academic sources. Focus on sources where the exact text matching the query was found.
            
            For each selected source, provide:
            1. Full citation with authors, title, journal, year, DOI
            2. Direct URL to the article
            3. Relevance score (1-10)
            4. EXACT QUOTES from the article that match or relate to the query (preserve the exact wording)
            5. Page number or location where the quote appears (if available)
            6. Full abstract copied directly from the article
            7. PDF filename if downloaded
            8. Explanation of why this source directly answers the query
            
            CRITICALLY IMPORTANT: 
            - You MUST preserve and include ANY text found that matches parts of the query word-for-word
            - Even partial matches to the query text are extremely valuable
            - The exact wording is essential for questions requiring precise answers
            - Include page numbers whenever possible so the exact text can be cited properly
            - If any PDFs were downloaded during the search, highlight these as they contain the full text
            
            Return your analysis in the following JSON format:
            {
              "selected_sources": [
                {
                  "citation": "Full citation of the source",
                  "url": "Direct URL to the article",
                  "relevance_score": number between 1-10,
                  "abstract": "Full abstract from the article",
                  "exact_quotes": [
                    {
                      "text": "Exact quote from the article that matches or relates to the query",
                      "page_number": "Page number or location (if available)",
                      "matches_query": true/false (whether this exactly matches part of the query)
                    }
                  ],
                  "relevance_explanation": "Detailed explanation of why this source directly answers the query"
                },
                ...
              ],
              "search_summary": "Brief summary of the search results and why these particular sources were selected",
              "exact_match_found": true/false (whether any exact matches to the query were found)
            }
            
            Only return the JSON structure, no additional text.
            """
            
            response = filter_llm.invoke(prompt)
            json_strings = response.content
            
            # Try to extract JSON if wrapped in markdown code block
            json_pattern = r'```(?:json)?\n(.*?)\n```'
            match = re.search(json_pattern, json_strings, re.DOTALL)
            if match:
                parsed_results = match.group(1)
            else:
                # Otherwise use the entire response
                parsed_results = json_strings
                
            try:
                result_obj = json.loads(parsed_results)
                selected_sources = result_obj.get("selected_sources", [])
                exact_match_found = result_obj.get("exact_match_found", False)
                
                # Format the results in a more readable way
                filtered_results = f"## Most Relevant Literature for: {query}\n\n"
                
                if exact_match_found:
                    filtered_results += f"### EXACT MATCH FOUND! The query text was located in the literature.\n\n"
                
                for i, source in enumerate(selected_sources, 1):
                    filtered_results += f"### Source {i}: Relevance Score {source.get('relevance_score')}/10\n"
                    filtered_results += f"**Citation**: {source.get('citation')}\n"
                    filtered_results += f"**URL**: {source.get('url', 'Not provided')}\n\n"
                    
                    # Add abstract
                    if source.get('abstract'):
                        filtered_results += f"**Abstract**: {source.get('abstract')}\n\n"
                    
                    # Add exact quotes section with formatting to highlight exact matches
                    filtered_results += "**Key Quotes**:\n"
                    for quote in source.get('exact_quotes', []):
                        quote_text = quote.get('text', '')
                        page_info = f" (Page: {quote.get('page_number')})" if quote.get('page_number') else ""
                        
                        if quote.get('matches_query'):
                            filtered_results += f"- **EXACT MATCH!** \"{quote_text}\"{page_info}\n"
                        else:
                            filtered_results += f"- \"{quote_text}\"{page_info}\n"
                    
                    filtered_results += f"\n**Why Relevant**: {source.get('relevance_explanation')}\n\n"
                    filtered_results += "----------------------\n\n"
                
                filtered_results += f"## Search Summary\n{result_obj.get('search_summary', 'No summary provided.')}"
                
                return filtered_results
                
            except json.JSONDecodeError:
                # If JSON parsing fails, return the raw search results
                logging.warning("Could not parse JSON from LLM response, returning raw search results")
                return f"Could not process search results in the expected format. Raw search results:\n\n{result.final_result}"
                
        except Exception as e:
            error_msg = f"Error filtering relevant literature: {str(e)}"
            logging.error(error_msg)
            return f"Literature search error: {str(e)}"

    def forward(self, query: str, max_results: int = 3, max_steps: int = 38, download_path: str = None) -> str:
        """
        Search for literature and return the most relevant sources for the query.
        
        Args:
            query: The research query or topic to search for
            max_results: Maximum number of sources to return (default 3)
            max_steps: Maximum number of steps the agent can take
            download_path: Path to download PDF files
            
        Returns:
            String containing the most relevant literature with explanations
        """
        return asyncio.run(self._search_and_filter_literature(query, max_results, max_steps, download_path))

class BookMatchExtractorTool(Tool):
    name = "book_match_extractor"
    description = "Extract book match snippets from Google Books search results for a query."
    inputs = {
        "query": {"type": "string", "description": "The query to search for in Google Books"},
        "max_steps": {"type": "integer", "description": "Maximum number of steps the agent can take", "default": 38, "nullable": True}
    }
    output_type = "string"

    def __init__(self, api_key=None, download_path=None, use_browser=True):
        """
        Initialize the book match extractor tool.
        
        Args:
            api_key: OpenAI API key
            download_path: Path to download PDF files
            use_browser: Whether to use a browser agent (set to False to use a text-only agent)
        """
        super().__init__()
        self.download_path = download_path or "book_match_downloads"
        # Use the system prompt from LiteratureSearchBrowser
        self.browser = LiteratureSearchBrowser(api_key=api_key, download_path=download_path, use_browser=use_browser)
        self.api_key = api_key

    async def _extract_book_matches_async(self, query: str, max_steps: int = 38) -> str:
        """
        Extract book match snippets from Google Books search results for a query.
        
        Args:
            query: The query to search for in Google Books
            max_steps: Maximum number of steps the agent can take
            
        Returns:
            Formatted string containing the extracted book match snippets
        """
        results = await self.browser.extract_book_matches(query, max_steps)
        
        search_url = results.get("search_url", None)
        book_matches = results.get("book_matches", None)
        print("results", results)
        
        if not book_matches:
            # If the browser method didn't get matches, try direct parsing
            if search_url:
                book_matches = await self.browser.parse_google_books_url(search_url)
            
        if not book_matches:
            return f"No book match snippets found for query: {query}\nSearch URL: {search_url}"
        
        # Format the results
        results = f"## Book Match Snippets for: {query}\n\n"
        results += f"Search URL: {search_url}\n\n"
        
        for i, match in enumerate(book_matches, 1):
            results += f"### Match {i}:\n\n"
            
            if 'book_title' in match and match['book_title']:
                results += f"**Book**: {match['book_title']}\n"
                
            if 'book_link' in match and match['book_link']:
                results += f"**Link**: {match['book_link']}\n\n"
            
            if 'snippet_html' in match and match['snippet_html']:
                # For HTML snippet, preserve the formatting but clean up for readability
                html_snippet = match['snippet_html']
                # Replace <em> tags with markdown bold for highlighting
                html_snippet = html_snippet.replace('<em>', '**').replace('</em>', '**')
                # Remove other HTML tags
                html_snippet = re.sub(r'<[^>]*>', '', html_snippet)
                results += f"**Snippet (with highlights)**:\n{html_snippet}\n\n"
            
            if 'snippet_text' in match and match['snippet_text']:
                results += f"**Plain Text Snippet**:\n{match['snippet_text']}\n\n"
            
            if 'highlights' in match and match['highlights']:
                results += f"**Highlighted Terms**: {', '.join(match['highlights'])}\n\n"
            
            results += "---\n\n"
        
        return results

    def forward(self, query: str, max_steps: int = 38) -> str:
        """
        Extract book match snippets from Google Books search results for a query.
        
        Args:
            query: The query to search for in Google Books
            max_steps: Maximum number of steps the agent can take
            
        Returns:
            Formatted string containing the extracted book match snippets
        """
        logging.info(f"Extracting book matches for query: {query}")
        return asyncio.run(self._extract_book_matches_async(query, max_steps))

class DirectGoogleBooksCrawlerTool(Tool):
    name = "direct_google_books_crawler"
    description = "Directly extract book match snippets from a Google Books search URL."
    inputs = {
        "url": {"type": "string", "description": "The Google Books search URL to extract snippets from"}
    }
    output_type = "string"

    def __init__(self, api_key=None, download_path=None, use_browser=True):
        """
        Initialize the direct Google Books crawler tool.
        
        Args:
            api_key: OpenAI API key
            download_path: Path to download PDF files
            use_browser: Whether to use a browser agent (set to False to use a text-only agent)
        """
        super().__init__()
        self.download_path = download_path or "book_match_downloads"
        # Use the system prompt from LiteratureSearchBrowser
        self.browser = LiteratureSearchBrowser(api_key=api_key, download_path=download_path, use_browser=use_browser)
        self.api_key = api_key

    async def _parse_google_books_url_async(self, url: str) -> str:
        """
        Parse a Google Books search URL to extract book match snippets.
        
        Args:
            url: The Google Books search URL
            
        Returns:
            Formatted string containing the extracted book match snippets
        """
        book_matches = await self.browser.parse_google_books_url(url)
        
        if not book_matches:
            return f"No book match snippets found at URL: {url}"
        
        # Format the results
        results = f"## Book Match Snippets from URL\n\n"
        results += f"Source URL: {url}\n\n"
        
        for i, match in enumerate(book_matches, 1):
            results += f"### Match {i}:\n\n"
            
            if 'book_title' in match and match['book_title']:
                results += f"**Book**: {match['book_title']}\n"
                
            if 'book_link' in match and match['book_link']:
                results += f"**Link**: {match['book_link']}\n\n"
            
            if 'snippet_html' in match and match['snippet_html']:
                # For HTML snippet, preserve the formatting but clean up for readability
                html_snippet = match['snippet_html']
                # Replace <em> tags with markdown bold for highlighting
                html_snippet = html_snippet.replace('<em>', '**').replace('</em>', '**')
                # Remove other HTML tags
                html_snippet = re.sub(r'<[^>]*>', '', html_snippet)
                results += f"**Snippet (with highlights)**:\n{html_snippet}\n\n"
            
            if 'snippet_text' in match and match['snippet_text']:
                results += f"**Plain Text Snippet**:\n{match['snippet_text']}\n\n"
            
            if 'highlights' in match and match['highlights']:
                results += f"**Highlighted Terms**: {', '.join(match['highlights'])}\n\n"
            
            results += "---\n\n"
        
        return results

    def forward(self, url: str) -> str:
        """
        Parse a Google Books search URL to extract book match snippets.
        
        Args:
            url: The Google Books search URL
            
        Returns:
            Formatted string containing the extracted book match snippets
        """
        logging.info(f"Parsing Google Books URL: {url}")
        return asyncio.run(self._parse_google_books_url_async(url))

class SpringerLlamaProcessor:
    """Springer Nature API client with LlamaParse integration."""
    
    BASE_URL = "https://api.springernature.com/openaccess"
    
    def __init__(self, springer_api_key=os.getenv("SPRINGER_API_KEY"),
                 llama_api_key=os.getenv("LLAMA_API_KEY")):
        """Initialize with API keys."""
        self.springer_api_key = springer_api_key
        if not self.springer_api_key:
            raise ValueError("Springer API key is missing")
        
        # Initialize LlamaParse with markdown output
        self.parser = LlamaParse(
            api_key=llama_api_key,
            result_type="markdown"  # "markdown" and "text" are available
        )
        
        # Set up file extractor for SimpleDirectoryReader
        self.file_extractor = {".pdf": self.parser}
    
    @sleep_and_retry
    @limits(calls=40, period=1)
    def search(self, query, max_results=40, full_text=False):
        """Search for publications on Springer Nature."""
        params = {
            'q': query,
            'p': min(max_results, 100),
            'api_key': self.springer_api_key,
            'content_type': 'article',
            'include_tdm': 'true',
            'include_fulltext': 'true' if full_text else 'false'
        }
        
        search_url = f"{self.BASE_URL}/json"
        print(f"Searching at: {search_url}")
        
        try:
            response = requests.get(search_url, params=params)
            response.raise_for_status()
            return self._parse_response(response.json())
        except requests.RequestException as e:
            raise RuntimeError(f"Request failed: {str(e)}")
    
    def _parse_response(self, response_data):
        """Parse Springer API response into standardized format."""
        results = []
        
        for item in response_data.get('records', []):
            publication = {
                'title': item.get('title', ''),
                'authors': [creator.get('creator', '') for creator in item.get('creators', [])],
                'publisher': 'Springer Nature',
                'published_date': item.get('publicationDate', ''),
                'description': item.get('abstract', ''),
                'doi': item.get('doi', ''),
                'url': item.get('url', [{}])[0].get('value', ''),
                'language': item.get('language', 'en'),
                'source': 'Springer Nature',
                'id': item.get('identifier', '')
            }
            results.append(publication)
            
        return {
            'total_items': response_data.get('total', 0),
            'results': results
        }
    
    def download_and_parse(self, doi_or_url, output_dir="downloads"):
        """
        Download a PDF from Springer and process it with LlamaParse
        
        Args:
            doi_or_url (str): The DOI or URL of the paper
            output_dir (str): Directory to save the PDF
        
        Returns:
            tuple: (pdf_path, parsed_documents)
        """
        try:
            print(f"\nProcessing input: {doi_or_url}")
            
            # Extract DOI if URL is provided
            if "doi:" in doi_or_url or "10." in doi_or_url:
                doi_match = re.search(r'(10\.\d{4}/.+?)(?:$|\&|\?)', doi_or_url)
                if doi_match:
                    doi = doi_match.group(1)
                    print(f"Extracted DOI: {doi}")
                else:
                    raise ValueError("Could not extract DOI from URL")
            else:
                doi = doi_or_url
                
            # Construct direct PDF URL
            pdf_url = f"https://link.springer.com/content/pdf/{doi}.pdf"
            print(f"Attempting download from: {pdf_url}")
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Create filename from DOI
            filename = f"{doi.replace('/', '_')}.pdf"
            output_path = os.path.join(output_dir, filename)
            
            # Send GET request with headers
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            print("Downloading PDF...")
            response = requests.get(pdf_url, headers=headers, stream=True)
            response.raise_for_status()
            
            # Download the file
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        
            print(f"Successfully downloaded to: {output_path}")
            
            # Process with LlamaParse using SimpleDirectoryReader
            print("\nProcessing with LlamaParse...")
            documents = SimpleDirectoryReader(
                input_files=[output_path],
                file_extractor=self.file_extractor
            ).load_data()
            print("LlamaParse processing complete!")
            
            return output_path, documents
            
        except requests.RequestException as e:
            print(f"Error downloading PDF: {e}")
            return None, None
        except Exception as e:
            print(f"Error processing with LlamaParse: {e}")
            return output_path, None
    
    def parse_input(self, user_input):
        """Extract structured information from user input."""
        sections = {
            "Main Concepts": [],
            "Related Disciplines": [],
            "Time and Place": [],
            "Core Themes": []
        }
        
        for section in sections:
            match = re.search(f"{section}: (.+?)(?:\n|$)", user_input)
            if match:
                sections[section] = [term.strip() for term in match.group(1).split(',')]
        print(sections)
        return sections
    
    def integrated_search(self, user_input, max_results=3, full_text=False):
        """Perform a structured search based on user input."""
        structured_data = self.parse_input(user_input)
        
        main_concepts = structured_data["Main Concepts"]
        related_disciplines = structured_data["Related Disciplines"]
        time_place = structured_data["Time and Place"]
        core_themes = structured_data["Core Themes"]
        
        queries = [
            ("single", main_concepts[:1]),
            ("exact_phrase", main_concepts[:1]),
            ("boolean_and", main_concepts + core_themes + time_place),
            ("boolean_or", main_concepts + core_themes),
            ("complex", [main_concepts[0]] + time_place)
        ]
        
        all_results = []
        for query_type, keywords in queries:
            if keywords:
                query = " AND ".join(f'"{kw}"' for kw in keywords)
                print(f"Running {query_type} search with: {query}")
                results = self.search(query, max_results)
                all_results.extend(results['results'])
        
        return all_results

class SpringerSearchTool(Tool):
    name = "springer_search"
    description = "Search for academic papers on Springer Nature's open access platform."
    inputs = {
        "query": {"type": "string", "description": "The research query to search for"},
        "max_results": {"type": "integer", "description": "Maximum number of results to return (default 10)", "default": 10, "nullable": True},
        "full_text": {"type": "boolean", "description": "Whether to include full text in search results", "default": False, "nullable": True}
    }
    output_type = "string"

    def __init__(self, springer_api_key=None, llama_api_key=None, download_path=None):
        """
        Initialize the Springer search tool.
        
        Args:
            springer_api_key: Springer Nature API key
            llama_api_key: LlamaParse API key
            download_path: Path to download PDF files
        """
        super().__init__()
        self.download_path = download_path or "springer_downloads"
        self.processor = SpringerLlamaProcessor(
            springer_api_key=springer_api_key,
            llama_api_key=llama_api_key
        )

    def forward(self, query: str, max_results: int = 10, full_text: bool = False) -> str:
        """
        Search for academic papers on Springer Nature.
        
        Args:
            query: The research query to search for
            max_results: Maximum number of results to return (default 10)
            full_text: Whether to include full text in search results
            
        Returns:
            String containing the search results formatted as JSON
        """
        logging.info(f"Searching Springer Nature for: {query}")
        try:
            results = self.processor.search(query, max_results, full_text)
            
            # Format the results as a readable string
            output = f"## Springer Nature Search Results for: {query}\n\n"
            output += f"Total results found: {results['total_items']}\n\n"
            
            if not results['results']:
                return output + "No results found."
            
            for i, result in enumerate(results['results'], 1):
                output += f"### {i}. {result['title']}\n"
                output += f"**Authors**: {', '.join(result['authors'])}\n"
                output += f"**Published**: {result['published_date']}\n"
                output += f"**DOI**: {result['doi']}\n"
                output += f"**URL**: {result['url']}\n\n"
                
                if result['description']:
                    output += f"**Abstract**:\n{result['description']}\n\n"
                
                output += "---\n\n"
            
            return output
        except Exception as e:
            return f"Error searching Springer Nature: {str(e)}"

class SpringerStructuredSearchTool(Tool):
    name = "springer_structured_search"
    description = "Perform a structured search on Springer Nature using categorized research concepts."
    inputs = {
        "user_input": {"type": "string", "description": "Structured research query with Main Concepts, Related Disciplines, Time and Place, and Core Themes"},
        "max_results": {"type": "integer", "description": "Maximum number of results per query type (default 3)", "default": 3, "nullable": True},
        "full_text": {"type": "boolean", "description": "Whether to include full text in search results", "default": False, "nullable": True}
    }
    output_type = "string"

    def __init__(self, springer_api_key=None, llama_api_key=None, download_path=None):
        """
        Initialize the Springer structured search tool.
        
        Args:
            springer_api_key: Springer Nature API key
            llama_api_key: LlamaParse API key
            download_path: Path to download PDF files
        """
        super().__init__()
        self.download_path = download_path or "springer_downloads"
        self.processor = SpringerLlamaProcessor(
            springer_api_key=springer_api_key,
            llama_api_key=llama_api_key
        )

    def forward(self, user_input: str, max_results: int = 3, full_text: bool = False) -> str:
        """
        Perform a structured search on Springer Nature.
        
        Args:
            user_input: Structured research query
            max_results: Maximum number of results per query type
            full_text: Whether to include full text in search results
            
        Returns:
            String containing the search results
        """
        logging.info(f"Performing structured search on Springer Nature")
        try:
            results = self.processor.integrated_search(user_input, max_results, full_text)
            
            # Remove duplicates based on DOI
            unique_results = []
            seen_dois = set()
            for result in results:
                if result['doi'] not in seen_dois:
                    unique_results.append(result)
                    seen_dois.add(result['doi'])
            
            # Format the results as a readable string
            output = f"## Springer Nature Structured Search Results\n\n"
            output += f"Query: {user_input}\n\n"
            output += f"Total unique results found: {len(unique_results)}\n\n"
            
            if not unique_results:
                return output + "No results found."
            
            for i, result in enumerate(unique_results, 1):
                output += f"### {i}. {result['title']}\n"
                output += f"**Authors**: {', '.join(result['authors'])}\n"
                output += f"**Published**: {result['published_date']}\n"
                output += f"**DOI**: {result['doi']}\n"
                output += f"**URL**: {result['url']}\n\n"
                
                if result['description']:
                    output += f"**Abstract**:\n{result['description']}\n\n"
                
                output += "---\n\n"
            
            return output
        except Exception as e:
            return f"Error performing structured search on Springer Nature: {str(e)}"

class SpringerDownloadAndParseTool(Tool):
    name = "springer_download_parse"
    description = "Download and parse a PDF from Springer Nature using LlamaParse."
    inputs = {
        "doi_or_url": {"type": "string", "description": "The DOI or URL of the paper to download and parse"},
        "output_dir": {"type": "string", "description": "Directory to save the PDF", "default": "springer_downloads", "nullable": True}
    }
    output_type = "string"

    def __init__(self, springer_api_key=None, llama_api_key=None, download_path=None):
        """
        Initialize the Springer download and parse tool.
        
        Args:
            springer_api_key: Springer Nature API key
            llama_api_key: LlamaParse API key
            download_path: Path to download PDF files
        """
        super().__init__()
        self.download_path = download_path or "springer_downloads"
        self.processor = SpringerLlamaProcessor(
            springer_api_key=springer_api_key,
            llama_api_key=llama_api_key
        )

    def forward(self, doi_or_url: str, output_dir: str = None) -> str:
        """
        Download and parse a PDF from Springer Nature.
        
        Args:
            doi_or_url: The DOI or URL of the paper to download and parse
            output_dir: Directory to save the PDF
            
        Returns:
            String containing the parsing results or error message
        """
        logging.info(f"Downloading and parsing paper: {doi_or_url}")
        try:
            actual_output_dir = output_dir or self.download_path
            pdf_path, documents = self.processor.download_and_parse(doi_or_url, actual_output_dir)
            
            if not pdf_path:
                return f"Error: Could not download PDF for {doi_or_url}"
            
            if not documents:
                return f"PDF downloaded to {pdf_path}, but parsing failed."
            
            # Format the parsed results
            output = f"## Parsed Paper: {doi_or_url}\n\n"
            output += f"PDF downloaded to: {pdf_path}\n\n"
            output += f"### Extracted Content:\n\n"
            
            for i, doc in enumerate(documents, 1):
                output += f"#### Document {i}:\n\n"
                
                # Truncate content if it's too long for display
                content = doc.text
                if len(content) > 3000:
                    content = content[:3000] + "...\n\n[Content truncated for display]"
                
                output += content + "\n\n"
            
            return output
        except Exception as e:
            return f"Error downloading and parsing paper: {str(e)}"

# Update the create_literature_tools function to include the new tools
def create_literature_tools(api_key=None, download_path=None, springer_api_key=None, llama_api_key=None):
    """
    Create a set of literature tools that use the provided API keys.
    
    Args:
        api_key: The OpenAI API key to use for browser tools
        download_path: Path to download PDF files
        springer_api_key: Springer Nature API key
        llama_api_key: LlamaParse API key
        
    Returns:
        Dictionary containing tool instances
    """
    return {
        "literature_searching_task": LiteratureSearchingTool(api_key=api_key, download_path=download_path),
        "general_browser_task": GeneralBrowserTool(api_key=api_key, download_path=download_path),
        "relevant_literature_finder": RelevantLiteratureFinderTool(api_key=api_key, download_path=download_path),
        "book_match_extractor": BookMatchExtractorTool(api_key=api_key, download_path=download_path),
        "direct_google_books_crawler": DirectGoogleBooksCrawlerTool(api_key=api_key, download_path=download_path),
        "springer_search": SpringerSearchTool(springer_api_key=springer_api_key, llama_api_key=llama_api_key, download_path=download_path),
        "springer_structured_search": SpringerStructuredSearchTool(springer_api_key=springer_api_key, llama_api_key=llama_api_key, download_path=download_path),
        "springer_download_parse": SpringerDownloadAndParseTool(springer_api_key=springer_api_key, llama_api_key=llama_api_key, download_path=download_path)
    }