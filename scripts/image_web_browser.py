# Shamelessly stolen from Microsoft Autogen team: thanks to them for this great resource!
# https://github.com/microsoft/autogen/blob/gaia_multiagent_v01_march_1st/autogen/browser_utils.py
import mimetypes
import os
import pathlib
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import unquote, urljoin, urlparse

import pathvalidate
import requests
from serpapi import GoogleSearch

from smolagents import Tool

from .cookies import COOKIES
from .mdconvert import FileConversionException, MarkdownConverter, UnsupportedFormatException


class SimpleImageBrowser:
    """(In preview) An extremely simple image-based web browser comparable to Lynx. Suitable for Agentic use."""

    def __init__(
        self,
        start_page: Optional[str] = None,
        viewport_size: Optional[int] = 1024 * 8,
        downloads_folder: Optional[Union[str, None]] = None,
        serpapi_key: Optional[Union[str, None]] = None,
        request_kwargs: Optional[Union[Dict[str, Any], None]] = None,
    ):
        self.start_page: str = start_page if start_page else "about:blank"
        self.viewport_size = viewport_size  # Applies only to the standard uri types
        self.downloads_folder = downloads_folder
        self.history: List[Tuple[str, float]] = list()
        self.page_title: Optional[str] = None
        self.viewport_current_page = 0
        self.viewport_pages: List[Tuple[int, int]] = list()
        self.set_address(self.start_page)
        self.serpapi_key = serpapi_key
        self.request_kwargs = request_kwargs
        self.request_kwargs["cookies"] = COOKIES
        self._mdconvert = MarkdownConverter()
        self._page_content: str = ""
        self._page_content_html: str = ""

        self._find_on_page_query: Union[str, None] = None
        self._find_on_page_last_result: Union[int, None] = None  # Location of the last result

    @property
    def address(self) -> str:
        """Return the address of the current page."""
        return self.history[-1][0]

    def set_address(self, uri_or_path: str, filter_year: Optional[int] = None) -> None:
        # TODO: Handle anchors
        self.history.append((uri_or_path, time.time()))

        # Handle special URIs
        if uri_or_path == "about:blank":
            self._set_page_content("")
        elif uri_or_path.startswith("google:"):
            self._serpapi_search(uri_or_path[len("google:") :].strip(), filter_year=filter_year)
        else:
            if (
                not uri_or_path.startswith("http:")
                and not uri_or_path.startswith("https:")
                and not uri_or_path.startswith("file:")
            ):
                if len(self.history) > 1:
                    prior_address = self.history[-2][0]
                    uri_or_path = urljoin(prior_address, uri_or_path)
                    # Update the address with the fully-qualified path
                    self.history[-1] = (uri_or_path, self.history[-1][1])
            self._fetch_page(uri_or_path)

        self.viewport_current_page = 0
        self.find_on_page_query = None
        self.find_on_page_viewport = None

    @property
    def viewport(self) -> str:
        """Return the content of the current viewport."""
        bounds = self.viewport_pages[self.viewport_current_page]
        return self.page_content[bounds[0] : bounds[1]]

    @property
    def page_content(self) -> str:
        """Return the full contents of the current page."""
        return self._page_content

    def _set_page_content(self, content: str) -> None:
        """Sets the text content of the current page."""
        self._page_content = content
        self._split_pages()
        if self.viewport_current_page >= len(self.viewport_pages):
            self.viewport_current_page = len(self.viewport_pages) - 1

    def page_down(self) -> None:
        self.viewport_current_page = min(self.viewport_current_page + 1, len(self.viewport_pages) - 1)

    def page_up(self) -> None:
        self.viewport_current_page = max(self.viewport_current_page - 1, 0)

    def find_on_page(self, query: str) -> Union[str, None]:
        """Searches for the query from the current viewport forward, looping back to the start if necessary."""

        # Did we get here via a previous find_on_page search with the same query?
        # If so, map to find_next
        if query == self._find_on_page_query and self.viewport_current_page == self._find_on_page_last_result:
            return self.find_next()

        # Ok it's a new search start from the current viewport
        self._find_on_page_query = query
        viewport_match = self._find_next_viewport(query, self.viewport_current_page)
        if viewport_match is None:
            self._find_on_page_last_result = None
            return None
        else:
            self.viewport_current_page = viewport_match
            self._find_on_page_last_result = viewport_match
            return self.viewport

    def find_next(self) -> Union[str, None]:
        """Scroll to the next viewport that matches the query"""

        if self._find_on_page_query is None:
            return None

        starting_viewport = self._find_on_page_last_result
        if starting_viewport is None:
            starting_viewport = 0
        else:
            starting_viewport += 1
            if starting_viewport >= len(self.viewport_pages):
                starting_viewport = 0

        viewport_match = self._find_next_viewport(self._find_on_page_query, starting_viewport)
        if viewport_match is None:
            self._find_on_page_last_result = None
            return None
        else:
            self.viewport_current_page = viewport_match
            self._find_on_page_last_result = viewport_match
            return self.viewport

    def _find_next_viewport(self, query: str, starting_viewport: int) -> Union[int, None]:
        """Search for matches between the starting viewport looping when reaching the end."""

        if query is None:
            return None

        # Normalize the query, and convert to a regular expression
        nquery = re.sub(r"\*", "__STAR__", query)
        nquery = " " + (" ".join(re.split(r"\W+", nquery))).strip() + " "
        nquery = nquery.replace(" __STAR__ ", "__STAR__ ")  # Merge isolated stars with prior word
        nquery = nquery.replace("__STAR__", ".*").lower()

        if nquery.strip() == "":
            return None

        idxs = list()
        idxs.extend(range(starting_viewport, len(self.viewport_pages)))
        idxs.extend(range(0, starting_viewport))

        for i in idxs:
            bounds = self.viewport_pages[i]
            content = self.page_content[bounds[0] : bounds[1]]

            # TODO: Remove markdown links and images
            ncontent = " " + (" ".join(re.split(r"\W+", content))).strip().lower() + " "
            if re.search(nquery, ncontent):
                return i

        return None

    def visit_page(self, path_or_uri: str, filter_year: Optional[int] = None) -> str:
        """Update the address, visit the page, and return the content of the viewport."""
        self.set_address(path_or_uri, filter_year=filter_year)
        return self.viewport

    def _split_pages(self) -> None:
        # Do not split search results
        if self.address.startswith("google:"):
            self.viewport_pages = [(0, len(self._page_content))]
            return

        # Handle empty pages
        if len(self._page_content) == 0:
            self.viewport_pages = [(0, 0)]
            return

        # Break the viewport into pages
        self.viewport_pages = []
        start_idx = 0
        while start_idx < len(self._page_content):
            end_idx = min(start_idx + self.viewport_size, len(self._page_content))  # type: ignore[operator]
            # Adjust to end on a space
            while end_idx < len(self._page_content) and self._page_content[end_idx - 1] not in [" ", "\t", "\r", "\n"]:
                end_idx += 1
            self.viewport_pages.append((start_idx, end_idx))
            start_idx = end_idx

    def _serpapi_search(self, query: str, filter_year: Optional[int] = None) -> None:
        if self.serpapi_key is None:
            raise ValueError("Missing SerpAPI key.")

        params = {
            "engine": "google",
            "q": query,
            "api_key": self.serpapi_key,
        }
        if filter_year is not None:
            params["tbs"] = f"cdr:1,cd_min:01/01/{filter_year},cd_max:12/31/{filter_year}"

        search = GoogleSearch(params)
        results = search.get_dict()
        self.page_title = f"{query} - Search"
        if "organic_results" not in results.keys():
            raise Exception(f"No results found for query: '{query}'. Use a less specific query.")
        if len(results["organic_results"]) == 0:
            year_filter_message = f" with filter year={filter_year}" if filter_year is not None else ""
            self._set_page_content(
                f"No results found for '{query}'{year_filter_message}. Try with a more general query, or remove the year filter."
            )
            return

        def _prev_visit(url):
            for i in range(len(self.history) - 1, -1, -1):
                if self.history[i][0] == url:
                    return f"You previously visited this page {round(time.time() - self.history[i][1])} seconds ago.\n"
            return ""

        web_snippets: List[str] = list()
        idx = 0
        if "organic_results" in results:
            for page in results["organic_results"]:
                idx += 1
                date_published = ""
                if "date" in page:
                    date_published = "\nDate published: " + page["date"]

                source = ""
                if "source" in page:
                    source = "\nSource: " + page["source"]

                snippet = ""
                if "snippet" in page:
                    snippet = "\n" + page["snippet"]

                redacted_version = f"{idx}. [{page['title']}]({page['link']}){date_published}{source}\n{_prev_visit(page['link'])}{snippet}"

                redacted_version = redacted_version.replace("Your browser can't play this video.", "")
                web_snippets.append(redacted_version)

        content = (
            f"A Google search for '{query}' found {len(web_snippets)} results:\n\n## Web Results\n"
            + "\n\n".join(web_snippets)
        )

        self._set_page_content(content)
    def _fetch_page(self, url: str) -> None:
        download_path = ""
        try:
            if url.startswith("file://"):
                download_path = os.path.normcase(os.path.normpath(unquote(url[7:])))
                res = self._mdconvert.convert_local(download_path)
                self.page_title = res.title if res else "No Title"
                self._set_page_content(res.text_content if res else "Unable to load content")
            else:
                # Prepare request parameters
                request_kwargs = self.request_kwargs.copy() if self.request_kwargs is not None else {}
                # Key modification: Don't use streaming request
                request_kwargs.pop("stream", None)  # Remove stream parameter if it exists
                
                # Send HTTP request - get all content at once
                response = requests.get(url, **request_kwargs)
                response.raise_for_status()
                
                # Save response content
                content_type = response.headers.get("content-type", "")
                
                # Text or HTML
                if "text/" in content_type.lower():
                    try:
                        # Save original HTML content
                        self._page_content_html = response.text
                        
                        # Convert content
                        res = self._mdconvert.convert(response, url=response.url)
                        if res is None:
                            # If conversion fails, create basic result
                            from collections import namedtuple
                            Result = namedtuple('Result', ['title', 'text_content'])
                            res = Result("Unable to parse page title", "Unable to parse page content. Original HTML has been saved, but conversion failed.")
                        
                        self.page_title = res.title
                        self._set_page_content(res.text_content)
                    except Exception as e:
                        # Catch errors during conversion process
                        self.page_title = f"Error: {str(e)[:50]}..."
                        self._set_page_content(f"Error parsing page: {str(e)}\n\nOriginal HTML content length: {len(self._page_content_html)} characters")
                # A download
                else:
                    # Try producing a safe filename
                    fname = None
                    download_path = None
                    try:
                        fname = pathvalidate.sanitize_filename(os.path.basename(urlparse(url).path)).strip()
                        download_path = os.path.abspath(os.path.join(self.downloads_folder, fname))

                        suffix = 0
                        while os.path.exists(download_path) and suffix < 1000:
                            suffix += 1
                            base, ext = os.path.splitext(fname)
                            new_fname = f"{base}__{suffix}{ext}"
                            download_path = os.path.abspath(os.path.join(self.downloads_folder, new_fname))

                    except NameError:
                        pass

                    # No suitable name, so make one
                    if fname is None:
                        extension = mimetypes.guess_extension(content_type)
                        if extension is None:
                            extension = ".download"
                        fname = str(uuid.uuid4()) + extension
                        download_path = os.path.abspath(os.path.join(self.downloads_folder, fname))

                    # Open a file for writing
                    with open(download_path, "wb") as fh:
                        for chunk in response.iter_content(chunk_size=512):
                            fh.write(chunk)

                    # Render it
                    local_uri = pathlib.Path(download_path).as_uri()
                    self.set_address(local_uri)

        except UnsupportedFormatException as e:
            print(e)
            self.page_title = ("Download complete.",)
            self._set_page_content(f"# Download complete\n\nSaved file to '{download_path}'")
        except FileConversionException as e:
            print(e)
            self.page_title = ("Download complete.",)
            self._set_page_content(f"# Download complete\n\nSaved file to '{download_path}'")
        except FileNotFoundError:
            self.page_title = "Error 404"
            self._set_page_content(f"## Error 404\n\nFile not found: {download_path}")
        except requests.exceptions.RequestException as request_exception:
            try:
                self.page_title = f"Error {response.status_code}"

                # If the error was rendered in HTML we might as well render it
                content_type = response.headers.get("content-type", "")
                if content_type is not None and "text/html" in content_type.lower():
                    res = self._mdconvert.convert(response)
                    self.page_title = f"Error {response.status_code}"
                    self._set_page_content(f"## Error {response.status_code}\n\n{res.text_content}")
                else:
                    text = ""
                    for chunk in response.iter_content(chunk_size=512, decode_unicode=True):
                        text += chunk
                    self.page_title = f"Error {response.status_code}"
                    self._set_page_content(f"## Error {response.status_code}\n\n{text}")
            except NameError:
                self.page_title = "Error"
                self._set_page_content(f"## Error\n\n{str(request_exception)}")

    def _state(self) -> Tuple[str, str]:
        header = f"Address: {self.address}\n"
        if self.page_title is not None:
            header += f"Title: {self.page_title}\n"

        current_page = self.viewport_current_page
        total_pages = len(self.viewport_pages)

        address = self.address
        for i in range(len(self.history) - 2, -1, -1):  # Start from the second last
            if self.history[i][0] == address:
                header += f"You previously visited this page {round(time.time() - self.history[i][1])} seconds ago.\n"
                break

        header += f"Viewport position: Showing page {current_page + 1} of {total_pages}.\n"
        return (header, self.viewport)

    def save_current_page_html(self, filename):
        """
        Save the HTML content of the current page to a file
        
        Args:
            filename: The name of the file to save the HTML content to
        """
        if hasattr(self, '_page_content_html') and self._page_content_html:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(self._page_content_html)
            return f"HTML content has been saved to {filename}"
        else:
            return "No HTML content available to save"


class SearchInformationTool_Image(Tool):
    name = "web_search"
    description = "Perform a web search query (think a google search) and returns the search results."
    inputs = {"query": {"type": "string", "description": "The web search query to perform."}}
    inputs["filter_year"] = {
        "type": "string",
        "description": "[Optional parameter]: filter the search results to only include pages from a specific year. For example, '2020' will only include pages from 2020. Make sure to use this parameter if you're trying to search for articles from a specific date!",
        "nullable": True,
    }
    output_type = "string"

    def __init__(self, browser):
        super().__init__()
        self.browser = browser

    def forward(self, query: str, filter_year: Optional[int] = None) -> str:
        self.browser.visit_page(f"google: {query}", filter_year=filter_year)
        header, content = self.browser._state()
        return header.strip() + "\n=======================\n" + content


class VisitTool_Image(Tool):
    name = "visit_page"
    description = "Visit a webpage at a given URL and return its text. Given a url to a YouTube video, this returns the transcript."
    inputs = {"url": {"type": "string", "description": "The relative or absolute url of the webpage to visit."}}
    output_type = "string"

    def __init__(self, browser):
        super().__init__()
        self.browser = browser

    def forward(self, url: str) -> str:
        self.browser.visit_page(url)
        header, content = self.browser._state()
        return header.strip() + "\n=======================\n" + content


class DownloadTool_Image(Tool):
    name = "download_file"
    description = """
Download a file at a given URL. The file should be of this format: [".xlsx", ".pptx", ".wav", ".mp3", ".m4a", ".png", ".docx"]
After using this tool, for further inspection of this page you should return the download path to your manager via final_answer, and they will be able to inspect it.
DO NOT use this tool for .pdf or .txt or .htm files: for these types of files use visit_page with the file url instead."""
    inputs = {"url": {"type": "string", "description": "The relative or absolute url of the file to be downloaded."}}
    output_type = "string"

    def __init__(self, browser):
        super().__init__()
        self.browser = browser

    def forward(self, url: str) -> str:
        if "arxiv" in url:
            url = url.replace("abs", "pdf")
        response = requests.get(url)
        content_type = response.headers.get("content-type", "")
        extension = mimetypes.guess_extension(content_type)
        if extension and isinstance(extension, str):
            new_path = f"./downloads/file{extension}"
        else:
            new_path = "./downloads/file.object"

        with open(new_path, "wb") as f:
            f.write(response.content)

        if "pdf" in extension or "txt" in extension or "htm" in extension:
            raise Exception("Do not use this tool for pdf or txt or html files: use visit_page instead.")

        return f"File was downloaded and saved under path {new_path}."


class ArchiveSearchTool_Image(Tool):
    name = "find_archived_url"
    description = "Given a url, searches the Wayback Machine and returns the archived version of the url that's closest in time to the desired date."
    inputs = {
        "url": {"type": "string", "description": "The url you need the archive for."},
        "date": {
            "type": "string",
            "description": "The date that you want to find the archive for. Give this date in the format 'YYYYMMDD', for instance '27 June 2008' is written as '20080627'.",
        },
    }
    output_type = "string"

    def __init__(self, browser):
        super().__init__()
        self.browser = browser

    def forward(self, url, date) -> str:
        no_timestamp_url = f"https://archive.org/wayback/available?url={url}"
        archive_url = no_timestamp_url + f"&timestamp={date}"
        response = requests.get(archive_url).json()
        response_notimestamp = requests.get(no_timestamp_url).json()
        if "archived_snapshots" in response and "closest" in response["archived_snapshots"]:
            closest = response["archived_snapshots"]["closest"]
            print("Archive found!", closest)

        elif "archived_snapshots" in response_notimestamp and "closest" in response_notimestamp["archived_snapshots"]:
            closest = response_notimestamp["archived_snapshots"]["closest"]
            print("Archive found!", closest)
        else:
            raise Exception(f"Your {url=} was not archived on Wayback Machine, try a different url.")
        target_url = closest["url"]
        self.browser.visit_page(target_url)
        header, content = self.browser._state()
        return (
            f"Web archive for url {url}, snapshot taken at date {closest['timestamp'][:8]}:\n"
            + header.strip()
            + "\n=======================\n"
            + content
        )


class PageUpTool_Image(Tool):
    name = "page_up"
    description = "Scroll the viewport UP one page-length in the current webpage and return the new viewport content."
    inputs = {"empty_string": {"type": "string", "description": "Empty string."}}
    output_type = "string"

    def __init__(self, browser):
        super().__init__()
        self.browser = browser

    def forward(self,empty_string) -> str:
        self.browser.page_up()
        header, content = self.browser._state()
        return header.strip() + "\n=======================\n" + content

class PageDownTool_Image(Tool):
    name = "page_down"
    description = "Scroll the viewport DOWN one page-length in the current webpage and return the new viewport content."
    inputs = {"empty_string": {"type": "string", "description": "Empty string."}}
    output_type = "string"

    def __init__(self, browser):
        super().__init__()
        self.browser = browser

    def forward(self, empty_string) -> str:
        self.browser.page_down()
        header, content = self.browser._state()
        return header.strip() + "\n=======================\n" + content


class FinderTool_Image(Tool):
    name = "find_on_page_ctrl_f"
    description = "Scroll the viewport to the first occurrence of the search string. This is equivalent to Ctrl+F."
    inputs = {
        "search_string": {
            "type": "string",
            "description": "The string to search for on the page. This search string supports wildcards like '*'",
        }
    }
    output_type = "string"

    def __init__(self, browser):
        super().__init__()
        self.browser = browser

    def forward(self, search_string: str) -> str:
        find_result = self.browser.find_on_page(search_string)
        header, content = self.browser._state()

        if find_result is None:
            return (
                header.strip()
                + f"\n=======================\nThe search string '{search_string}' was not found on this page."
            )
        else:
            return header.strip() + "\n=======================\n" + content


class FindNextTool_Image(Tool):
    name = "find_next"
    description = "Scroll the viewport to next occurrence of the search string. This is equivalent to finding the next match in a Ctrl+F search."
    inputs = {"empty_string": {"type": "string", "description": "Empty string."}}
    output_type = "string"

    def __init__(self, browser):
        super().__init__()
        self.browser = browser

    def forward(self,empty_string) -> str:
        find_result = self.browser.find_next()
        header, content = self.browser._state()

        if find_result is None:
            return (
                header.strip()
                + "\n=======================\nThe search string was not found on this page."
            )
        else:
            return header.strip() + "\n=======================\n" + content

from bs4 import BeautifulSoup
import re

class ImageContextExtractorTool(Tool):
    name = "extract_image_context"
    description = "Extract context description text related to a specific image URL from links returned by reverse_image_search_tool, not the initially provided image URL. Extracts text related to the image, including alt attributes, title attributes, figcaption attributes, parent elements, context, and surrounding elements. Generally used together with reverse_image_search_tool to extract context of an image based on the original image URL."
    inputs = {
        "image_url": {
            "type": "string", 
            "description": "The image URL or part of URL to find."
        },
        "context_range": {
            "type": "integer",
            "description": "[Optional] Number of elements around the image to extract as context, default is 3.",
            "nullable": True
        }
    }
    output_type = "string"

    def __init__(self, browser):
        super().__init__()
        self.browser = browser

    def forward(self, image_url: str, context_range: Optional[int] = 3) -> str:
        """Extract context description related to a specific image URL"""
        # Get current page content
        if not hasattr(self.browser, '_page_content_html'):
            return "Error: Current page has no HTML content or no page has been visited yet."
            
        html_content = self.browser._page_content_html
        if not html_content:
            return "Error: Current page has no HTML content."
            
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Find image elements containing the URL
            images = []
            for img in soup.find_all('img'):
                if img.has_attr('src') and image_url in img['src']:
                    images.append(img)
                    
            # Find link elements containing the URL
            links = []
            for a in soup.find_all('a'):
                if a.has_attr('href') and image_url in a['href']:
                    links.append(a)
                    
            if not images and not links:
                return f"No images or links containing URL '{image_url}' were found."
                
            results = []
            
            # Process found images
            for img in images:
                result = ["## Image Found:"]
                
                # Extract image attributes
                if img.has_attr('alt') and img['alt'].strip():
                    result.append(f"Image description (alt): {img['alt']}")
                    
                if img.has_attr('title') and img['title'].strip():
                    result.append(f"Image title: {img['title']}")
                
                # Find parent elements of the image
                parent = img.parent
                if parent:
                    # Check if parent is a figure element
                    if parent.name == 'figure':
                        # Look for figcaption
                        figcaption = parent.find('figcaption')
                        if figcaption:
                            result.append(f"Image caption (figcaption): {figcaption.get_text(strip=True)}")
                
                # Get context around the image
                result.append("\n### Image Context:")
                
                # Look up for the nearest meaningful container
                container = img
                for _ in range(4):  # Look up maximum 4 levels
                    if container.parent and container.parent.name not in ['html', 'body']:
                        container = container.parent
                    else:
                        break
                
                # Extract text content from the container
                paragraphs = container.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'div', 'span', 'li'])
                if paragraphs:
                    for p in paragraphs[:context_range]:
                        text = p.get_text(strip=True)
                        if text:
                            result.append(f"- {text}")
                else:
                    # If no paragraphs found, extract container text directly
                    text = container.get_text(strip=True)
                    if text:
                        result.append(f"- {text}")
                
                results.append("\n".join(result))
            
            # Process found links
            for a in links:
                result = ["## Link Found:"]
                
                # Extract link text
                link_text = a.get_text(strip=True)
                if link_text:
                    result.append(f"Link text: {link_text}")
                
                # Extract link title
                if a.has_attr('title') and a['title'].strip():
                    result.append(f"Link title: {a['title']}")
                
                # Check if link contains an image
                img_in_link = a.find('img')
                if img_in_link:
                    if img_in_link.has_attr('alt') and img_in_link['alt'].strip():
                        result.append(f"Image description in link: {img_in_link['alt']}")
                
                # Get context around the link
                result.append("\n### Link Context:")
                
                # Look up for the nearest meaningful container
                container = a
                for _ in range(4):  # Look up maximum 4 levels
                    if container.parent and container.parent.name not in ['html', 'body']:
                        container = container.parent
                    else:
                        break
                
                # Extract text content from the container
                paragraphs = container.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'div', 'span', 'li'])
                if paragraphs:
                    for p in paragraphs[:context_range]:
                        text = p.get_text(strip=True)
                        if text and text != link_text:  # Avoid duplicate link text
                            result.append(f"- {text}")
                else:
                    # If no paragraphs found, extract container text directly
                    text = container.get_text(strip=True)
                    if text and text != link_text:  # Avoid duplicate link text
                        result.append(f"- {text}")
                
                results.append("\n".join(result))
            
            return "\n\n".join(results)
            
        except Exception as e:
            return f"Error extracting image context: {str(e)}"

class SaveHTMLTool(Tool):
    name = "save_html"
    description = "Save the current webpage's HTML content to a file"
    inputs = {"filename": {"type": "string", "description": "Filename to save the HTML content"}}
    output_type = "string"

    def __init__(self, browser):
        super().__init__()
        self.browser = browser

    def forward(self, filename: str) -> str:
        """Save the current page's HTML content to a file"""
        if hasattr(self.browser, '_page_content_html') and self.browser._page_content_html:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(self.browser._page_content_html)
            return f"HTML content has been saved to {filename}"
        else:
            return "No HTML content available to save"

class VisitImageSearchResultsTool(Tool):
    name = "visit_image_search_results"
    description = "Visit links from image reverse search results and extract image context"
    inputs = {
        "search_results": {"type": "string", "description": "Text results from image reverse search"},
        "max_links": {
            "type": "integer", 
            "description": "[Optional] Maximum number of links to visit, default is 3",
            "nullable": True
        }
    }
    output_type = "string"

    def __init__(self, browser):
        super().__init__()
        self.browser = browser

    def forward(self, search_results: str, max_links: Optional[int] = 3) -> str:
        """
        Extract links from image reverse search results, visit these links and extract image context
        """
        # Parse search results, extract links
        links = []
        image_urls = []
        
        # Use regular expressions to extract links and image URLs
        # Support both English and Chinese patterns
        link_pattern = re.compile(r'- (?:链接|Link): (https?://[^\s]+)')
        image_pattern = re.compile(r'- (?:原图|缩略图|Thumbnail|Image|Original): (https?://[^\s]+)')
        
        for match in link_pattern.finditer(search_results):
            links.append(match.group(1))
            
        for match in image_pattern.finditer(search_results):
            image_urls.append(match.group(1))
            
        if not links:
            return "No links found in search results"
            
        # Limit number of links
        links = links[:max_links]
        
        results = []
        for i, link in enumerate(links):
            try:
                # Visit link
                results.append(f"\n## Visiting link {i+1}: {link}")
                self.browser.visit_page(link)
                
                # If there's a corresponding image URL, extract context
                if i < len(image_urls):
                    image_url = image_urls[i]
                    # Extract image filename as search keyword
                    image_filename = image_url.split('/')[-1]
                    
                    # Direct HTML search approach
                    html_content = self.browser._page_content_html
                    if html_content:
                        # First try: direct search for the exact image filename in HTML
                        if image_filename in html_content:
                            results.append(f"\n### Image found in HTML source!")
                            
                            # Try to extract surrounding context using regex
                            # Look for HTML elements containing the image filename
                            surrounding_pattern = re.compile(r'([^>]{0,100}' + re.escape(image_filename) + r'[^<]{0,100})')
                            matches = surrounding_pattern.findall(html_content)
                            
                            if matches:
                                results.append("Surrounding HTML context:")
                                for j, match in enumerate(matches[:5]):  # Limit to first 5 matches
                                    results.append(f"Context {j+1}: {match}")
                            
                            # Try to find the image element using BeautifulSoup
                            soup = BeautifulSoup(html_content, 'html.parser')
                            
                            # Look for img tags with src containing the filename
                            img_tags = soup.find_all('img', src=lambda src: src and image_filename in src)
                            
                            if img_tags:
                                results.append(f"\nFound {len(img_tags)} matching image tags:")
                                
                                for j, img in enumerate(img_tags[:3]):  # Limit to first 3 images
                                    results.append(f"Image {j+1}:")
                                    results.append(f"- Tag: {img}")
                                    
                                    # Get attributes
                                    for attr, value in img.attrs.items():
                                        results.append(f"- {attr}: {value}")
                                    
                                    # Get parent element and its text
                                    parent = img.parent
                                    if parent:
                                        parent_text = parent.get_text(strip=True)
                                        if parent_text:
                                            results.append(f"- Parent text: {parent_text[:200]}...")
                                            
                                        # Get grandparent for more context
                                        grandparent = parent.parent
                                        if grandparent:
                                            results.append(f"- Section: {grandparent.name}")
                                            section_text = grandparent.get_text(strip=True)
                                            if section_text and len(section_text) > len(parent_text):
                                                results.append(f"- Section text: {section_text[:300]}...")
                            else:
                                # If no direct img tags found, try looking for the filename in any attribute
                                elements_with_filename = soup.find_all(lambda tag: any(image_filename in str(value) for value in tag.attrs.values()) if tag.attrs else False)
                                
                                if elements_with_filename:
                                    results.append(f"\nFound {len(elements_with_filename)} elements containing the filename in attributes:")
                                    for j, elem in enumerate(elements_with_filename[:3]):
                                        results.append(f"Element {j+1}: {elem.name}")
                                        results.append(f"- HTML: {elem}")
                                        elem_text = elem.get_text(strip=True)
                                        if elem_text:
                                            results.append(f"- Text: {elem_text[:200]}...")
                        else:
                            # If exact filename not found, try with just the base name
                            base_filename = os.path.splitext(image_filename)[0]
                            if base_filename in html_content:
                                results.append(f"\n### Base filename '{base_filename}' found in HTML source!")
                                # Use the same extraction logic as above with base_filename
                                # (Code omitted for brevity but would be similar to the above)
                            else:
                                # Fall back to the original extractor
                                extractor = ImageContextExtractorTool(self.browser)
                                context = extractor.forward(image_filename)
                                results.append(f"\n### Image context:")
                                results.append(context)
                    else:
                        results.append("No HTML content available for analysis")
                else:
                    results.append("No corresponding image URL found")
                    
            except Exception as e:
                results.append(f"Error visiting link {link}: {str(e)}")
                
        return "\n".join(results)

class GetRawHTMLTool(Tool):
    name = "get_raw_html"
    description = "Get the raw HTML content of the current page"
    inputs = {}
    output_type = "string"

    def __init__(self, browser):
        super().__init__()
        self.browser = browser

    def forward(self) -> str:
        """Return the raw HTML content of the current page"""
        if hasattr(self.browser, '_page_content_html') and self.browser._page_content_html:
            # Return first 1000 characters as preview
            preview = self.browser._page_content_html[:1000] + "...\n[HTML content truncated, total length: " + str(len(self.browser._page_content_html)) + " characters]"
            return preview
        else:
            return "No HTML content available"
        
def test_pdf_image_extraction():
    # Load environment variables
    load_dotenv(override=True)
    
    # Create browser instance
    browser = SimpleImageBrowser()
    
    # Create image context extraction tool
    extractor = ImageContextExtractorTool(browser)
    
    # PDF file path
    pdf_path = "https://ed.arte.gov.tw/uploadfile/Book/3481_古典書法的範疇.pdf"  # Please replace with the actual PDF file path
    
    try:
        # Visit PDF file
        print(f"Visiting PDF file: {pdf_path}")
        browser.visit_page(pdf_path)
        
        # Get all images from PDF
        html_content = browser._page_content_html
        if html_content:
            # Use BeautifulSoup to parse HTML content
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Find all images
            images = soup.find_all('img')
            
            if images:
                print(f"\nFound {len(images)} images in PDF")
                
                # Extract context for each image
                for i, img in enumerate(images):
                    img_url = img.get('src', '')
                    if img_url:
                        print(f"\nProcessing image {i+1}:")
                        print(f"Image URL: {img_url}")
                        
                        # Extract image context
                        context = extractor.forward(img_url)
                        print("\nImage context information:")
                        print(context)
            else:
                print("No images found in PDF")
        else:
            print("Unable to get PDF content")
            
    except Exception as e:
        print(f"Error occurred during processing: {str(e)}")

if __name__ == "__main__":
    test_pdf_image_extraction()
        