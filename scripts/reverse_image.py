import requests
import json
import os
import time
from typing import Dict, Any, List, Optional, Tuple
from smolagents import Tool

class GoogleLensManager:
    """Simple manager for Google Lens search functionality"""

    def __init__(
        self,
        imgbb_api_key: str,
        serpapi_api_key: str,
        search_api_key: str,
        downloads_folder: Optional[str] = "downloads"
    ):
        self.imgbb_api_key = imgbb_api_key
        self.serpapi_api_key = serpapi_api_key
        self.search_api_key = search_api_key
        self.downloads_folder = downloads_folder
        self.history: List[Tuple[str, float]] = list()  # Record search history
        self.current_results: Dict = {}  # Current search results

        # Ensure the downloads folder exists
        os.makedirs(downloads_folder, exist_ok=True)

    def _upload_to_imgbb(self, image_path: str) -> Optional[str]:
        """Upload an image to ImgBB and return the URL"""
        # Add file existence check
        if not os.path.exists(image_path):
            print(f"Error: File does not exist - {image_path}")
            return None

        try:
            # Check if the file is readable
            if not os.access(image_path, os.R_OK):
                print(f"Error: File is not accessible - {image_path}")
                return None

            with open(image_path, "rb") as f:
                response = requests.post(
                    "https://api.imgbb.com/1/upload",
                    params={"key": self.imgbb_api_key},
                    files={"image": f}
                )
            if response.status_code == 200:
                return response.json()["data"]["url"]
            else:
                print(f"Upload failed, status code: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error uploading image: {str(e)}")
            return None

    def _google_lens_search(self, image_url: str) -> Dict:
        """Search an image using Google Lens via SearchAPI.io"""
        if not self.search_api_key:
            print("Error: search_api_key not provided")
            return {"error": "Missing search_api_key"}

        url = "https://www.searchapi.io/api/v1/search"
        params = {
            "engine": "google_lens",
            "search_type": "all",
            "url": image_url,
            "api_key": self.search_api_key  # Replace with your actual API key if needed
        }

        try:
            response = requests.get(url, params=params)
            results = response.text
            # Record search history
            self.history.append((image_url, time.time()))
            self.current_results = results
            return results
        except Exception as e:
            print(f"Google Lens search error: {str(e)}")
            return {"error": str(e)}

    def get_visual_matches(self, limit: int = 5) -> List[Dict]:
        """Get visual matches results"""
        if "visual_matches" in self.current_results:
            return self.current_results["visual_matches"][:limit]
        return []

    def get_related_content(self, limit: int = 5) -> List[Dict]:
        """Get related content (not directly available in the new format)"""
        return []

    def get_urls(self, limit: int = 5) -> List[str]:
        """Get all related URLs from visual matches"""
        urls = []

        # Get URLs from visual matches
        for match in self.get_visual_matches(limit):
            if "link" in match:
                urls.append(match["link"])

        return list(set(urls))[:limit]  # Deduplicate and limit quantity

    def search(self, image_path: str) -> Dict:
        """Perform image search and return results"""
        # Normalize path
        image_path = os.path.normpath(image_path)

        # Check if file exists
        if not os.path.exists(image_path):
            error_msg = f"File does not exist: {image_path}"
            print(error_msg)
            return {"error": error_msg}

        # Check if file is readable
        if not os.access(image_path, os.R_OK):
            error_msg = f"File is not accessible: {image_path}"
            print(error_msg)
            return {"error": error_msg}

        # Check file extension
        ext = os.path.splitext(image_path)[1].lower()
        if ext not in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']:
            error_msg = f"Unsupported file format: {ext}"
            print(error_msg)
            return {"error": error_msg}

        # Upload image
        image_url = self._upload_to_imgbb(image_path)
        if not image_url:
            error_msg = "Image upload failed"
            print(error_msg)
            return {"error": error_msg}

        # Search image
        results_raw = self._google_lens_search(image_url)
        if "error" in results_raw:
            return json.loads(results_raw) # Ensure error is parsed

        try:
            results = json.loads(results_raw)
            self.current_results = results
        except json.JSONDecodeError as e:
            error_msg = f"Error decoding JSON response: {e}"
            print(error_msg)
            return {"error": error_msg, "raw_response": results_raw}

        return {
            "image_url": image_url,
            "visual_matches": self.get_visual_matches(),
            "related_content": self.get_related_content(),
            "urls": self.get_urls()
        }

    @property
    def last_search(self) -> Optional[Tuple[str, float]]:
        """Get information about the most recent search"""
        return self.history[-1] if self.history else None

class GoogleLensSearchTool(Tool):
    name = "google_lens_search"
    description = """Use Google Lens to search images and return related webpage URLs and visual matches.
    Input an image file path, returns:
    1. Online URL of the image
    2. Visual match results
    3. Related webpage links
    Note: Image must be a valid image file (.jpg, .png, .jpeg, etc.)"""

    inputs = {
        "image_path": {
            "type": "string",
            "description": "Path to the image file to search, please note relative paths",
            "nullable": True
        },
        "limit": {
            "type": "integer",
            "description": "Maximum number of results to return",
            "default": 5,
            "nullable": True
        }
    }
    output_type = "string"

    def __init__(self, imgbb_api_key: str, serpapi_api_key: str, search_api_key: str):
        super().__init__()
        self.manager = GoogleLensManager(imgbb_api_key, serpapi_api_key, search_api_key)

    def forward(self, image_path: str = None, limit: int = 5) -> str:
        """Perform search and return formatted results"""
        if not image_path:
            return "Error: No image path provided"

        # Normalize path
        try:
            image_path = os.path.normpath(image_path)
        except Exception as e:
            return f"Error: Invalid path format - {str(e)}"

        # Validate file format
        ext = os.path.splitext(image_path)[1].lower()
        if ext not in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']:
            return f"Error: Unsupported image format {ext}"

        # Execute search
        try:
            results = self.manager.search(image_path)
        except Exception as e:
            return f"Error: Exception occurred during search - {str(e)}"

        if "error" in results:
            return f"Error: {results['error']}"
 
        # Reformat output
        output = [
            f"# Google Lens Search Results",
            f"Original image path: {image_path}",
            f"Online image URL: {results['image_url']}",
            "\n## Visual Matches:"
        ]

        # Format visual matches
        if results.get('visual_matches'):
            for i, match in enumerate(results['visual_matches'][:limit], 1):
                output.extend([
                    f"\n### Match {i}:",
                    f"- Title: {match.get('title', 'Unknown title')}",
                    f"- Source: {match.get('source', 'Unknown source')}",
                    f"- Link: {match.get('link', 'No link')}",
                    f"- Image URL: {match.get('image', {}).get('link', match.get('image', 'No image URL'))}" # Handle both image link formats
                ])

        output.append("\n## Related URLs:")
        if results.get('urls'):
            for i, url in enumerate(results['urls'][:limit], 1):
                output.append(f"{i}. {url}")
        else:
            output.append("No related URLs found.")
        output_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reverse_image_results_histbench")
        os.makedirs(output_folder, exist_ok=True)
        base_filename = os.path.basename(image_path)
        txt_filename = os.path.splitext(base_filename)[0] + ".txt"
        txt_filepath = os.path.join(output_folder, txt_filename)
        with open(txt_filepath, "w") as f:
            f.write("\n".join(output))
        return "\n".join(output)
