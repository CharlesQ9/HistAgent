# Open Deep Research

Welcome to this open replication of [OpenAI's Deep Research]!

To install it, first run
```bash
pip install -r requirements.txt
```

And install smolagents dev version
```bash
pip install smolagents[dev]
```

## Environment Setup

This system requires several environment variables to be configured for proper operation. You can set these variables in your shell session or add them to your `.env` file.

### Required Environment Variables

- **OPENAI_API_KEY**: Required for accessing OpenAI's models (GPT-4, etc.)
  - Used for the main agent's reasoning and processing capabilities
  - Must be a valid API key with sufficient quota

### Optional Environment Variables

- **SERPAPI_API_KEY**: Used for enhanced web search capabilities
  - Enables more accurate and comprehensive search results
  - Recommended for complex historical queries requiring precise information

- **IMGBB_API_KEY**: Used for image hosting and processing
  - Required when using the image agent for uploading and analyzing images
  - Enables reverse image search and other image-based research features

- **SPRINGER_API_KEY**: Provides access to Springer Nature academic resources
  - Used by the literature agent for academic paper searches
  - Falls back to a built-in key if not provided

- **LLAMA_API_KEY**: Used for LlamaParse document processing
  - Required for advanced PDF parsing and document analysis
  - Falls back to a built-in key if not provided

### Setting Up Environment Variables

You can set these variables using the following methods:

1. **Direct export in your terminal session**:
```bash
export OPENAI_API_KEY="your_api_key_here"
export SERPAPI_API_KEY="your_serpapi_key_here"
export IMGBB_API_KEY="your_imgbb_key_here"
```

2. **Using a .env file**:
Create a file named `.env` in the project root directory with the following content:
```
OPENAI_API_KEY=your_api_key_here
SERPAPI_API_KEY=your_serpapi_key_here
IMGBB_API_KEY=your_imgbb_key_here
SPRINGER_API_KEY=your_springer_key_here
LLAMA_API_KEY=your_llama_key_here
```

3. **Command line parameters**:
Some API keys can also be provided directly as command line parameters:
```bash
python run_hist.py --run-name "test_run" --api-key "your_openai_key" --springer-api-key "your_springer_key" --llama-api-key "your_llama_key"
```

## Custom browser_use Library

After installation, you need to replace the library files to use the custom browser_use implementation:

```bash
# Replace the installed library with the provided browser_use folder
# First, find the Python library installation path
python -c "import site; print(site.getsitepackages()[0])"
# Then copy the browser_use folder to that path
cp -r browser_use /path/to/your/site-packages/
```

This custom version of the browser_use library contains specific optimizations and fixes that must be replaced to ensure the system works properly.

Then you're good to go! Run the run.py script, as in:
```bash
python run.py --model-id "o1" "Your question here!"
```

## Chinese History Q&A System

This system is an automated question-answering system for Chinese history questions, utilizing large language models (such as GPT-4) combined with a specialized agent hierarchy to answer historical questions.

## Running the System

Make sure you have configured the environment variables:

```
export OPENAI_API_KEY="your OpenAI API key"
export SERPAPI_API_KEY="your SerpApi API key (optional)"
export IMGBB_API_KEY="your ImgBB API key (optional)"
```

Then, use the following command to run the program:

```bash
cd HistoryDeepResearch/smolagents/examples/open_deep_research
python run_hist.py --run-name "test_run" --use-image-agent --use-file-agent
```

## Command Line Arguments

- `--concurrency`: Number of concurrent tasks (default: 8)
- `--model-id`: Model ID (default: gpt-4o)
- `--run-name`: Run name (required)
- `--api-key`: OpenAI API key (defaults to environment variable)
- `--use-image-agent`: Enable image information agent for analyzing images and performing reverse image searches
- `--use-file-agent`: Enable file processing agent for handling various file types
- `--use-literature-agent`: Enable literature search agent for finding and analyzing academic literature
- `--no-agent-webbrowser-agent`: Disable text browser agent (enabled by default)
- `--use-springer`: Enable Springer academic resource tools (enabled by default)
- `--no-springer`: Disable Springer academic resource tools
- `--use-browser`: Enable interactive browser functionality for literature search tools
- `--baseline`: Use baseline agent instead of agent hierarchy, results stored in output_baseline/ directory
- `--level`: Specify the question level to test, options are "level1", "level2", or "level3" (default: level2)
- `--results-json-path`: Specify the path to a previous run's JSON results file, used to filter out already correctly answered questions
- `--question-ids`: Specify specific question IDs to run, comma-separated (e.g., "16,24,35"). Can use numeric IDs (level prefix will be added automatically) or full IDs
- `--start-id`: Specify the starting ID of the question ID range to run
- `--end-id`: Specify the ending ID of the question ID range to run
- `--springer-api-key`: Springer Nature API key (defaults to environment variable or built-in key)
- `--llama-api-key`: LlamaParse API key (defaults to environment variable or built-in key)

### Example Commands

Using standard agent hierarchy (for level2 questions):
```bash
python run_hist.py --run-name "test_all" --use-image-agent --use-file-agent
```

Using baseline agent (for level1 questions):
```bash
python run_hist.py --run-name "baseline_test" --baseline --level level1
```

Using full-featured agent hierarchy (including literature search, for level3 questions):
```bash
python run_hist.py --run-name "full_research" --use-image-agent --use-file-agent --use-literature-agent --level level3
```

Disabling Springer academic resource tools:
```bash
python run_hist.py --run-name "no_springer" --use-literature-agent --no-springer
```

Running literature search tools with interactive browser functionality:
```bash
python run_hist.py --run-name "interactive_browser" --use-literature-agent --use-browser
```

Running specific question IDs (e.g., running questions 16, 24, and 35 from level2):
```bash
python run_hist.py --run-name "specific_questions" --level level2 --question-ids "16,24,35"
```

Running a range of questions (e.g., running questions 10 through 30 from level1):
```bash
python run_hist.py --run-name "question_range" --level level1 --start-id 10 --end-id 30
```

Combining multiple features (running questions 5 and 8 from level3 with image agent):
```bash
python run_hist.py --run-name "combined_example" --level level3 --use-image-agent --question-ids "5,8"
```

## Running with HLE.json Dataset

The system also provides the `run_histjson.py` script, which functions the same as `run_hist.py` but uses a dataset in HLE.json format instead of the default dataset. This script is suitable for history Q&A datasets already formatted in JSON.

### Basic Usage

```bash
python run_histjson.py --run-name "json_dataset_test" --use-image-agent --use-file-agent
```

### Command Line Arguments

`run_histjson.py` supports all the same command line arguments as `run_hist.py`, including agent selection, concurrency settings, model selection, etc. The only difference is that it processes data from the HLE.json format.

### Example Commands

Using standard agent hierarchy with JSON dataset:
```bash
python run_histjson.py --run-name "json_test" --use-image-agent --use-file-agent
```

Using baseline agent with specific question IDs:
```bash
python run_histjson.py --run-name "json_baseline" --baseline --question-ids "10,15,20"
```

## Output Files

After running, the program generates the following files in the specified directory (default is `output/level2_summary` or `output_baseline/level2_summary`):

- JSONL file: Contains detailed execution results
- Excel file: For data analysis
- TXT file: Human-readable output
- Statistics file: Records accuracy analysis
- Log files: For monitoring and debugging

## Logging System

The program uses Python's logging module to record the running process. Log files are stored in the `output/{SET}/logs/` directory:

- `main.log`: Contains the main program flow and overall running information
- `task_{task_id}.log`: Logs for each specific question, containing detailed information about question processing
- `errors.log`: Records all errors and exceptions

## Agent Description

The system supports the following agents:

- **Image Information Agent**: Processes image files, performs OCR recognition, reverse image searches, etc.
- **File Processing Agent**: Handles various types of files such as PDF, Word, Excel, etc.
- **Web Browsing Agent**: Performs online search queries and information retrieval
- **Literature Search Agent**: Specialized agent for searching and analyzing academic literature, including the following tools:
  - `literature_searching_task`: Searches for high-impact, recent academic articles in Google Scholar
  - `general_browser_task`: Performs general web searches
  - `springer_search`: Searches for academic papers on Springer Nature's open access platform (can be disabled with `--no-springer`)
  - Interactive browser functionality: Enabled with the `--use-browser` parameter, allows agents to interact directly with web pages for more precise literature information
- **Baseline Agent**: Simplified agent used with the `--baseline` parameter, without complex hierarchy

## Academic Literature Search Capabilities

The system now has enhanced support for historical academic research:

1. **High-Quality Literature Search**: Finds high-citation, high-impact academic articles from scholarly databases like Google Scholar
2. **Intelligent Literature Filtering**: Automatically filters and sorts the most relevant literature based on relevance and citation impact
3. **Full-Text Information Extraction**: Extracts key information from filtered literature, supporting precise matching for fill-in-the-blank questions
4. **Springer Nature Integration**: Direct access to academic resources on Springer Nature's open access platform
   - Supports traditional and structured academic searches
   - Uses LlamaParse to process and analyze PDF documents
   - Can be disabled with the `--no-springer` parameter

These features are especially suitable for:
- Questions requiring authoritative historical sources
- Questions requiring research literature on specific historical periods or events
- Complex historical questions requiring corroboration from multiple sources
- In-depth historical research requiring access to specialized academic databases

## Results Combination Tool

To facilitate analysis of results produced by different models or strategies, we provide a dedicated results combination tool `combine_results.py`, which can extract correct answers from multiple result files and merge them into a single file, while supporting multiple output formats.

### Features

- **Multi-file Merging**: Merges correct answers from multiple JSONL result files
- **Conflict Resolution Strategies**: Supports various conflict resolution strategies (keep first, keep latest, select by model)
- **Multi-format Output**: Supports JSONL, Excel, and TXT format outputs
- **Statistical Analysis**: Generates detailed statistics including file contributions, task type distribution, and model performance
- **Automatic Report Generation**: Optionally generates a README.md report file for easy result sharing and analysis

### Usage

```bash
python combine_results.py [input_files] [options]
```

### Command Line Arguments

| Parameter | Description |
|------|------|
| `input_files` | Input JSONL file paths, multiple files can be specified |
| `--output-dir`, `-o` | Output directory, default is output/combined |
| `--output-name` | Output file name (without extension), default uses timestamp |
| `--conflict-strategy` | Conflict resolution strategy: first=keep first correct answer, latest=keep latest answer, model=keep answer from specified model |
| `--preferred-model` | When using model conflict strategy, specify the preferred model ID |
| `--formats` | Specify output formats, options are jsonl, excel, txt, or all (all formats), default is all |
| `--add-readme` | Generate a README.md file in the output directory explaining the merge process and results |
| `--level` | Filter questions of a specific level, options are level1, level2, level3, or all (all levels), default is all |
| `--copy-images` | Copy images related to correct answers to the output directory |
| `--images-dir` | Root directory containing image files, default is dataset |

### Usage Examples

1. Merge correct answers from two result files:

```bash
python combine_results.py output/results1.jsonl output/results2.jsonl
```

2. Merge multiple files and specify output directory and name:

```bash
python combine_results.py output/*.jsonl --output-dir analysis --output-name combined_results
```

3. Use specific conflict resolution strategy:

```bash
python combine_results.py output/*.jsonl --conflict-strategy latest
```

4. Prioritize results from a specific model:

```bash
python combine_results.py output/*.jsonl --conflict-strategy model --preferred-model "gpt-4"
```

5. Output only in Excel format:

```bash
python combine_results.py output/*.jsonl --formats excel
```

6. Generate detailed README report:

```bash
python combine_results.py output/*.jsonl --add-readme
```

7. Filter questions of a specific level:

```bash
python combine_results.py output/*.jsonl --level level2
```

8. Copy related image files:

```bash
python combine_results.py output/*.jsonl --copy-images --images-dir dataset/images
```

9. Combine multiple parameters:

```bash
python combine_results.py output/*.jsonl --level level3 --conflict-strategy model --preferred-model "gpt-4-turbo" --formats jsonl txt --add-readme --copy-images
```
### Image File Processing

When using the `--copy-images` parameter, the combination tool will automatically:

1. Extract image file references from question text
2. Attempt to locate related images based on task ID
3. Copy found images to the images subfolder in the output directory
4. Add image references to each question in the TXT output file
5. Create an image index table in README.md for easy viewing

This is particularly useful for handling questions that include image analysis, allowing the final merged report to display questions, answers, and related images together.
