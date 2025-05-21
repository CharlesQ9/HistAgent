# HistAgent

This repository contains all necessary components to reproduce our system for historical question answering with multi-agent orchestration. The system integrates retrieval, OCR, image search, file parsing, and academic literature access.

---

## 🛠️ Installation

### Step 1: Create a Conda Environment

```bash
conda create -n HistAgent python=3.12
conda activate HistAgent
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
pip install smolagents[dev]
```

---

## ⚙️ Environment Configuration

This project relies on several API keys. These can be configured either via a `.env` file or by setting environment variables directly in your terminal.

### Option 1: Using a `.env` File

Create a file named `.env` in the root directory with the following contents:

```
SERPAPI_API_KEY=your_serpapi_api_key
IMGBB_API_KEY=your_imgbb_api_key
OPENAI_API_KEY=your_openai_api_key
HF_TOKEN=your_huggingface_token
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_API_BASE=your_openrouter_api_base
SEARCH_API_KEY=your_search_api_key
SPRINGER_API_KEY=your_springer_api_key
LLAMA_API_KEY=your_llama_api_key
TRANSKRIBUS_USERNAME=your_transkribus_username
TRANSKRIBUS_PASSWORD=your_transkribus_password
```

### Option 2: Export from Terminal

```bash
export SERPAPI_API_KEY="your_key"
export IMGBB_API_KEY="your_key"
...
```

### Option 3: Pass via Command Line

```bash
python run_hist.py --run-name "test_run" --api-key "your_openai_key" --springer-api-key "your_springer_key" --llama-api-key "your_llama_key"
```

---

## 📦 Replace `browser_use` Library

Our system uses a modified version of the `browser_use` library. You must replace the default installation:

```bash
python -c "import site; print(site.getsitepackages()[0])"
cp -r browser_use /path/to/site-packages/
```

---

## 🧠 HistAgent: Running the System

Basic command to launch the system:

```bash
python run_hist.py --run-name "test_run" --use-image-agent --use-file-agent
```

### Available Arguments

| Flag                                    | Description                                        |
| --------------------------------------- | -------------------------------------------------- |
| `--run-name`                            | Required. Name of the run.                         |
| `--model-id`                            | Language model (default: `gpt-4o`)                 |
| `--concurrency`                         | Number of parallel tasks (default: 8)              |
| `--use-image-agent`                     | Enables the image processing agent                 |
| `--use-file-agent`                      | Enables file processing capabilities               |
| `--use-literature-agent`                | Enables academic literature retrieval              |
| `--no-agent-webbrowser-agent`           | Disables the web search agent                      |
| `--use-springer` / `--no-springer`      | Toggle Springer integration                        |
| `--use-browser`                         | Enables browser-based literature tools             |
| `--baseline`                            | Use baseline agent architecture                    |
| `--level`                               | Select task level: `level1`, `level2`, or `level3` |
| `--results-json-path`                   | Load prior run for filtering                       |
| `--question-ids`                        | Run specific IDs (e.g., `12,15,19`)                |
| `--start-id`, `--end-id`                | Range of question IDs                              |
| `--springer-api-key`, `--llama-api-key` | API keys (optional override)                       |

---

## 🧪 Example Commands

Standard agent hierarchy (level2):

```bash
python run_hist.py --run-name "test_all" --use-image-agent --use-file-agent
```

Baseline for level1:

```bash
python run_hist.py --run-name "baseline_test" --baseline --level level1
```

Full system with literature search and browser (level3):

```bash
python run_hist.py --run-name "full_research" --use-image-agent --use-file-agent --use-literature-agent --use-browser --level level3
```

Specific questions:

```bash
python run_hist.py --run-name "questions_subset" --question-ids "5,8,20"
```

---

## 📂 JSON Dataset Support

To run with a custom dataset in HLE.json format:

```bash
python run_hlejson.py --run-name "json_dataset_test" --use-image-agent --use-file-agent
```

All command line arguments supported in `run_hist.py` also apply to `run_hlejson.py`.

---

## 📊 Output Files

Results are stored in either `output/{LEVEL}_summary/` or `output_baseline/{LEVEL}_summary/`, including:

* `.jsonl`: Detailed execution records
* `.xlsx`: Spreadsheet for further analysis
* `.txt`: Readable summary
* Statistics file
* Log files

Logs are organized by question and run:

* `main.log`: Overview
* `task_{id}.log`: Detailed trace
* `errors.log`: Captures all exceptions

---

## 🧩 Agent Descriptions

| Agent                 | Functionality                             |
| --------------------- | ----------------------------------------- |
| **Image Agent**       | OCR, reverse search, image classification |
| **File Agent**        | PDF, Word, Excel parsing                  |
| **Web Browser Agent** | Search and extraction from web pages      |
| **Literature Agent**  | Retrieves scholarly articles via:         |

* `literature_searching_task`
* `springer_search`
* `general_browser_task`
* Browser-based tools (if enabled) |
  \| **Baseline Agent** | Simple pipeline without hierarchy |

---

## 🔍 Academic Literature Tools

The system is tailored for historical academic research:

* **Smart Retrieval**: Cites impactful papers from Google Scholar
* **Filter by Relevance**: Citation- and content-aware filtering
* **Full-text Analysis**: Extractive reading from structured and scanned PDFs
* **Springer API**: Direct access to open-access scholarly papers

Use `--no-springer` to disable Springer tools.

---

## 🔗 Results Combination Tool

Use `combine_results.py` to merge results from different models or runs.

### Basic Usage

```bash
python combine_results.py output/*.jsonl --output-dir combined --output-name all_results
```

### Arguments

| Flag                  | Description                                               |
| --------------------- | --------------------------------------------------------- |
| `input_files`         | Multiple `.jsonl` files                                   |
| `--output-dir`        | Output directory (default: `output/combined`)             |
| `--output-name`       | Custom name (default: timestamp)                          |
| `--conflict-strategy` | Choose from: `first`, `latest`, `model`                   |
| `--preferred-model`   | Used with `--conflict-strategy model`                     |
| `--formats`           | Choose from: `jsonl`, `excel`, `txt`, `all`               |
| `--add-readme`        | Adds summary README to output folder                      |
| `--level`             | Filter by task level: `level1`, `level2`, `level3`, `all` |
| `--copy-images`       | Copy related images                                       |
| `--images-dir`        | Image root (default: `dataset/`)                          |

---
