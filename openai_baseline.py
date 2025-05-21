import os
import argparse
import logging
import json
import time
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from openai import OpenAI
from dotenv import load_dotenv

from dataset_loader import load_custom_dataset
from smolagents import (
    LiteLLMModel,
)
from smolagents.models import MessageRole

# Load environment variables
load_dotenv(override=True)

# Global variables
SET = None
RELATIVE_EXCEL_PATH = "Historical/HLEjson/HLE.json"
EXCEL_PATH = os.path.abspath(RELATIVE_EXCEL_PATH)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--model-id", type=str, default="gpt-4o")
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--api-key", type=str, help="OpenAI API key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--output-dir", type=str, default="output_openai_baseline", help="Output directory for results")
    parser.add_argument("--level", type=str, default="level2", choices=["level1", "level2", "level3"], help="Specify which level of questions to test")
    parser.add_argument("--question-ids", type=str, help="Comma-separated list of specific question IDs to run (e.g., '16,24,35')")
    parser.add_argument("--start-id", type=int, help="Starting question ID for a range of questions to run")
    parser.add_argument("--end-id", type=int, help="Ending question ID for a range of questions to run")
    parser.add_argument("--results-json-path", type=str, default=None, help="Path to previous results JSON file for filtering already correct answers")
    return parser.parse_args()

def setup_logging(output_dir, run_name):
    """Set up logging for the application"""
    # Create logger
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)
    
    # Create console handler with formatting
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    
    # Create file handler
    log_file = os.path.join(output_dir, f"{run_name}.log")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def get_task_logger(log_dir, task_id):
    """Create a task-specific logger"""
    logger = logging.getLogger(f"task_{task_id}")
    
    # Skip if this logger already has handlers
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.INFO)
    
    # Create task-specific log file
    task_log_file = os.path.join(log_dir, f"task_{task_id}.log")
    file_handler = logging.FileHandler(task_log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    file_handler.setFormatter(formatter)
    
    # Add file handler to logger
    logger.addHandler(file_handler)
    
    return logger

def get_examples_to_answer(answers_file, eval_ds, args=None) -> List[dict]:
    """Get examples to answer, filtering out already answered questions"""
    # Get main logger
    logger = logging.getLogger("main")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    
    logger.info(f"Loading answers from {answers_file}...")
    try:
        # If file doesn't exist, exception will be raised and handled in except block
        done_questions = pd.read_json(answers_file, lines=True)["question"].tolist()
        logger.info(f"Found {len(done_questions)} previous results!")
    except Exception as e:
        logger.error(f"Error when loading records: {e}")
        logger.info("No usable records! ▶️ Starting new.")
        # Ensure file exists, even if empty
        Path(answers_file).touch()
        done_questions = []
    
    # Filter out already completed questions
    examples = [line for line in eval_ds.to_list() if line["question"] not in done_questions]
    
    # If command line arguments provided, filter by ID or range
    if args:
        filtered_examples = []
        level_prefix = f"{args.level}_"  # e.g., "level1_"
        
        # Handle specific ID list
        if args.question_ids:
            question_ids = [id.strip() for id in args.question_ids.split(',')]
            logger.info(f"Filtering specific question IDs: {question_ids}")
            
            # Convert numeric IDs to full ID format (e.g., "16" -> "level_1_16")
            full_ids = []
            # Modify level_prefix format
            level_prefix = f"level_{args.level.replace('level', '')}_"  # e.g., "level_1_"
            for id in question_ids:
                if id.startswith(level_prefix):
                    full_ids.append(id)
                else:
                    full_ids.append(f"{level_prefix}{id}")
            
            # Filter questions
            for example in examples:
                if example.get("task_id") in full_ids:
                    filtered_examples.append(example)
        
        # Handle ID range
        elif args.start_id is not None or args.end_id is not None:
            start_id = args.start_id if args.start_id is not None else 1
            end_id = args.end_id if args.end_id is not None else float('inf')
            
            logger.info(f"Filtering question ID range: {start_id} to {end_id}")
            level_prefix = f"level_{args.level.replace('level', '')}_"
            for example in examples:
                task_id = example.get("task_id", "")
                if task_id.startswith(level_prefix):
                    try:
                        # Extract numeric part
                        id_num = int(task_id[len(level_prefix):])
                        if start_id <= id_num <= end_id:
                            filtered_examples.append(example)
                    except ValueError:
                        # Skip if ID format is incorrect
                        continue
        
        # If filtering was applied, use filtered list
        if args.question_ids or args.start_id is not None or args.end_id is not None:
            logger.info(f"Number of questions after filtering: {len(filtered_examples)}/{len(examples)}")
            return filtered_examples
    
    return examples

def append_answer(entry: dict, jsonl_file: str) -> None:
    """Append a single answer entry to the results file"""
    with open(jsonl_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def analyze_results(answers_file):
    """Analyze results file, counting correct answers and calculating accuracy"""
    logger = logging.getLogger("main")
    
    # Default return value (empty results)
    default_result = {"total": 0, "correct": 0, "accuracy": 0, "by_task": {}, "by_file_type": {}}
    
    try:
        # Check if file exists
        if not os.path.exists(answers_file):
            logger.warning(f"Results file doesn't exist: {answers_file}")
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
                if not line.strip():  # Skip empty lines
                    continue
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"Warning: Skipping invalid JSON line")
        
        if not results:
            logger.warning("Results file is empty or incorrectly formatted")
            return default_result
        
        # Calculate overall accuracy
        total = len(results)
        correct = sum(1 for r in results if r.get("is_correct", False))
        accuracy = correct / total if total > 0 else 0
        
        # Group by task type
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
        
        # Group by file type
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
    """Print statistics results"""
    logger = logging.getLogger("main")
    
    logger.info("\n" + "="*50)
    logger.info("Results Statistics")
    logger.info("="*50)
    
    # Overall statistics
    logger.info(f"\nTotal questions: {stats['total']}")
    logger.info(f"Correct answers: {stats['correct']}")
    logger.info(f"Overall accuracy: {stats['accuracy']*100:.2f}%")
    
    # By task
    logger.info("\nAccuracy by task:")
    for task, data in stats["by_task"].items():
        logger.info(f"  {task}: {data['correct']}/{data['total']} - {data['accuracy']*100:.2f}%")
    
    # By file type
    logger.info("\nAccuracy by file type:")
    for file_type, data in stats["by_file_type"].items():
        logger.info(f"  {file_type}: {data['correct']}/{data['total']} - {data['accuracy']*100:.2f}%")

def check_answer_internal(model_answer, true_answer, question):
    """Use LLM to determine if the answer is correct"""
    try:
        print("Evaluating if the answer is correct...")
        # Use message format consistent with text_inspector_tool.py
        messages = [
            {
                "role": MessageRole.SYSTEM,
                "content": [
                    {
                        "type": "text",
                        "text": "As a fair evaluator, please assess whether the model's answer is semantically consistent with the correct answer."
                    }
                ],
            },
            {
                "role": MessageRole.USER,
                "content": [
                    {
                        "type": "text",
                        "text": f"""Question: {question}

Correct answer: {true_answer}

Model answer: {model_answer}

Please carefully analyze the semantic content of both answers, rather than their literal expression. Even if the expression is different, as long as the core meaning is the same, it should be considered correct.
Consider the following factors:
1. Whether the core information of the answers is consistent
2. Whether all necessary key points are included
3. Whether there are substantial errors or omissions

Please directly answer "correct" or "incorrect" without any explanation."""
                    }
                ],
            },
        ]
        model = LiteLLMModel(
        "openai/gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
        max_completion_tokens=8192,
        drop_params=True,
        )
        response = model(messages)
        # Parse the result
        result = response.content if hasattr(response, 'content') else str(response)
        
        # Determine the result
        if "correct" in result.lower() or "yes" in result.lower():
            # task_logger.info("Answer evaluation result: Correct")
            return True
        else:
            # task_logger.info("Answer evaluation result: Incorrect")
            return False
    except Exception as e:
        # task_logger.error(f"Error evaluating answer: {e}")
        # Default to incorrect when error occurs
        return False

def answer_single_question(example, model_id, answers_file, args):
    """Answer a single question using OpenAI API and save results"""
    # Get task ID as string
    task_id = str(example["task_id"])
    
    # Create task-specific logger
    LOG_DIR = Path(args.output_dir) / SET / "logs"  # Will be created in main
    task_logger = get_task_logger(LOG_DIR, task_id)
    task_logger.info(f"Starting task ID: {task_id}")
    task_logger.info(f"Question: {example['question']}")
    task_logger.info(f"Using model: {model_id}")
    
    # Initialize OpenAI client
    client = OpenAI(api_key=args.api_key)
    
    # Get question and answer type
    question = example["question"]
    answer_type = example.get("answer_type", "")
    
    # Process the question
    task_logger.info("Sending question to OpenAI API...")
    start_time = time.time()
    
    try:
        # Call OpenAI API with web search tool
        response = client.responses.create(
            model=model_id,
            input=[
                {"role": "system", "content": "You are a helpful AI assistant skilled at answering difficult historical questions. Use the web search tool to find accurate information."},
                {"role": "user", "content": question}
            ],
            tools=[{"type": "web_search_preview"}]
        )
        
        # Extract model's answer
        model_answer = response.output_text
        task_logger.info(f"Received answer: {model_answer}")
        
        # Check if the answer is correct
        true_answer = example.get("true_answer", "")
        is_correct = check_answer_internal(model_answer, true_answer, question)
        task_logger.info(f"Answer correct: {is_correct}")
        
        # Calculate processing time
        processing_time = time.time() - start_time
        task_logger.info(f"Processing time: {processing_time:.2f} seconds")
        
        # Create result entry
        entry = {
            "task_id": task_id,
            "question": question,
            "true_answer": true_answer,
            "model_answer": model_answer,
            "is_correct": is_correct,
            "processing_time": processing_time,
            "task": example.get("task", ""),
            "file_type": example.get("file_type", ""),
            "file_name": example.get("file_name", ""),
            "answer_type": answer_type,
            "model": model_id
        }
        
        # Save result
        append_answer(entry, answers_file)
        task_logger.info("Results saved successfully")
        
        return entry
    
    except Exception as e:
        task_logger.error(f"Error processing question: {e}")
        import traceback
        task_logger.error(f"Detailed error: {traceback.format_exc()}")
        
        # Create error entry
        error_entry = {
            "task_id": task_id,
            "question": question,
            "true_answer": example.get("true_answer", ""),
            "model_answer": f"ERROR: {str(e)}",
            "is_correct": False,
            "processing_time": time.time() - start_time,
            "task": example.get("task", ""),
            "file_type": example.get("file_type", ""),
            "file_name": example.get("file_name", ""),
            "answer_type": answer_type,
            "model": model_id,
            "error": str(e)
        }
        
        # Save error entry
        append_answer(error_entry, answers_file)
        task_logger.info("Error results saved")
        
        return error_entry

def main():
    """Run the main program."""
    # Parse arguments
    args = parse_args()
    
    # Set global SET variable based on level
    global SET
    SET = f"{args.level}_summary"
    
    # Convert args.level to dataset_loader.py format (e.g., "level2" to "level 2")
    sheet_name = args.level.replace("level", "level ")
    
    # Create output directory
    output_dir = Path(args.output_dir) / SET
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create logs directory
    global LOG_DIR
    LOG_DIR = output_dir / "logs"
    LOG_DIR.mkdir(exist_ok=True, parents=True)
    
    # Setup logging
    logger = setup_logging(LOG_DIR, args.run_name)
    
    # Log level information
    logger.info(f"Testing questions from level: {args.level}")
    
    # Start time for the entire run
    start_time = time.time()
    
    # Log start info
    logger.info(f"Starting run with arguments: {args}")
    
    # Load custom Excel dataset
    eval_ds = load_custom_dataset(EXCEL_PATH, test_mode=False, results_json_path=args.results_json_path, sheet_name=sheet_name)
    
    # Define output file paths
    answers_file = f"{output_dir}/{args.run_name}.jsonl"
    txt_file = answers_file.replace(".jsonl", ".txt")
    
    # Check if results file already exists, analyze if it does
    if os.path.exists(answers_file) and os.path.getsize(answers_file) > 0:
        logging.info(f"Detected existing results file: {answers_file}")
        stats = analyze_results(answers_file)
        print_statistics(stats)
        
        # Ask whether to continue running
        response = input("Continue running the test? (y/n): ")
        if response.lower() != 'y':
            logging.info("User chose to exit")
            return
        
        # If continuing, append to TXT file rather than overwriting
        with open(txt_file, "a", encoding="utf-8") as f:
            f.write(f"\n\nContinuing test run: {args.run_name}\n")
            f.write(f"Model: {args.model_id}\n")
            f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    else:
        # If new run, create new TXT file
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write(f"Test run: {args.run_name}\n")
            f.write(f"Model: {args.model_id}\n")
            f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    # Get tasks to run
    tasks_to_run = get_examples_to_answer(answers_file, eval_ds, args)
    
    # Process tasks with thread pool for concurrency
    with ThreadPoolExecutor(max_workers=args.concurrency) as exe:
        futures = [
            exe.submit(answer_single_question, example, args.model_id, answers_file, args)
            for example in tasks_to_run
        ]
        for f in tqdm(as_completed(futures), total=len(tasks_to_run), desc="Processing tasks"):
            f.result()
    
    logging.info("All tasks completed.")
    
    # Final statistics summary
    logging.info("\nFinal statistics summary...")
    stats = analyze_results(answers_file)
    print_statistics(stats)
    
    logging.info(f"Results saved to {answers_file}")
    logging.info(f"Total runtime: {(time.time() - start_time) / 60:.2f} minutes")

if __name__ == "__main__":
    main()