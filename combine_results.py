#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Combine correct answers from multiple output files

This script is used to merge correct answers from multiple JSONL format result files,
and save the results as new JSONL and Excel files.
"""

import argparse
import json
import os
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
import shutil
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("combine_results")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Combine correct answers from multiple result files")
    # Create mutually exclusive group, --input_files and --input_dir cannot be used simultaneously
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input_files", 
        nargs="+", 
        help="Input JSONL file paths, can specify multiple files"
    )
    input_group.add_argument(
        "--input_dir",
        type=str,
        help="Specify a directory containing JSONL files, will process all .json and .jsonl files within"
    )
    parser.add_argument(
        "--output-dir", 
        "-o", 
        type=str, 
        default="output/combined", 
        help="Output directory, default is output/combined"
    )
    parser.add_argument(
        "--output-name", 
        type=str, 
        default=f"combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}", 
        help="Output filename (without extension), default uses timestamp"
    )
    parser.add_argument(
        "--conflict-strategy", 
        type=str, 
        choices=["first", "latest", "model"], 
        default="first",
        help="Conflict resolution strategy: first=keep first correct answer, latest=keep newest answer, model=keep answer from specified model"
    )
    parser.add_argument(
        "--preferred-model", 
        type=str,
        help="When using model conflict strategy, specify the preferred model ID"
    )
    parser.add_argument(
        "--formats",
        type=str,
        nargs="+",
        choices=["jsonl", "excel", "txt", "all"],
        default=["all"],
        help="Specify output formats, can choose jsonl, excel, txt or all, default is all"
    )
    parser.add_argument(
        "--add-readme",
        action="store_true",
        help="Generate a README.md file in the output directory, explaining the merge process and results"
    )
    parser.add_argument(
        "--level",
        type=str,
        choices=["level1", "level2", "level3", "all"],
        default="all",
        help="Filter questions of a specific level, can choose level1, level2, level3 or all, default is all"
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images related to correct answers to the output directory"
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="dataset",
        help="Root directory where image files are located, default is dataset"
    )
    return parser.parse_args()

def read_jsonl_file(file_path):
    """
    Read a JSONL file and return a parsed list
    
    Parameters:
        file_path (str): JSONL file path
        
    Returns:
        list: A list containing all JSON objects from the file
    """
    logger.info(f"Reading file: {file_path}")
    results = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"Warning: Line {i+1} is not valid JSON, skipped")
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
    
    logger.info(f"Successfully read {len(results)} records")
    return results

def extract_correct_answers(results, level=None):
    """
    Extract correct answers from results
    
    Parameters:
        results (list): Results list
        level (str): Question level to filter, if None or 'all' no filtering is applied
        
    Returns:
        dict: Dictionary of correct answers with task_id as keys
    """
    correct_answers = {}
    for item in results:
        try:
            # If level filtering is needed
            if level and level != "all":
                # Extract level information from task_id
                task_id = item.get("task_id", "")
                # Try different task ID formats
                level_in_id = None
                
                # Format 1: "level1/task123"
                if "/" in task_id:
                    level_in_id = task_id.split("/")[0]
                # Format 2: "level1_task123"
                elif "_" in task_id and task_id.split("_")[0] in ["level1", "level2", "level3"]:
                    level_in_id = task_id.split("_")[0]
                # Format 3: Identify through other fields
                elif "level" in item:
                    level_in_id = item.get("level")
                    
                # Skip if not matching the required level
                if level_in_id != level:
                    continue
            
            # Only keep correct answers
            if item.get("is_correct", False):
                task_id = item.get("task_id", "unknown")
                correct_answers[task_id] = item
                
        except (KeyError, TypeError) as e:
            logger.warning(f"Error processing result: {e}")
    
    return correct_answers

def combine_results(input_files, conflict_strategy="first", preferred_model=None, level=None):
    """
    Combine correct answers from multiple files
    
    Parameters:
        input_files (list): List of input file paths
        conflict_strategy (str): Conflict resolution strategy ("first", "latest", "model")
        preferred_model (str): Preferred model ID when strategy is "model"
        level (str): Question level to filter, if None or 'all' no filtering is applied
        
    Returns:
        dict: Combined dictionary of correct answers
        dict: Statistics of correct answers contributed by each file
        int: Number of conflicts
    """
    all_correct_answers = {}
    file_stats = {}
    conflict_count = 0
    
    for file_path in input_files:
        file_name = os.path.basename(file_path)
        file_stats[file_name] = {"total": 0, "added": 0, "conflicts": 0}
        
        # Read file
        results = read_jsonl_file(file_path)
        
        # Extract correct answers
        correct_answers = extract_correct_answers(results, level)
        file_stats[file_name]["total"] = len(correct_answers)
        
        # Merge into total results
        for task_id, answer in correct_answers.items():
            if task_id not in all_correct_answers:
                # New answer, add directly
                all_correct_answers[task_id] = answer
                file_stats[file_name]["added"] += 1
            else:
                # Conflict found, handle according to strategy
                conflict_count += 1
                file_stats[file_name]["conflicts"] += 1
                
                existing_answer = all_correct_answers[task_id]
                
                if conflict_strategy == "first":
                    # Keep the first answer (default behavior)
                    logger.info(f"Conflict: Question {task_id} has correct answers in multiple files, keeping the first one")
                    continue
                
                elif conflict_strategy == "latest":
                    # Compare timestamps, keep the newest
                    existing_timestamp = existing_answer.get("timestamp", 0)
                    new_timestamp = answer.get("timestamp", 0)
                    
                    if new_timestamp > existing_timestamp:
                        logger.info(f"Conflict: Question {task_id} keeping the newer answer ({file_name})")
                        all_correct_answers[task_id] = answer
                        file_stats[file_name]["added"] += 1
                    else:
                        logger.info(f"Conflict: Question {task_id} keeping existing answer (newer)")
                
                elif conflict_strategy == "model" and preferred_model:
                    # Choose based on model ID
                    existing_model = existing_answer.get("model_id", "")
                    new_model = answer.get("model_id", "")
                    
                    if new_model == preferred_model and existing_model != preferred_model:
                        logger.info(f"Conflict: Question {task_id} keeping answer from preferred model {preferred_model}")
                        all_correct_answers[task_id] = answer
                        file_stats[file_name]["added"] += 1
                    else:
                        logger.info(f"Conflict: Question {task_id} keeping existing answer (not preferred model)")
    
    # Calculate simplified file statistics (only keep number of added entries)
    simple_file_stats = {name: stats["added"] for name, stats in file_stats.items()}
    
    return all_correct_answers, simple_file_stats, conflict_count

def save_results(combined_answers, output_dir, output_name):
    """
    Save the combined results
    
    Parameters:
        combined_answers (dict): Combined dictionary of correct answers
        output_dir (str): Output directory
        output_name (str): Output filename (without extension)
    
    Returns:
        tuple: (jsonl_path, excel_path) Paths of saved JSONL and Excel files
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare output file paths
    jsonl_path = os.path.join(output_dir, f"{output_name}.jsonl")
    excel_path = os.path.join(output_dir, f"{output_name}.xlsx")
    
    # Convert to list
    answers_list = list(combined_answers.values())
    
    # Save as JSONL
    logger.info(f"Saving JSONL file: {jsonl_path}")
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for item in answers_list:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # Save as Excel
    logger.info(f"Saving Excel file: {excel_path}")
    df = pd.DataFrame(answers_list)
    df.to_excel(excel_path, index=False)
    
    return jsonl_path, excel_path

def generate_statistics(combined_answers, file_stats, conflict_count):
    """
    Generate statistics for the merged results
    
    Parameters:
        combined_answers (dict): Combined dictionary of correct answers
        file_stats (dict): Number of correct answers contributed by each file
        conflict_count (int): Number of conflicts
        
    Returns:
        str: Formatted statistics string
    """
    stats = []
    stats.append("=" * 50)
    stats.append("Merge Results Statistics")
    stats.append("=" * 50)
    stats.append(f"Total correct answers after merging: {len(combined_answers)}")
    stats.append(f"Number of questions with conflicts: {conflict_count}")
    stats.append("\nContribution statistics by file:")
    
    for file_name, count in file_stats.items():
        stats.append(f" - {file_name}: {count} correct answers")
    
    if len(combined_answers) > 0:
        # Calculate the number of each task type
        task_types = {}
        for item in combined_answers.values():
            task = item.get("task", "Unknown")
            if task not in task_types:
                task_types[task] = 0
            task_types[task] += 1
        
        stats.append("\nStatistics by task type:")
        for task, count in sorted(task_types.items(), key=lambda x: x[1], reverse=True):
            percentage = count / len(combined_answers) * 100
            stats.append(f" - {task}: {count} items ({percentage:.2f}%)")
        
        # Statistics by model
        model_stats = {}
        for item in combined_answers.values():
            model = item.get("model_id", "Unknown")
            if model not in model_stats:
                model_stats[model] = 0
            model_stats[model] += 1
        
        if len(model_stats) > 1:  # Only display when there are multiple models
            stats.append("\nStatistics by model:")
            for model, count in sorted(model_stats.items(), key=lambda x: x[1], reverse=True):
                percentage = count / len(combined_answers) * 100
                stats.append(f" - {model}: {count} items ({percentage:.2f}%)")
    
    return "\n".join(stats)

def collect_image_files(combined_answers, images_dir):
    """
    Collect image files related to correct answers
    
    Parameters:
        combined_answers (dict): Combined dictionary of correct answers
        images_dir (str): Root directory where image files are located
        
    Returns:
        dict: Dictionary with task_id as keys and lists of image file paths as values
    """
    image_files = {}
    
    # Image file extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
    
    # Iterate through each correct answer
    for task_id, answer in combined_answers.items():
        task_images = []
        
        # Extract image references from questions
        question = answer.get("question", "")
        
        # Possible patterns for finding image paths
        image_patterns = [
            # Extract image file paths from question text
            r'(Image: \s*([^\s]+\.(?:jpg|jpeg|png|gif|bmp|tiff)))',
            r'(Image: \s*([^\s]+\.(?:jpg|jpeg|png|gif|bmp|tiff)))',
            r'(Fig\s*\d+: \s*([^\s]+\.(?:jpg|jpeg|png|gif|bmp|tiff)))',
            r'(Image\s*\d+: \s*([^\s]+\.(?:jpg|jpeg|png|gif|bmp|tiff)))',
            # Extract possible relative paths
            r'((?:\/)?(?:dataset|data|images)\/[^\s]+\.(?:jpg|jpeg|png|gif|bmp|tiff))'
        ]
        
        # Extract image paths from question
        image_paths = []
        for pattern in image_patterns:
            matches = re.findall(pattern, question, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    # If match result is a tuple, take the last element (actual file path)
                    image_path = match[-1].strip()
                else:
                    # If match result is a string, use directly
                    image_path = match.strip()
                
                # Remove quotes and other unnecessary characters
                image_path = re.sub(r'^[\'"]|[\'"]$', '', image_path)
                
                # Add to path list
                if image_path and image_path not in image_paths:
                    image_paths.append(image_path)
        
        # Extract image information from file fields
        files = answer.get("files", [])
        if isinstance(files, list):
            for file_info in files:
                if isinstance(file_info, dict):
                    file_path = file_info.get("path", "")
                    if any(file_path.lower().endswith(ext) for ext in image_extensions):
                        if file_path and file_path not in image_paths:
                            image_paths.append(file_path)
        
        # Try to find images using task_id
        if "/" in task_id:
            # Handle task_id of the form "level2/task123"
            parts = task_id.split("/")
            level = parts[0]
            task_name = parts[-1]
            
            # Try to find corresponding image files
            possible_image_locations = [
                os.path.join(images_dir, level, f"{task_name}.jpg"),
                os.path.join(images_dir, level, f"{task_name}.png"),
                os.path.join(images_dir, level, "images", f"{task_name}.jpg"),
                os.path.join(images_dir, level, "images", f"{task_name}.png")
            ]
            
            for img_path in possible_image_locations:
                if os.path.exists(img_path) and img_path not in image_paths:
                    image_paths.append(img_path)
        
        # Check if all paths exist
        for path in image_paths:
            # Handle relative paths
            if not os.path.isabs(path):
                # Try different base paths
                possible_paths = [
                    path,  # Original path
                    os.path.join(images_dir, path),  # Relative to images directory
                    # If path starts with 'dataset/', 'data/' or 'images/', try removing this prefix
                    re.sub(r'^(?:dataset|data|images)\/', '', path)
                ]
                
                for p in possible_paths:
                    if os.path.exists(p):
                        task_images.append(p)
                        break
            else:
                # Absolute path
                if os.path.exists(path):
                    task_images.append(path)
        
        # If image files were found, add to results
        if task_images:
            image_files[task_id] = task_images
    
    return image_files

def copy_images_to_output(image_files, output_dir):
    """
    Copy image files to the output directory
    
    Parameters:
        image_files (dict): Dictionary with task_id as keys and lists of image file paths as values
        output_dir (str): Output directory
        
    Returns:
        dict: Dictionary with task_id as keys and lists of copied image file paths as values
    """
    # Create image output directory
    images_output_dir = os.path.join(output_dir, "images")
    os.makedirs(images_output_dir, exist_ok=True)
    
    copied_images = {}
    
    for task_id, image_paths in image_files.items():
        task_copied_images = []
        
        for i, src_path in enumerate(image_paths):
            # Construct destination path
            file_ext = os.path.splitext(src_path)[1]
            dest_filename = f"{task_id.replace('/', '_')}_{i+1}{file_ext}"
            dest_path = os.path.join(images_output_dir, dest_filename)
            
            # Copy file
            try:
                shutil.copy2(src_path, dest_path)
                logger.info(f"Copying image file: {src_path} -> {dest_path}")
                
                # Store relative path (for reference in README and TXT)
                rel_path = os.path.join("images", dest_filename)
                task_copied_images.append(rel_path)
            except Exception as e:
                logger.error(f"Error copying image file: {e}")
        
        if task_copied_images:
            copied_images[task_id] = task_copied_images
    
    return copied_images

def save_to_txt(combined_answers, output_dir, output_name, copied_images=None):
    """
    Save merged results as TXT format
    
    Parameters:
        combined_answers (dict): Combined dictionary of correct answers
        output_dir (str): Output directory
        output_name (str): Output filename (without extension)
        copied_images (dict, optional): Dictionary with task_id as keys and lists of image file paths as values
        
    Returns:
        str: Path of the saved TXT file
    """
    output_path = os.path.join(output_dir, f"{output_name}.txt")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"Merged Results Report - Generated time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Processed {len(combined_answers)} questions\n\n")
        
        # Sort by task_id
        sorted_answers = sorted(combined_answers.items(), key=lambda x: x[0])
        
        for idx, (task_id, item) in enumerate(sorted_answers, 1):
            f.write("-" * 80 + "\n")
            f.write(f"Question {idx}: {task_id}\n")
            f.write("-" * 80 + "\n\n")
            
            # Basic information
            question = item.get("question", "No question text")
            answer_type = item.get("answer_type", "")
            data_request = item.get("data_request", "")
            our_answer = item.get("answer", "No answer")
            correct_answer = item.get("correct_answer", "")
            
            f.write(f"Question: {question}\n\n")
            
            if answer_type:
                f.write(f"Answer type: {answer_type}\n")
            if data_request:
                f.write(f"Data requirements: {data_request}\n")
            
            f.write("\n")
            
            # Display image references
            if copied_images and task_id in copied_images:
                f.write("Related images:\n")
                for img_path in copied_images[task_id]:
                    f.write(f"- {img_path}\n")
                f.write("\n")
            
            # Question answers
            f.write(f"Our answer: {our_answer}\n\n")
            
            if correct_answer:
                f.write(f"Correct answer: {correct_answer}\n\n")
            
            # Additional information
            file_name = item.get("file_name", "")
            file_type = item.get("file_type", "")
            tool = item.get("tool", "")
            model_id = item.get("model_id", "")
            
            if file_name:
                f.write(f"Filename: {file_name}\n")
            if file_type:
                f.write(f"File type: {file_type}\n")
            if tool:
                f.write(f"Tool used: {tool}\n")
            if model_id:
                f.write(f"Model ID: {model_id}\n")
            
            # Summary
            summary = item.get("summary", "")
            if summary:
                f.write("\nSummary:\n")
                f.write(summary + "\n")
            
            f.write("\n\n")
    
    return output_path

def create_readme(path, input_files, args, saved_files, statistics, answer_count, copied_images=None):
    """
    Create README.md file, documenting the merge process and results
    
    Parameters:
        path (str): README file path
        input_files (list): List of input files
        args (Namespace): Command line arguments
        saved_files (dict): Paths of saved files
        statistics (str): Statistics information
        answer_count (int): Number of correct answers
        copied_images (dict, optional): Dictionary with task_id as keys and lists of image file paths as values
    """
    with open(path, 'w', encoding='utf-8') as f:
        f.write("# Results Merge Report\n\n")
        
        # Time information
        now = datetime.now()
        f.write(f"**Generation Time**: {now.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Merge information
        f.write("## Merge Information\n\n")
        f.write(f"- Merged **{len(input_files)}** result files\n")
        f.write(f"- Total of **{answer_count}** correct answers\n")
        
        if args.level != "all":
            f.write(f"- Filtered level: **{args.level}**\n")
            
        f.write(f"- Conflict resolution strategy: **{args.conflict_strategy}**")
        if args.conflict_strategy == "model":
            f.write(f", Preferred model: **{args.preferred_model}**")
        f.write("\n\n")
        
        # Input information
        f.write("## Input Information\n\n")
        if args.input_dir:
            f.write(f"Collected **{len(input_files)}** files from directory `{args.input_dir}`\n\n")
        
        # Input file list
        f.write("### Input Files\n\n")
        for i, file_path in enumerate(input_files, 1):
            f.write(f"{i}. `{os.path.basename(file_path)}`\n")
        f.write("\n")
        
        # Output files
        f.write("## Output Files\n\n")
        for format_name, file_path in saved_files.items():
            f.write(f"- **{format_name}**: [`{os.path.basename(file_path)}`]({os.path.basename(file_path)})\n")
        f.write("\n")
        
        # Related images
        if copied_images and len(copied_images) > 0:
            f.write("## Related Images\n\n")
            f.write(f"Collected **{sum(len(imgs) for imgs in copied_images.values())}** images related to questions\n\n")
            
            # Create image index table
            f.write("### Image Index\n\n")
            f.write("| Question ID | Images |\n")
            f.write("|--------|------|\n")
            
            for task_id, img_paths in sorted(copied_images.items()):
                image_cells = []
                for img_path in img_paths:
                    # Create image thumbnail link
                    image_cells.append(f"[![{task_id}]({img_path})]({img_path})")
                
                # Add to table
                f.write(f"| {task_id} | {' '.join(image_cells)} |\n")
            
            f.write("\n")
        
        # Statistics information
        f.write("## Statistics\n\n")
        f.write("```\n")
        f.write(statistics)
        f.write("\n```\n\n")
        
        # Usage instructions
        f.write("## Usage Instructions\n\n")
        f.write("### JSONL Format\n\n")
        f.write("The JSONL format file contains complete information for each correct answer, with each line being a JSON object. Can be processed with any tool or language that supports JSON.\n\n")
        
        f.write("### Excel Format\n\n")
        f.write("The Excel file contains structured answer information, suitable for intuitive viewing and filtering of data.\n\n")
        
        f.write("### TXT Format\n\n")
        f.write("The TXT file is a human-readable format, providing a clear presentation of questions, answers, and related information.\n\n")
        
        # Command line reference
        f.write("## Command Line Reference\n\n")
        f.write("Here is the command line used to generate this result:\n\n")
        
        cmd = ["python combine_results.py"]
        if args.input_dir:
            cmd.append(f'--input_dir "{args.input_dir}"')
        else:
            cmd.append("--input_files")
            cmd.extend([f'"{file}"' for file in args.input_files])
            
        if args.output_dir != "output/combined":
            cmd.append(f'--output-dir "{args.output_dir}"')
        if not args.output_name.startswith("combined_"):
            cmd.append(f'--output-name "{args.output_name}"')
        if args.conflict_strategy != "first":
            cmd.append(f'--conflict-strategy {args.conflict_strategy}')
        if args.preferred_model:
            cmd.append(f'--preferred-model "{args.preferred_model}"')
        if args.formats != ["all"]:
            cmd.append(f'--formats {" ".join(args.formats)}')
        if args.level != "all":
            cmd.append(f'--level {args.level}')
        if args.copy_images:
            cmd.append(f'--copy-images')
            if args.images_dir != "dataset":
                cmd.append(f'--images-dir "{args.images_dir}"')
        if args.add_readme:
            cmd.append('--add-readme')
        
        f.write("```\n")
        f.write(" \\\n    ".join(cmd))
        f.write("\n```\n")

def main():
    """Main function"""
    args = parse_args()
    
    # Process input files
    input_files = []
    
    # If input directory is specified, collect all json and jsonl files from the directory
    if args.input_dir:
        if not os.path.isdir(args.input_dir):
            logger.error(f"Error: Directory {args.input_dir} does not exist")
            return
            
        logger.info(f"Collecting JSON files from directory {args.input_dir}...")
        for file_name in os.listdir(args.input_dir):
            if file_name.endswith(('.json', '.jsonl')):
                file_path = os.path.join(args.input_dir, file_name)
                input_files.append(file_path)
        
        if not input_files:
            logger.error(f"Error: No JSON or JSONL files found in directory {args.input_dir}")
            return
            
        logger.info(f"Found {len(input_files)} JSON/JSONL files")
    else:
        # Use file list provided in command line arguments
        for file_path in args.input_files:
            if os.path.exists(file_path):
                input_files.append(file_path)
            else:
                logger.error(f"Error: File {file_path} does not exist")
    
    if not input_files:
        logger.error("Error: No valid input files")
        return
    
    # Check conflict strategy
    if args.conflict_strategy == "model" and not args.preferred_model:
        logger.warning("Warning: Conflict strategy set to 'model' but no preferred model specified, falling back to 'first' strategy")
        args.conflict_strategy = "first"
    
    # Merge results
    logger.info(f"Starting to merge results from {len(input_files)} files")
    logger.info(f"Conflict resolution strategy: {args.conflict_strategy}" + 
                (f", Preferred model: {args.preferred_model}" if args.conflict_strategy == "model" else ""))
    
    if args.level != "all":
        logger.info(f"Filtering level: {args.level}")
    
    combined_answers, file_stats, conflict_count = combine_results(
        input_files, 
        conflict_strategy=args.conflict_strategy,
        preferred_model=args.preferred_model,
        level=args.level
    )
    
    # Generate statistics
    statistics = generate_statistics(combined_answers, file_stats, conflict_count)
    logger.info("\n" + statistics)
    
    # Save results
    if combined_answers:
        # Determine formats to save
        output_formats = args.formats
        if "all" in output_formats:
            output_formats = ["jsonl", "excel", "txt"]
        
        # Ensure output directory exists
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Build output name, if level is filtered, include level information in output name
        output_name = args.output_name
        if args.level != "all":
            output_name = f"{output_name}_{args.level}"
        
        # Collect and copy image files
        copied_images = None
        if args.copy_images:
            logger.info("Starting to collect images related to questions...")
            image_files = collect_image_files(combined_answers, args.images_dir)
            logger.info(f"Found {sum(len(imgs) for imgs in image_files.values())} images, distributed across {len(image_files)} questions")
            
            if image_files:
                logger.info("Starting to copy image files to output directory...")
                copied_images = copy_images_to_output(image_files, args.output_dir)
                logger.info(f"Copied {sum(len(imgs) for imgs in copied_images.values())} images")
        
        # Save files according to selected formats
        saved_files = {}
        
        if "jsonl" in output_formats:
            jsonl_path = os.path.join(args.output_dir, f"{output_name}.jsonl")
            answers_list = list(combined_answers.values())
            with open(jsonl_path, 'w', encoding='utf-8') as f:
                for item in answers_list:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            saved_files["JSONL"] = jsonl_path
            logger.info(f" - JSONL: {jsonl_path}")
        
        if "excel" in output_formats:
            excel_path = os.path.join(args.output_dir, f"{output_name}.xlsx")
            df = pd.DataFrame(list(combined_answers.values()))
            # Try to optimize column order
            columns_order = ["task_id", "question", "answer", "reasoning", "model_id", 
                            "timestamp", "is_correct", "summary"]
            # Filter existing columns
            columns = [col for col in columns_order if col in df.columns]
            # Add remaining columns
            columns.extend([col for col in df.columns if col not in columns])
            df = df[columns]
            df.to_excel(excel_path, index=False)
            saved_files["Excel"] = excel_path
            logger.info(f" - Excel: {excel_path}")
        
        if "txt" in output_formats:
            txt_path = save_to_txt(combined_answers, args.output_dir, output_name, copied_images)
            saved_files["TXT"] = txt_path
            logger.info(f" - TXT: {txt_path}")
        
        # Save statistics
        stats_path = os.path.join(args.output_dir, f"{output_name}_stats.txt")
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write(statistics)
        saved_files["Statistics"] = stats_path
        logger.info(f" - Stats: {stats_path}")
        
        # Generate README.md if needed
        if args.add_readme:
            readme_path = os.path.join(args.output_dir, "README.md")
            create_readme(readme_path, input_files, args, saved_files, statistics, len(combined_answers), copied_images)
            logger.info(f" - README: {readme_path}")
    else:
        logger.warning("Warning: No correct answers found, no output files generated")

if __name__ == "__main__":
    main() 