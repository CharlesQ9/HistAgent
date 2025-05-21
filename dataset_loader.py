import os
import pandas as pd
import json
from datasets import Dataset
from typing import Optional, List

def load_custom_dataset(
    file_path: str,
    files_dir: Optional[str] = "Historical/HistBench",
    sheet_name: Optional[str] = 'level 2',
    test_mode: bool = False,
    results_json_path: Optional[str] = None
) -> Dataset:
    """
    Load custom dataset in JSON or Excel format, handling various types of Data Requirements
    
    Parameters:
        file_path: Path to JSON or Excel file
        files_dir: Directory containing related files
        sheet_name: Name of worksheet or JSON key to load, None means load all sheets/keys
        test_mode: Whether to load only the first three records for testing
        results_json_path: Path to previous results JSON file, if provided, will filter dataset based on correct answers
        
    Returns:
        Dataset: Dataset object compatible with GAIA dataset format
    """
    # Ensure directory exists
    os.makedirs(files_dir, exist_ok=True)
    
    # Check file type (JSON or Excel)
    file_ext = os.path.splitext(file_path)[1].lower()
    is_json = file_ext == '.json'
    is_excel = file_ext in ['.xlsx', '.xls', '.xlsm']
    
    if is_json:
        sheet_name = None
    
    if not (is_json or is_excel):
        print(f"Unsupported file type: {file_ext}, only JSON and Excel are supported")
        return Dataset.from_pandas(pd.DataFrame({
            "task_id": ["1"],
            "question": ["Example question"],
            "true_answer": ["Example answer"],
            "task": ["Example task"],
            "file_name": [""],
            "file_type": [""],
            "file_tool": [""],
            "answer_type": [""]
        }))
    
    try:
        if is_json:
            # JSON loading logic
            print(f"Loading JSON file: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Read all keys or specified key
            if sheet_name is None:
                # Get top-level keys (equivalent to Excel sheets)
                sheet_keys = list(data.keys())
                print(f"Detected {len(sheet_keys)} keys: {sheet_keys}")
                
                # Read data from all keys
                all_data = []
                for key in sheet_keys:
                    print(f"Reading key: {key}")
                    
                    # Assume each key corresponds to an array containing a list of questions
                    if not isinstance(data[key], list):
                        print(f"Key {key} is not in list format, skipping")
                        continue
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(data[key])
                    
                    # Check if data is empty
                    if df.empty:
                        print(f"Data for key {key} is empty, skipping")
                        continue
                    
                    # Check for necessary columns - for JSON files, Data Requirements is not required
                    if is_json:
                        required_columns = ["ID", "Question", "Answer", "Answer Type"]
                        # Check if Data Requirements exists, if not, will look for {id}.png later
                        has_data_req = "Data Requirements" in df.columns
                        print(f"JSON data {'has' if has_data_req else 'does not have'} Data Requirements field")
                    else:
                        required_columns = ["ID", "Question", "Answer", "Data Requirements", "Answer Type"]
                    
                    missing_columns = [col for col in required_columns if col not in df.columns]
                    if missing_columns:
                        print(f"Key {key} is missing required columns: {missing_columns}, trying alternative column names")
                        
                        # Try mapping column names
                        column_mapping = {
                            "ID": ["ID", "id", "number", "sequence"],
                            "Question": ["Question", "question", "problem"],
                            "Answer": ["Answer", "answer", "solution"],
                            "Data Requirements": ["Data Requirements", "data requirements", "file", "attachment", "dataRequirements"],
                            "Answer Type": ["Answer Type", "answer type", "answer_type", "type", "answerType"]
                        }
                        
                        for req_col, possible_names in column_mapping.items():
                            # For JSON files, if Data Requirements is missing, don't try to map it
                            if is_json and req_col == "Data Requirements" and not has_data_req:
                                continue
                                
                            for col_name in df.columns:
                                if col_name in possible_names or any(name.lower() in col_name.lower() for name in possible_names):
                                    df = df.rename(columns={col_name: req_col})
                                    print(f"Mapped column '{col_name}' to '{req_col}'")
                                    break
                    
                    # Check required columns again
                    missing_columns = [col for col in required_columns if col not in df.columns]
                    if missing_columns:
                        print(f"Key {key} still missing required columns: {missing_columns}, skipping this key")
                        continue
                    
                    # Add key name as task name
                    df["task"] = key
                    
                    # For JSON files, if no Data Requirements column, check if there's a corresponding ID.png file
                    if is_json and not has_data_req and "ID" in df.columns:
                        # Get directory containing the JSON file
                        json_dir = os.path.dirname(file_path)
                        
                        # Add file paths for records missing Data Requirements
                        for idx, row in df.iterrows():
                            id_val = row["ID"]
                            if pd.isna(id_val) or id_val is None or id_val == "":
                                continue
                                
                            # Try to find {id}.png file
                            img_filename = f"{id_val}.png"
                            img_path = os.path.join(json_dir, img_filename)
                            
                            if os.path.exists(img_path):
                                print(f"Found image for ID {id_val}: {img_path}")
                                # If no Data Requirements column, add one
                                if "Data Requirements" not in df.columns:
                                    df["Data Requirements"] = ""
                                # Set file path for this ID
                                df.at[idx, "Data Requirements"] = img_path
                            else:
                                print(f"No image found for ID {id_val}: {img_path}")
                        
                        # Check if any files were added
                        if "Data Requirements" in df.columns and df["Data Requirements"].any():
                            print(f"Found {df['Data Requirements'].notna().sum()} image files based on ID")
                        else:
                            print("No image files found for any ID")
                    
                    # Add to total data
                    all_data.append(df)
                
                if not all_data:
                    print("All keys fail to meet requirements, returning empty dataset")
                    return Dataset.from_pandas(pd.DataFrame({
                        "task_id": ["1"],
                        "question": ["Example question"],
                        "true_answer": ["Example answer"],
                        "task": ["Example task"],
                        "file_name": [""],
                        "file_type": [""],
                        "file_tool": [""],
                        "answer_type": [""]
                    }))
                
                # Merge all data
                df = pd.concat(all_data, ignore_index=True)
            else:
                # Read specified key
                if sheet_name not in data:
                    print(f"Specified key {sheet_name} does not exist, returning empty dataset")
                    return Dataset.from_pandas(pd.DataFrame({
                        "task_id": ["1"],
                        "question": ["Example question"],
                        "true_answer": ["Example answer"],
                        "task": ["Example task"],
                        "file_name": [""],
                        "file_type": [""],
                        "file_tool": [""],
                        "answer_type": [""]
                    }))
                
                # Convert to DataFrame
                if isinstance(data[sheet_name], list):
                    df = pd.DataFrame(data[sheet_name])
                else:
                    print(f"Key {sheet_name} is not in list format, trying to convert")
                    if isinstance(data[sheet_name], dict):
                        # If it's a dictionary, try to convert to list
                        df = pd.DataFrame([data[sheet_name]])
                    else:
                        print(f"Cannot process data for key {sheet_name}, returning empty dataset")
                        return Dataset.from_pandas(pd.DataFrame({
                            "task_id": ["1"],
                            "question": ["Example question"],
                            "true_answer": ["Example answer"],
                            "task": ["Example task"],
                            "file_name": [""],
                            "file_type": [""],
                            "file_tool": [""],
                            "answer_type": [""]
                        }))
                
                df["task"] = str(sheet_name)  # Use key name as task name
                
                # For JSON files, if no Data Requirements column, check if there's a corresponding ID.png file
                if is_json and "ID" in df.columns and "Data Requirements" not in df.columns:
                    # Get directory containing the JSON file
                    json_dir = os.path.dirname(file_path)
                    
                    # Add file paths for records missing Data Requirements
                    for idx, row in df.iterrows():
                        id_val = row["ID"]
                        if pd.isna(id_val) or id_val is None or id_val == "":
                            continue
                            
                        # Try to find {id}.png file
                        img_filename = f"{id_val}.png"
                        img_path = os.path.join(json_dir, img_filename)
                        
                        if os.path.exists(img_path):
                            print(f"Found image for ID {id_val}: {img_path}")
                            # If no Data Requirements column, add one
                            if "Data Requirements" not in df.columns:
                                df["Data Requirements"] = ""
                            # Set file path for this ID
                            df.at[idx, "Data Requirements"] = img_path
                        else:
                            print(f"No image found for ID {id_val}: {img_path}")
                    
                    # Check if any files were added
                    if "Data Requirements" in df.columns and df["Data Requirements"].any():
                        print(f"Found {df['Data Requirements'].notna().sum()} image files based on ID")
                    else:
                        print("No image files found for any ID")
        else:
            # Excel loading logic
            print(f"Loading Excel file: {file_path}")
            if sheet_name is None:
                # First get all sheet names
                xls = pd.ExcelFile(file_path)
                sheet_names = xls.sheet_names
                print(f"Detected {len(sheet_names)} sheets: {sheet_names}")
                
                # Read data from all sheets
                all_data = []
                for sheet in sheet_names:
                    print(f"Reading sheet: {sheet}")
                    df = pd.read_excel(file_path, sheet_name=sheet)
                    
                    # Check if sheet is empty
                    if df.empty:
                        print(f"Sheet {sheet} is empty, skipping")
                        continue
                    
                    # Check if sheet has necessary columns
                    required_columns = ["ID", "Question", "Answer", "Data Requirements", "Answer Type"]
                    missing_columns = [col for col in required_columns if col not in df.columns]
                    if missing_columns:
                        print(f"Sheet {sheet} is missing required columns: {missing_columns}, trying alternative column names")
                        
                        # Try mapping column names
                        column_mapping = {
                            "ID": ["ID", "id", "number", "sequence"],
                            "Question": ["Question", "question", "problem"],
                            "Answer": ["Answer", "answer", "solution"],
                            "Data Requirements": ["Data Requirements", "data requirements", "file", "attachment"],
                            "Answer Type": ["Answer Type", "answer type", "type"]
                        }
                        
                        for req_col, possible_names in column_mapping.items():
                            for col_name in df.columns:
                                if col_name in possible_names or any(name.lower() in col_name.lower() for name in possible_names):
                                    df = df.rename(columns={col_name: req_col})
                                    print(f"Mapped column '{col_name}' to '{req_col}'")
                                    break
                    
                    # Check required columns again
                    missing_columns = [col for col in required_columns if col not in df.columns]
                    if missing_columns:
                        print(f"Sheet {sheet} still missing required columns: {missing_columns}, skipping this sheet")
                        continue
                    
                    # Add sheet name as task name
                    df["task"] = sheet
                    
                    # Add to total data
                    all_data.append(df)
                
                if not all_data:
                    print("All sheets fail to meet requirements, returning empty dataset")
                    return Dataset.from_pandas(pd.DataFrame({
                        "task_id": ["1"],
                        "question": ["Example question"],
                        "true_answer": ["Example answer"],
                        "task": ["Example task"],
                        "file_name": [""],
                        "file_type": [""],
                        "file_tool": [""],
                        "answer_type": [""]
                    }))
                
                # Merge all sheet data
                df = pd.concat(all_data, ignore_index=True)
            else:
                # Read specified sheet
                try:
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                    df["task"] = str(sheet_name)  # Use sheet name as task name
                except Exception as e:
                    print(f"Error reading sheet {sheet_name}: {e}")
                    return Dataset.from_pandas(pd.DataFrame({
                        "task_id": ["1"],
                        "question": ["Example question"],
                        "true_answer": ["Example answer"],
                        "task": ["Example task"],
                        "file_name": [""],
                        "file_type": [""],
                        "file_tool": [""],
                        "answer_type": [""]
                    }))
    except Exception as e:
        print(f"Error loading file: {e}")
        return Dataset.from_pandas(pd.DataFrame({
            "task_id": ["1"],
            "question": ["Example question"],
            "true_answer": ["Example answer"],
            "task": ["Example task"],
            "file_name": [""],
            "file_type": [""],
            "file_tool": [""],
            "answer_type": [""]
        }))
    
    # Define column mapping
    column_mapping = {
        "ID": "task_id",
        "Question": "question",
        "Answer": "true_answer",
        "Data Requirements": "data_requirement",  # Renamed to data_requirement for accuracy
        "Answer Type": "answer_type"
    }
    
    # Apply column renaming
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df = df.rename(columns={old_col: new_col})
    
    # Add missing columns
    for col in ["task_id", "question", "true_answer", "task", "data_requirement", "answer_type", 
                "file_name", "file_type", "file_tool", "data_type"]:  # Added data_type field
        if col not in df.columns:
            df[col] = ""
    
    # Add file type columns
    df["file_type"] = ""
    df["file_tool"] = ""
    
    # Process Data Requirements
    def process_data_requirement(row):
        """Process Data Requirements field, determine its type and appropriate handling method"""
        # Get data_requirement value, ensure it's a string
        data_req = row.get("data_requirement", "")
        
        # Check if it's NaN or None, convert to empty string
        if pd.isna(data_req) or data_req is None:
            data_req = ""
        else:
            # Ensure it's a string type
            data_req = str(data_req).strip()
        
        # If it's an empty string, set type to none
        if not data_req:
            row["data_type"] = "none"
            return row
            
        # Check if it contains multiple file paths (separated by semicolons)
        if ";" in data_req:
            # Correctly split paths and filter empty strings
            file_paths = [path.strip() for path in data_req.split(";") if path.strip()]
            print(f"Detected multiple file paths: {file_paths}")
            
            # Initialize file lists
            valid_files = []
            missing_files = []
            
            # Process all file paths
            for path in file_paths:
                full_path = os.path.join(files_dir, path)
                # Check if file exists
                if os.path.exists(full_path):
                    valid_files.append(full_path)
                else:
                    # If file doesn't exist but has valid extension, consider it valid
                    if path.lower().endswith(('.pdf', '.docx', '.xlsx', '.jpg', '.jpeg', '.png', '.gif', '.mp3', '.wav', '.zip')):
                        # Ensure absolute path is used
                        valid_files.append(full_path)
                        print(f"File {path} doesn't exist but has valid extension, marking as valid")
                    else:
                        missing_files.append(path)
            
            # If there are valid files, set type to file
            if valid_files:
                # Save main file (first valid file) for compatibility with old code
                row["file_name"] = valid_files[0]
                # Save all valid file paths to file_names field - this is a list
                row["file_names"] = valid_files
                row["data_type"] = "file"
                
                print(f"Number of valid files: {len(valid_files)}")
                print(f"Main file: {row['file_name']}")
                if len(valid_files) > 5:
                    print(f"All valid files: Too many files, showing only first 5 {valid_files[:5]} ...")
                else:
                    print(f"All valid files: {valid_files}")
                    
                if missing_files:
                    print(f"Warning: The following files don't exist: {missing_files}")
            else:
                print(f"Warning: No valid files: {data_req}")
                row["data_type"] = "unknown"
                row["file_name"] = ""
                row["file_names"] = []
            
            return row
            
        # Check if it's a single file path
        file_path = os.path.join(files_dir, data_req)
        if os.path.exists(file_path) or data_req.lower().endswith(('.pdf', '.docx', '.xlsx', '.jpg', '.jpeg', '.png', '.gif', '.mp3', '.wav', '.zip')):
            # Set single file
            row["file_name"] = file_path
            # Also set file_names list for uniform handling
            row["file_names"] = [file_path]
            row["data_type"] = "file"
            print(f"Detected single file: {file_path}")
        
        # Check if it's foreign language text (contains non-ASCII characters and is not a file path)
        elif any(ord(c) > 127 for c in data_req) and not os.path.exists(file_path):
            row["data_type"] = "foreign_text"
            print(f"Detected foreign language text: {data_req[:50]}...")
        
        # Check if it's a book title or information to search for
        elif any(keyword in data_req.lower() for keyword in ["book", "novel", "article", "paper", "search", "find", "look up"]):
            row["data_type"] = "search_query"
            print(f"Detected search query: {data_req}")
        
        # Other cases, might be plain text or instructions
        else:
            row["data_type"] = "text"
            print(f"Detected plain text: {data_req[:50]}...")
            
        return row
    
    # Apply data requirement processing
    print("Processing Data Requirements...")
    df = df.apply(process_data_requirement, axis=1)

    
    # Filter dataset based on previous results
    if results_json_path and os.path.exists(results_json_path):
        print(f"Detected results JSON file: {results_json_path}")
        try:
            # Read results JSON file
            with open(results_json_path, 'r', encoding='utf-8') as f:
                results_data = []
                for line in f:
                    try:
                        results_data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            
            # Extract task_ids of correctly answered questions
            correct_task_ids = [item['task_id'] for item in results_data if item.get('is_correct', "true")]
            
            if correct_task_ids:
                # Filter out already correctly answered questions
                original_count = len(df)
                df = df[~df['task_id'].astype(str).isin([str(task_id) for task_id in correct_task_ids])]
                filtered_count = original_count - len(df)
                
                print(f"Filtering based on results JSON: Number of correctly answered questions: {len(correct_task_ids)}")
                print(f"Number of questions removed from dataset: {filtered_count}")
                print(f"Size of filtered dataset: {len(df)}")
        except Exception as e:
            print(f"Error processing results JSON: {e}")
            
            
    # If in test mode, keep only the first three records
    if test_mode and len(df) > 3:
        print(f"Test mode: Keeping only the first 3 records (out of {len(df)})")
        df = df.iloc[[31]]  
    
    # # Ensure all columns have correct data types (convert to string)
    # for col in df.columns:
    #     df[col] = df[col].fillna("").astype(str)
    #     print(f"Column '{col}' converted to string type")

    # Ensure all columns have correct data types (convert to string)
    file_names_columns = {}  # Save original file_names list data
    
    for i, row in df.iterrows():
        if "file_names" in row and isinstance(row["file_names"], list):
            file_names_columns[i] = row["file_names"]
    
    # Convert other columns to string
    for col in df.columns:
        if col != "file_names":  # Skip file_names column
            df[col] = df[col].fillna("").astype(str)
    
    # Restore file_names list data
    for i, file_names in file_names_columns.items():
        df.at[i, "file_names"] = file_names
    
    # Convert to Dataset object
    dataset = Dataset.from_pandas(df)
    print(f"Dataset loading complete, {len(dataset)} records total")
    
    return dataset

def load_json_dataset(
    json_path: str,
    files_dir: Optional[str] = "Historical_js/Historical_js"
) -> Dataset:
    """
    Load JSON format dataset, automatically handling various file types
    
    Parameters:
        json_path: Path to JSON file
        files_dir: Directory containing related files
        
    Returns:
        Dataset: Dataset object compatible with GAIA dataset format
    """
    # Ensure directory exists
    os.makedirs(files_dir, exist_ok=True)
    
    # Load JSON file
    print(f"Loading JSON file: {json_path}")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return Dataset.from_pandas(pd.DataFrame({
            "task_id": ["1"],
            "question": ["Example question"],
            "true_answer": ["Example answer"],
            "task": ["Example task"],
            "file_name": [""],
            "file_type": [""],
            "file_tool": [""]
        }))
    
    # Extract all questions
    all_questions = []
    
    # Process JSON structure - assuming structure similar to provided example
    for level, questions in data.items():
        if isinstance(questions, list):
            for q in questions:
                # Map fields
                question_data = {
                    "task_id": q.get("id", ""),
                    "question": q.get("question", ""),
                    "true_answer": q.get("answer", ""),
                    "file_name": q.get("data_requirement", ""),
                    "file_type": "",
                    "file_tool": ""
                }
                all_questions.append(question_data)
    
    # If no questions found, display warning
    if not all_questions:
        print(f"Warning: No questions found in JSON file. Please check if JSON structure is correct.")
        print(f"JSON structure: {list(data.keys())}")
        # Return an empty dataset
        return Dataset.from_pandas(pd.DataFrame({
            "task_id": ["1"],
            "question": ["Example question"],
            "true_answer": ["Example answer"],
            "task": ["Example task"],
            "file_name": [""],
            "file_type": [""],
            "file_tool": [""]
        }))
    
    # Create DataFrame
    df = pd.DataFrame(all_questions)
    
    # Process file paths and add file type column
    def process_file_info(row):
        """Process file path and detect file type"""
        if not row.get("file_name") or not isinstance(row["file_name"], str):
            row["file_name"] = ""
            return row
            
        file_path = row["file_name"].strip()
        
        # Remove potentially illegal characters
        clean_path = file_path.replace("\n", "").replace("\r", "")
        
        # Avoid adding directory prefix multiple times
        if clean_path and not clean_path.startswith(files_dir):
            row["file_name"] = os.path.join(files_dir, clean_path)
        else:
            row["file_name"] = clean_path  # If already contains prefix, don't add again
        
        # Detect file type
        if row["file_name"] and os.path.exists(row["file_name"]):
            file_type, tool_name = detect_file_type(row["file_name"])
            row["file_type"] = file_type
            if isinstance(tool_name, list):
                # If there are multiple possible tools, use the first one by default
                row["file_tool"] = tool_name[0]
            elif tool_name:
                row["file_tool"] = tool_name
            
            print(f"Detected file type: {row['file_name']} -> {file_type}, Tool: {row['file_tool']}")
        else:
            if row["file_name"]:
                print(f"Warning: File doesn't exist: {row['file_name']}")
            row["file_type"] = "unknown"
            row["file_tool"] = ""
            
        return row
    
    # Apply file processing
    print("Processing file information...")
    df = df.apply(process_file_info, axis=1)
    
    # Ensure all columns have correct data types (convert to string)
    for col in df.columns:
        df[col] = df[col].fillna("").astype(str)
        print(f"Column '{col}' converted to string type")
    
    # Convert to Dataset object
    try:
        dataset = Dataset.from_pandas(df)
        print(f"Successfully loaded JSON dataset: {len(dataset)} records")
        
        # Print first few records of dataset
        print("First 3 records of dataset:")
        for i in range(min(3, len(dataset))):
            print(f"Record {i+1}:")
            try:
                record = dataset[i]
                if hasattr(record, 'items'):
                    for k, v in record.items():
                        if k == "file_name" and v:
                            try:
                                file_exists = os.path.exists(v)
                            except:
                                file_exists = False
                            print(f"  {k}: {v} (File exists: {file_exists})")
                        else:
                            print(f"  {k}: {v}")
                else:
                    print(f"  Record type error: {type(record)}")
            except Exception as e:
                print(f"  Cannot display record {i+1}: {e}")
        
        return dataset
    except Exception as e:
        print(f"Error converting to Dataset: {e}")
        # Try to diagnose the problem
        print("Attempting to diagnose the problem...")
        print(f"DataFrame information:")
        print(f"- Rows: {len(df)}")
        print(f"- Columns: {len(df.columns)}")
        print(f"- Column names: {df.columns.tolist()}")
        print(f"- Data types: {df.dtypes}")
        
        # Return a simple dataset as fallback
        print("Creating an empty fallback dataset...")
        empty_df = pd.DataFrame({
            "task_id": ["1"],
            "question": ["Example question"],
            "true_answer": ["Example answer"],
            "task": ["Example task"],
            "file_name": [""],
            "file_type": [""],
            "file_tool": [""]
        })
        return Dataset.from_pandas(empty_df)

def detect_file_type(file_path):
    """
    Automatically detect file type and recommend appropriate processing tool
    
    Args:
        file_path (str): File path
        
    Returns:
        tuple: (file_type, tool_name) File type and recommended tool
    """
    if not os.path.exists(file_path):
        return "unknown", None
        
    # Extract file extension
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    # Type mapping based on extension
    image_exts = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp']
    audio_exts = ['.wav', '.mp3', '.m4a', '.ogg', '.flac']
    video_exts = ['.mp4', '.avi', '.mov', '.wmv', '.mkv']
    document_exts = {
        '.pdf': ("pdf", "PDF_Tool"),
        '.docx': ("docx", "DOCX_Tool"),
        '.doc': ("doc", "DOCX_Tool"),  # Use DOCX tool for processing
        '.xlsx': ("xlsx", "XLSX_Tool"),
        '.xls': ("xls", "XLSX_Tool"),  # Use XLSX tool for processing
        '.pptx': ("pptx", "PPTX_Tool"),
        '.ppt': ("ppt", "PPTX_Tool"),  # Use PPTX tool for processing
        '.txt': ("text", "Text_Inspector_Tool"),
        '.csv': ("csv", "XLSX_Tool"),  # Use XLSX tool for processing
        '.json': ("json", "Text_Inspector_Tool"),
        '.xml': ("xml", "Text_Inspector_Tool"),
        '.html': ("html", "Text_Inspector_Tool")
    }
    
    # Image file processing - can use OCR or image analysis
    if ext in image_exts:
        return "image", ["Image_Analysis_Tool", "Text_Detector_Tool"]
    
    # Audio file processing
    elif ext in audio_exts:
        return "audio", "Speech_Recognition_Tool"
    
    # Video file processing - currently no dedicated video tool, but can extract key frames
    elif ext in video_exts:
        return "video", None  # Video processing not directly supported yet
    
    # Document file processing
    elif ext in document_exts:
        return document_exts[ext]
    
    # Try to detect type through file content
    else:
        try:
            # Read file header to detect file type
            with open(file_path, 'rb') as f:
                header = f.read(20)  # Read first 20 bytes
                
                # PDF signature detection
                if header.startswith(b'%PDF'):
                    return "pdf", "PDF_Tool"
                
                # Image signature detection
                if (header.startswith(b'\xff\xd8\xff') or  # JPEG
                    header.startswith(b'\x89PNG\r\n\x1a\n') or  # PNG
                    header.startswith(b'GIF8') or  # GIF
                    header.startswith(b'BM')):  # BMP
                    return "image", ["Image_Analysis_Tool", "Text_Detector_Tool"]
                
                # Office document signature detection
                if header.startswith(b'PK\x03\x04'):  # ZIP format, could be Office document
                    if ext == '.docx' or 'word' in file_path.lower():
                        return "docx", "DOCX_Tool"
                    elif ext == '.xlsx' or 'excel' in file_path.lower():
                        return "xlsx", "XLSX_Tool"
                    elif ext == '.pptx' or 'powerpoint' in file_path.lower():
                        return "pptx", "PPTX_Tool"
                    else:
                        return "archive", None  # ZIP or other archive file
                
        except Exception as e:
            print(f"Error detecting file type: {e}")
        
        # If type cannot be determined, try to process as text file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                f.read(100)  # Try to read first 100 characters
                return "text", "Text_Inspector_Tool"  # If readable as text, treat as text file
        except UnicodeDecodeError:
            pass  # Not a text file
            
        return "binary", None  # Unrecognized binary file