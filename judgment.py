import os
import json
import argparse
import asyncio
import glob
from typing import Literal
from pydantic import BaseModel
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
import dotenv

dotenv.load_dotenv()
# Initialize OpenAI async client
client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),  # Replace with your actual API key
    timeout=300.0, 
    max_retries=1
)

# Judging prompt template
JUDGE_PROMPT = """You are a fair evaluator. Judge whether the following [response] to [question] is semantically consistent with the [correct_answer] below.  

[question]: {question}  

[response]: {response}  

[correct_answer]: {correct_answer}  

When you judge, consider only whether the core meaning and all necessary key points in the response match the correct answer.  Even if wording or format differs, treat equivalent semantics as correct. Treat missing key points or any substantive error or omission as incorrect. For numerical answers, a small rounding difference is acceptable. Tolerate substantive deviations from the correct answer. If the extracted_final_answer is a more specific instance of the correct_answer (for example, “Pieter Schenk II” vs “Pieter Schenk”), and it still contains the core string of the correct_answer, treat it as correct.

Please output exactly in the format and criteria specified below:  

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0|\%| and 100|\%| from [response]. Put 100 if there is no confidence score available."""


# Pydantic model definition
class ExtractedAnswer(BaseModel):
    extracted_final_answer: str
    reasoning: str
    correct: Literal["yes", "no"]
    confidence: int
    strict: Literal[True]

async def extract_answer(question: str, correct_answer: str, response: str, judge_model: str) -> ExtractedAnswer:
    prompt = JUDGE_PROMPT.format(
        question=question,
        response=response,
        correct_answer=correct_answer
    )
    api_response = await client.beta.chat.completions.parse(
        model=judge_model,
        max_completion_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
        response_format=ExtractedAnswer,
    )
    return api_response.choices[0].message.parsed

async def judge_single(entry: dict, judge_model: str, semaphore: asyncio.Semaphore) -> dict:
    """
    Judge a single entry and return a new result dict.
    """
    async with semaphore:
        parsed = await extract_answer(
            question=entry.get("question", ""),
            correct_answer=entry.get("true_answer", ""),
            response=entry.get("model_answer", ""),
            judge_model=judge_model
        )
        # Construct new result entry
        return {
            "task_id": entry.get("task_id"),
            "question": entry.get("question"),
            "model_answer": entry.get("answer"),
            "true_answer": entry.get("true_answer"),
            "is_correct": parsed.correct == "yes",
            "extracted_final_answer": parsed.extracted_final_answer,
            "reasoning": parsed.reasoning,
            "confidence": parsed.confidence
        }

async def judge_all(entries: list, judge_model: str, num_workers: int) -> list:
    semaphore = asyncio.Semaphore(num_workers)
    tasks = [judge_single(e, judge_model, semaphore) for e in entries]
    return await tqdm_asyncio.gather(*tasks)

def main():
    parser = argparse.ArgumentParser(description="Judge QA JSON entries")
    parser.add_argument("--judge", type=str, default="gpt-4o", help="Judging model")
    parser.add_argument("--num_workers", type=int, default=20, help="Number of concurrent calls")
    parser.add_argument("--input_dir", type=str, help="Input directory containing jsonl files to process")
    parser.add_argument("--output_dir", type=str, help="Output directory for saving judging results")
    parser.add_argument("--input_file", type=str, help="Single input file path (optional, higher priority than input_dir)")
    parser.add_argument("--output_file", type=str, help="Single output file path (optional, higher priority than output_dir)")
    args = parser.parse_args()

    if args.input_file and args.output_file:
        # Process a single file
        process_file(args.input_file, args.output_file, args.judge, args.num_workers)
    elif args.input_dir and args.output_dir:
        # Process all files in the directory
        os.makedirs(args.output_dir, exist_ok=True)
        jsonl_files = glob.glob(os.path.join(args.input_dir, "*.jsonl"))
        
        if not jsonl_files:
            print(f"No .jsonl files found in {args.input_dir}")
            return
        
        for input_file in jsonl_files:
            filename = os.path.basename(input_file)
            output_filename = os.path.splitext(filename)[0] + "_judgment.jsonl"
            output_file = os.path.join(args.output_dir, output_filename)
            
            print(f"Processing file: {input_file} -> {output_file}")
            process_file(input_file, output_file, args.judge, args.num_workers)
    else:
        # Use default file paths
        input_json = "output_openai_baseline/level2_summary/test_baseline4.jsonl"
        output_json = "output_openai_baseline/level2_summary/test_baseline4_judgment.jsonl"
        process_file(input_json, output_json, args.judge, args.num_workers)

def process_file(input_json, output_json, judge_model, num_workers):
    """Helper function to process a single file"""
    # Read input
    entries = []
    with open(input_json, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Skipping unparseable line: {e}")

    if not entries:
        print(f"Warning: No valid data in {input_json}")
        return

    # Parallel judging
    judged_results = asyncio.run(judge_all(entries, judge_model, num_workers))

    # Calculate Accuracy
    total = len(judged_results)
    correct_cnt = sum(1 for r in judged_results if r["is_correct"])
    accuracy = correct_cnt / total * 100 if total > 0 else 0.0
    print(f"*** {os.path.basename(input_json)} Accuracy: {accuracy:.2f}% ({correct_cnt}/{total}) ***")

    # Write new JSON
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        for result in judged_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
