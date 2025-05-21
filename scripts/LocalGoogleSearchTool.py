import random
from typing import List, Dict, Any, Optional

from smolagents import Tool
from smolagents.models import MessageRole, Model

class LocalGoogleSearchTool(Tool):
     name = "local_google_search"
     description = """This tool performs a Google web search or a targeted search on specific websites (e.g., JSTOR) based on the given query. It preprocesses the query to optimize it for search engines, retrieves relevant search results, and outputs the summarized content from the visited pages."""
     
     inputs = {
          "question": {
               "type": "string",
               "description": "The web search query to perform."
          },
     }
     output_type = "string"

     used_urls = []

     def __init__(self, model, browser):
          super().__init__()
          self.model = model
          self.browser = browser

     def forward(self, question: str) -> str:
          simplified_question_messages = [
               {
                    "role": MessageRole.SYSTEM,
                    "content": [
                         {
                              "type": "text",
                              "text": f"""{question} Please make the question more suitable for Google search. 
                              If the sentence is a question, change it to a declarative sentence. 
                              If the question contains quotation, only keep the content inside the quote.
                              If the question contains time information, place it at the beginning of the output. 
                              Only output the revised answer. """
                         }
                    ]
               }
          ]
          simplified_question = self.model(simplified_question_messages).content
          print(f"simplified_question:{simplified_question}")

          results = []
          results.append("Use a LOOP to visit 6 urls in the following in one step.")
          
          random_choices = [1, 2]
          random_weights = [0.8, 0.2]
          random_number = random.choices(random_choices, weights=random_weights, k=1)[0]
          
          if random_number == 1:
               try:
                    browser_tmp = self.browser
                    browser_tmp.visit_page(f"google:  {simplified_question}")
                    header, content = browser_tmp._state()
                    results.append(header.strip() + "\n=======================\n" + content[:(len(content) // 3)])
               except Exception as e:
                    pass
          
          else:
               # put the jstor.org into the google_search
               try:
                    browser_tmp = self.browser
                    browser_tmp.visit_page(f"google: site:jstor.org {simplified_question}")
                    header, content = browser_tmp._state()
                    results.append(header.strip() + "\n=======================\n" + content[:(len(content) // 4)])
               except Exception as e:
                    pass

               # put the original question into the google_search
               try:
                    browser_tmp = self.browser
                    browser_tmp.visit_page(f"google:  {question}")
                    header, content = browser_tmp._state()
                    results.append(header.strip() + "\n=======================\n" + content[:(len(content) // 4)])
               except Exception as e:
                    pass

          final_result = "\n".join(results)

          print(f"local_google_search:answers:{final_result}")
          return final_result

def load_web_data(
     excel_path: str,
     files_dir: Optional[str] = "Historical/Historical",
     sheet_name: Optional[str] = 'Sheet1',
) -> str:
     """
     Load custom Excel format web data index (including URLs and descriptions)    
     Parameters:
          excel_path: Path to the Excel file
          files_dir: Directory containing related files
          sheet_name: Name or index of the worksheet to load, None to load all sheets
          
     Returns:
          Index data text
     """
     os.makedirs(files_dir, exist_ok=True)
     
     print(f"Loading Excel file: {excel_path}")
     try:
          # Read all sheets
          if sheet_name is None:
               # First get all sheet names
               xls = pd.ExcelFile(excel_path)
               sheet_names = xls.sheet_names
               print(f"Detected {len(sheet_names)} sheets: {sheet_names}")
               
               # Read data from all sheets
               all_data = []
               for sheet in sheet_names:
                    print(f"Reading sheet: {sheet}")
                    df = pd.read_excel(excel_path, sheet_name=sheet)
                    
                    # Check if sheet is empty
                    if df.empty:
                         print(f"Sheet {sheet} is empty, skipping")
                         continue
                
                    # Check if sheet has necessary columns
                    required_columns = ["序号", "网址", "简介"]
                    missing_columns = [col for col in required_columns if col not in df.columns]
                    if missing_columns:
                         print(f"Sheet {sheet} is missing required columns: {missing_columns}, trying alternative column names")
                         
                         # Try mapping column names
                         column_mapping = {
                              "序号": ["ID", "id", "编号", "序号"],
                              "网址": ["Website", "website", "网站"],
                              "简介": ["简述", "introduction", "Introduction"]
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
                
                    # Add to total data
                    all_data.append(df)
            
               if not all_data:
                    print("All sheets failed requirements, returning empty dataset")
                    return "No Data Available"
            
               # Merge all sheet data
               df = pd.concat(all_data, ignore_index=True)
          else:
               # Read specified sheet
               df = pd.read_excel(excel_path, sheet_name=sheet_name)

     except Exception as e:
          print(f"Error loading Excel file: {e}")
          return "No Data Available"
     
     web_data_text = "This is a data index text: each line corresponds to a database URL and its description. Please find the URL you want to visit based on the description (find the most relevant one based on country, time, data type, etc.) to search for your question\n"
     for _ , row in df.iterrows():
          id = row['序号']  
          website_url = row['网址']    
          introduction = row['简介']  

          # if website_url.rstrip()[-1] == "=":
          web_data_text += f"ID:{id} URL:{website_url} Description:{introduction}\n"
    
     return web_data_text
