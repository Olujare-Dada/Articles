# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 23:38:52 2024

@author: olanr
"""

"""
Many times, when using LLMs you often want them to go through some input data and return an output which could be a piece of information from the input, some predefined response
from a given set of responses, or you just want it to return some character or a boolean. Usually, this output would determine the next flow or block of code to be executed. For
example, you want the LLM to respond True or False if an input block of text is perceived as spam or not. The result of this check is parsed and then used to execute a function
which helps to move the message into the spam folder or input email folder. This system represents two separate blocks: the check system implemented with the LLM, and the 
custom code that performs the action. One drawback of this is that the LLM becomes a binary output system good only for outputing boolean values. If you want extra functionalities
you have to perform further prompt engineering which can get bloated and complex fairly quickly. One way to solve this would be to give the LLM the option to use your custom
code to perform your action (i.e. move an email to a spam folder or move an email to the inbox folder). This way, you can derive more value from the LLM since you can 
still use the LLM for normal day tasks or other specific tasks without writing complex and/or bloated prompts.

Function Calling is a functionality that allows an LLM to call a user-defined function if necessary for solving a particular task. This makes LLMs more powerful because through
function calling, they're able to interact with the outside world. A popular usecase for function calling enables LLMs, which were once terrible at math, to become more
proficient at math computations through calculator APIs it can make calls to, through a function written to call that calculator API. 
Another usecase is a customer support chatbot which handles FAQs using an LLM, but for specific inquiries like retrieving an order status or processing a return request, 
it issues calls to a backend function that interfaces with your e-commerce system.

In this article, I will be sharing my unique usecase of function calling. 

Scenario: I have an LLM I use to explore the latest data about a website's traffic moved into a container service (like S3 bucket or Azure Data Lake). There are two statistics
I am interested in: Summarized General Statistics like the average number of website traffic spikes, outliers, bot edits over a certain threshold, and so on; and an aggregated
statistic for data grouped by users and their various namespaces. There will be two functions for achieving these functionalities and in this article, we would be using the
function calling capability of the OpenAI LLM to execute.


Before we begin coding, let us set up the environment.
1. Create a new folder to host the project.
2. Copy and paste the following packages into a .txt file called "requirements.txt":
azure-storage-filedatalake
pandas
openai
python-dotenv

3. Open up your terminal (CMD), navigate to the folder where your project directory is and type ```pip install -r requirements.txt```
4. Before performing step 3, it is good practice for you to set up a virtual environment but we can also proceed without doing so. If you are interested in knowing how to 
create a virtual environment but don't know how, check out this article: LINK
5. If you created a virtual environment, you can use the following command in the terminal to start it up before installing the packages in the requirement.txt file:
```my_env\Scripts\activate``` or ```conda activate -n my_env``` for conda virtual environments environments.
6. Once the packages are done installing, we can proceed with the code
7. Next, create another .txt file in the project folder, name it ".env" and inside this file enter your credentials. It should look like this:
AZURE_STORAGE_ACCOUNT= "azure-storage-account-name"
AZURE_CONTAINER_NAME= "azure-container-name"
AZURE_SAS_TOKEN = "azure-container-sas-token"
OPENAI_API_KEY=sk-abcdefg



#IMPORT RELEVANT PACKAGES:
from azure.storage.filedatalake import DataLakeServiceClient, FileSystemClient
from typing import Dict
from azure.core.exceptions import ResourceNotFoundError
import io
import os
import pandas as pd
from openai import OpenAI
from openai import APIConnectionError, RateLimitError
from datetime import datetime
from enum import Enum
import json
from dotenv import load_dotenv

#Load the environment variables
load_dotenv()

Create the date and time values so they can be used in the code ahead:
```current_year = datetime.now().year
current_month = datetime.now().month
current_month_name = datetime.now().strftime("%B")
current_day = 26 #datetime.now().day
current_hour = datetime.now().hour
current_minute = datetime.now().minute
current_second = datetime.now().second
```

Defining the folder where the log files are written to as well as the log file itself. An example of the folder structure is: wikilogs_llm/2024/February/27/llm_logs_2024_02_27.parquet
llm_logfile = f"llm_logs_{current_year}_{current_month}_{current_day}.parquet"
LLM_LOG_FOLDER = f"wikilogs_llm/{current_year}/{current_month_name}/{current_day}/"


Get our credentials stored in the ".env" file
account_name = os.getenv("AZURE_STORAGE_ACCOUNT")
sas_token = os.getenv("AZURE_SAS_TOKEN")
container_name = os.getenv("AZURE_CONTAINER_NAME")
print(f"{container_name = }")


Create the service and container client for interacting with the data lake folder:
```service_client = DataLakeServiceClient(
    account_url=f"https://{account_name}.dfs.core.windows.net",
    credential=sas_token
)

container_client: FileSystemClient = service_client.get_file_system_client(file_system=container_name)
```

Create the two functions to read and extact the general statistic and aggregated statistic

def summarize_general_stats(folder_path: str, file_name: str)-> Dict:
    directory_client = container_client.get_directory_client(folder_path)
    
    directory_client.create_directory()
    
    file_client = directory_client.get_file_client(file_name)
    
    try:
        downloaded_data = file_client.download_file()
        
    except ResourceNotFoundError:
        print(f"File with name {file_name} could not be found in folder {folder_path} on the ADLS")
        return "No data to summarize"
    
    parquet_data = downloaded_data.readall()
    adls_df = pd.read_parquet(io.BytesIO(parquet_data))
    
    general_stats = adls_df.describe()
    
    return general_stats.to_dict(orient = "list")

def generate_aggregated_stats(folder_path: str, file_name: str, word_cutoff:int = 2_600)-> Dict:
    directory_client = container_client.get_directory_client(folder_path)
    
    directory_client.create_directory()
    
    file_client = directory_client.get_file_client(file_name)
    
    try:
        downloaded_data = file_client.download_file()
        
    except ResourceNotFoundError:
        print(f"File with name {file_name} could not be found in folder {folder_path} on the ADLS")
        return "No data to summarize"
    
    parquet_data = downloaded_data.readall()
    adls_df = pd.read_parquet(io.BytesIO(parquet_data))
    
    aggregated_df = adls_df.drop(["timestamp", "ingestion_date", "id"], axis = 1).groupby(["user", "namespace"]).sum()
    aggregated_df = aggregated_df.reset_index()
    
    for idxx, row in aggregated_df.iterrows():
        idx = idxx + 1
        num_words_title = len(aggregated_df.head(idx)["title"].values[0].split())
        num_words_user = len(aggregated_df.head(idx)["user"].values[0].split())
        num_words_title_url = len(aggregated_df.head(idx)["title_url"].values[0].split())
        num_parsedcomment = len(aggregated_df.head(idx)["parsedcomment"].values[0].split())
        
        total_words = num_words_title + num_words_user + num_words_title_url + num_parsedcomment
        
        if total_words > word_cutoff:
            if idx == 1:
                truncated_df = aggregated_df.head(1)
                truncated_df["parsedcomment"].astype(str).str[:50]
                
                return truncated_df.head(1).to_dict(orient = "list")
            return aggregated_df.head(idx - 1).to_dict(orient = "list")
    
    return aggregated_df.to_dict(orient = "list")


Create the tools array that references the two functions:

tools_definition = [
  {
    "type": "function",
    "function": {
      "name": "summarize_general_stats",
      "description": "Get the latest summary statistics for a parquet logfile using df.describe.",
      "parameters": {
        "type": "object",
        "properties": {
          "folder_path": {
            "type": "string",
            "description": "Folder where the required parquet logfile is located"
          },
          "file_name": {
            "type": "string",
            "description": "The parquet logfile located in the folder_path with the information required."
          }
        },
        "required": ["folder_path", "file_name"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "generate_aggregated_stats",
      "description": "Get the latest aggregated statistics from parquet logfile using groupby 'user' and 'namespace'.",
      "parameters": {
        "type": "object",
        "properties": {
          "folder_path": {
            "type": "string",
            "description": "Folder where the required parquet logfile is located"
          },
          "file_name": {
            "type": "string",
            "description": "The parquet logfile located in the folder_path with the information required."
          }
        },
        "required": ["folder_path", "file_name"]
      }
    }
  }
]


This function is designed to prepare data for JSON serialization, with special handling for pd.Timestamp objects (to convert them into ISO string format) and NaN values (to replace them with None).

```
def prepare_data_for_json(data):
    prepared_data = {}
    for key, values in data.items():
        # Convert Timestamp objects to ISO strings and replace NaN with None
        prepared_data[key] = [
            value.isoformat() if isinstance(value, pd.Timestamp) else (None if pd.isna(value) else value)
            for value in values
        ]
    return prepared_data
```

This function is designed to handle a conversation with OpenAI's GPT-4 model, allowing the model to make function calls (e.g., calling external functions for data processing) and return a response based on those function calls.

def run_conversation(content):
    messages = [{"role": "user", "content": content}]

    response = client.chat.completions.create(model = "gpt-4o", 
                                              messages = messages, 
                                              tools = tools_definition, 
                                              tool_choice= "auto"
                                              )
    
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    
    if tool_calls:
        
        messages.append(response_message)
        
        available_functions = {"summarize_general_stats": summarize_general_stats,
                               "generate_aggregated_stats": generate_aggregated_stats
                               }
        
        for tool_call in tool_calls:
            print(f"Function: {tool_call.function.name}")
            print(f"Params: {tool_call.function.arguments}")
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(
                folder_path = function_args.get("folder_path"),
                file_name = function_args.get("file_name")
                )
            
            function_response = prepare_data_for_json(function_response)
            
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": json.dumps(function_response)
                    }
                )
    
    second_response = client.chat.completions.create(
        model = "gpt-4o",
        messages = messages,
        )
    
    return second_response.choices[0].message.content


messages = [{"role": "user", "content": content}]:

Initializes the conversation with a user message containing the input content. This content can be any query or instruction that the user provides.
response = client.chat.completions.create(...):

Sends a request to OpenAI's GPT-4 model using chat.completions.create. The request includes:
model="gpt-4o": The model to use for the conversation (likely GPT-4).
messages=messages: The list of messages to send to the model. Initially, this contains the user's message.
tools=tools_definition: A list of available tools (defined elsewhere in the code). These tools correspond to user-defined functions that the model can call during the conversation.
tool_choice="auto": Instructs the model to automatically select and call the appropriate tool (function) based on the conversation.
response_message = response.choices[0].message:

Extracts the message from the model's response. The model will include information about any function calls that it believes are necessary.
tool_calls = response_message.tool_calls:

If the model has made any tool calls, they are included in tool_calls. This is a list of the functions that the model has selected to call based on the conversation.
if tool_calls::

If there are any tool calls, the code processes them:
messages.append(response_message): Adds the response from the model (which may include function call instructions) to the conversation history.
available_functions = {...}: Defines a mapping of tool names (from the model) to the actual Python functions (summarize_general_stats and generate_aggregated_stats).
for tool_call in tool_calls::
Iterates over the tool calls and prints the function name and parameters.
function_name = tool_call.function.name: Extracts the function name from the tool call.
function_args = json.loads(tool_call.function.arguments): Parses the arguments for the function call.
function_response = function_to_call(...): Calls the appropriate function with the parsed arguments (e.g., folder_path and file_name).
function_response = prepare_data_for_json(function_response): Prepares the response from the function (e.g., converting timestamps to ISO format and handling missing data).
messages.append(...): Appends the function's response back to the conversation, marking it as a tool response.
second_response = client.chat.completions.create(...):

After all tool calls have been processed, a second request is sent to the model to generate the final response, which incorporates the results from the tool calls.
return second_response.choices[0].message.content:

Returns the content of the final response from the model.
Key Points:
This function allows the LLM to call external functions (like summarize_general_stats and generate_aggregated_stats) and process their responses within the conversation, creating a dynamic and interactive flow.
The LLM can use the available functions to fetch data, process it, and return meaningful results, all while maintaining the flow of the conversation.



The PromptTemplate class is an enumeration (Enum) that defines a template for constructing prompts in the application.

class PromptTemplate(Enum):
    
    params = "Folder: {folder}\nFile: {file}\n\nThe file mentioned above is used to monitor the logs of a wiki stream for suspicious bot activity."
    



def ask_question(question: str):
    
    content = PromptTemplate.params.value.format(folder = LLM_LOG_FOLDER, file = llm_logfile)
    content += question

    assistant_response = run_conversation(content)
    
    return assistant_response



"======================================== Execution ========================================================"


if __name__ == "__main__":
    question = "Give me the aggregated statistics for today"

"""
from azure.storage.filedatalake import DataLakeServiceClient, FileSystemClient
from typing import Dict
from azure.core.exceptions import ResourceNotFoundError
import io
import os
import pandas as pd
from openai import OpenAI
from openai import APIConnectionError, RateLimitError
from datetime import datetime
from enum import Enum
import json
from dotenv import load_dotenv

load_dotenv()

current_year = datetime.now().year
current_month = datetime.now().month
current_month_name = datetime.now().strftime("%B")
current_day = 26 #datetime.now().day
current_hour = datetime.now().hour
current_minute = datetime.now().minute
current_second = datetime.now().second


llm_logfile = f"llm_logs_{current_year}_{current_month}_{current_day}.parquet"
LLM_LOG_FOLDER = f"wikilogs_llm/{current_year}/{current_month_name}/{current_day}/"


account_name = os.getenv("AZURE_STORAGE_ACCOUNT")
sas_token = os.getenv("AZURE_SAS_TOKEN")
container_name = os.getenv("AZURE_CONTAINER_NAME")
print(f"{container_name = }")

service_client = DataLakeServiceClient(
    account_url=f"https://{account_name}.dfs.core.windows.net",
    credential=sas_token
)

# Get the container client
container_client: FileSystemClient = service_client.get_file_system_client(file_system=container_name)


def summarize_general_stats(folder_path: str, file_name: str)-> Dict:
    directory_client = container_client.get_directory_client(folder_path)
    
    directory_client.create_directory()
    
    file_client = directory_client.get_file_client(file_name)
    
    try:
        downloaded_data = file_client.download_file()
        
    except ResourceNotFoundError:
        print(f"File with name {file_name} could not be found in folder {folder_path} on the ADLS")
        return "No data to summarize"
    
    parquet_data = downloaded_data.readall()
    adls_df = pd.read_parquet(io.BytesIO(parquet_data))
    
    general_stats = adls_df.describe()
    
    return general_stats.to_dict(orient = "list")



def generate_aggregated_stats(folder_path: str, file_name: str, word_cutoff:int = 2_600)-> Dict:
    directory_client = container_client.get_directory_client(folder_path)
    
    directory_client.create_directory()
    
    file_client = directory_client.get_file_client(file_name)
    
    try:
        downloaded_data = file_client.download_file()
        
    except ResourceNotFoundError:
        print(f"File with name {file_name} could not be found in folder {folder_path} on the ADLS")
        return "No data to summarize"
    
    parquet_data = downloaded_data.readall()
    adls_df = pd.read_parquet(io.BytesIO(parquet_data))
    
    aggregated_df = adls_df.drop(["timestamp", "ingestion_date", "id"], axis = 1).groupby(["user", "namespace"]).sum()
    aggregated_df = aggregated_df.reset_index()
    
    for idxx, row in aggregated_df.iterrows():
        idx = idxx + 1
        num_words_title = len(aggregated_df.head(idx)["title"].values[0].split())
        num_words_user = len(aggregated_df.head(idx)["user"].values[0].split())
        num_words_title_url = len(aggregated_df.head(idx)["title_url"].values[0].split())
        num_parsedcomment = len(aggregated_df.head(idx)["parsedcomment"].values[0].split())
        
        total_words = num_words_title + num_words_user + num_words_title_url + num_parsedcomment
        
        if total_words > word_cutoff:
            if idx == 1:
                truncated_df = aggregated_df.head(1)
                truncated_df["parsedcomment"].astype(str).str[:50]
                
                return truncated_df.head(1).to_dict(orient = "list")
            return aggregated_df.head(idx - 1).to_dict(orient = "list")
    
    return aggregated_df.to_dict(orient = "list")
    



tools_definition = [
  {
    "type": "function",
    "function": {
      "name": "summarize_general_stats",
      "description": "Get the latest summary statistics for a parquet logfile using df.describe.",
      "parameters": {
        "type": "object",
        "properties": {
          "folder_path": {
            "type": "string",
            "description": "Folder where the required parquet logfile is located"
          },
          "file_name": {
            "type": "string",
            "description": "The parquet logfile located in the folder_path with the information required."
          }
        },
        "required": ["folder_path", "file_name"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "generate_aggregated_stats",
      "description": "Get the latest aggregated statistics from parquet logfile using groupby 'user' and 'namespace'.",
      "parameters": {
        "type": "object",
        "properties": {
          "folder_path": {
            "type": "string",
            "description": "Folder where the required parquet logfile is located"
          },
          "file_name": {
            "type": "string",
            "description": "The parquet logfile located in the folder_path with the information required."
          }
        },
        "required": ["folder_path", "file_name"]
      }
    }
  }
]


def prepare_data_for_json(data):
    prepared_data = {}
    for key, values in data.items():
        # Convert Timestamp objects to ISO strings and replace NaN with None
        prepared_data[key] = [
            value.isoformat() if isinstance(value, pd.Timestamp) else (None if pd.isna(value) else value)
            for value in values
        ]
    return prepared_data


client = OpenAI()

def run_conversation(content):
    messages = [{"role": "user", "content": content}]

    response = client.chat.completions.create(model = "gpt-4o", 
                                              messages = messages, 
                                              tools = tools_definition, 
                                              tool_choice= "auto"
                                              )
    
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    
    if tool_calls:
        
        messages.append(response_message)
        
        available_functions = {"summarize_general_stats": summarize_general_stats,
                               "generate_aggregated_stats": generate_aggregated_stats
                               }
        
        for tool_call in tool_calls:
            print(f"Function: {tool_call.function.name}")
            print(f"Params: {tool_call.function.arguments}")
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(
                folder_path = function_args.get("folder_path"),
                file_name = function_args.get("file_name")
                )
            
            function_response = prepare_data_for_json(function_response)
            
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": json.dumps(function_response)
                    }
                )
    
    second_response = client.chat.completions.create(
        model = "gpt-4o",
        messages = messages,
        )
    
    return second_response.choices[0].message.content



class PromptTemplate(Enum):
    
    params = "Folder: {folder}\nFile: {file}\n\nThe file mentioned above is used to monitor the logs of a wiki stream for suspicious bot activity."
    



def ask_question(question: str):
    
    content = PromptTemplate.params.value.format(folder = LLM_LOG_FOLDER, file = llm_logfile)
    content += question

    assistant_response = run_conversation(content)
    
    return assistant_response



"======================================== Execution ========================================================"


if __name__ == "__main__":
    question = "Give me the aggregated statistics for today"
    #ask_question(question)
    ga = generate_aggregated_stats(LLM_LOG_FOLDER, llm_logfile)
    sg = summarize_general_stats(LLM_LOG_FOLDER, llm_logfile)
    print(ga)
    print("\n\n\n\n\n")
    print(sg)