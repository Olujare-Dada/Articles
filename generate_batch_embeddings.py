"""
==== Job Title Embeddings Module ====

This module facilitates the generation of embeddings for job titles using the OpenAI API. 
It includes functionality to:
- Prepare job titles from a CSV file.
- Create JSONL files for batch processing.
- Upload batches of job titles to the OpenAI API.
- Monitor the status of batch jobs asynchronously.
- Retrieve and save embeddings in structured formats (JSONL and CSV).
"""



import pandas as pd
import numpy as np
import json
from openai import OpenAI
from openai import APIConnectionError, RateLimitError
from dotenv import load_dotenv
import os
import time
from typing import Dict, Any, List
import asyncio


load_dotenv()




def create_request_object(*, request_number: int, job_title: str)->Dict[str, Any]:

    """
    Creates a request object for the OpenAI embeddings API.

    Args:
        request_number (int): The sequential request number.
        job_title (str): The job title to be embedded.

    Returns:
        Dict[str, Any]: A dictionary representing the API request object.
    """
    
    request_object = {
        "custom_id": f"request-{request_number}",
        "method": "POST",
        "url": "/v1/embeddings",
        "body": {
            "model": "text-embedding-3-small",
            "input": job_title,
            "encoding_format": "float",
            
            }
        }
    return request_object



def create_folder(foldername: str)->None:

    """
    Creates a folder if it does not already exist.

    Args:
        foldername (str): The name of the folder to create.
    """
    
    if not os.path.exists(foldername):
        os.mkdir(foldername)




def create_batch_jsonl(*,
                       batch_num: int,
                       job_titles_folder: str,
                       job_titles: List,
                       updated_idx_start: int = 0
                       )-> None:


    """
    Creates a JSONL file containing a batch of job title requests.

    Args:
        batch_num (int): The batch number.
        job_titles_folder (str): The folder to save the JSONL file.
        job_titles (List): A list of job titles to include in the batch.
        updated_idx_start (int): The starting index for job titles.
    """
    
    if not job_titles:
        print("There are no Job Titles provided. Check your data source")
        print(f"Current batch starts at job index: {updated_idx_start}\n")
        return
    
    with open(f"{job_titles_folder}/job_title_batch_{batch_num + 1}.jsonl", "w") as f:
        for idx, job_title in enumerate(job_titles):

            request_number = idx+1 + updated_idx_start
            job_request_object = create_request_object(request_number = request_number,
                                                       job_title = job_title
                                                       )
            
            f.write(json.dumps(job_request_object) + "\n")



def create_jsonl_for_job_titles(*, job_titles_array: np.ndarray)->None:

    """
    Splits job titles into batches and creates JSONL files for each batch.

    Args:
        job_titles_array (np.ndarray): An array of job titles to process.
    """
    
    batch_sum = 2000 if len(job_titles_array) > 2000 else int(len(job_titles_array)/2)#sum of all the lines in 1 .jsonl file
    new_beginning = 0

    if job_titles_array.size <= 0:
        print(f"Job titles array is empty.\n{job_titles_array = }")
        return
    
    for batch_idx, batch_start_val in enumerate(range(0, len(job_titles_array)+1, batch_sum)):
        
        batch_job_titles = job_titles_array.tolist()[new_beginning: new_beginning + batch_sum]

        create_batch_jsonl(batch_num = batch_idx,
                           job_titles_folder = job_titles_folder,
                           job_titles = batch_job_titles,
                           updated_idx_start = batch_start_val
                           )
        
        print(f"{new_beginning = }\n {batch_sum = }")
        new_beginning += batch_sum



def create_batch_input_files(*, openai_client: OpenAI, job_titles_folder: str)-> List:

    """
    Creates batch input files by uploading JSONL files to OpenAI.

    Args:
        openai_client (OpenAI): The OpenAI client instance.
        job_titles_folder (str): The folder containing the JSONL files.

    Returns:
        List: A list of batch input file IDs.
    """
    
    batch_input_file_ids = []
    jsonl_files = os.listdir(job_titles_folder)
    if not jsonl_files:
        return []
    
    for jsonl_file in jsonl_files:
        jsonl_filepath = os.path.join(job_titles_folder, jsonl_file)

        try:
            batch_input_file = openai_client.files.create(
              file=open(jsonl_filepath, "rb"),
              purpose="batch"
            )
        except (APIConnectionError, RateLimitError) as e:
            print(f"\nAn error occurred.\n{e}")
            print("Returning an empty List array\n")
            return []

        batch_input_file_ids.append(batch_input_file.id)

    return batch_input_file_ids



def batch_job_starter(*, openai_client: OpenAI, batch_input_file_id: str)->str:

    """
    Starts a batch job for processing embeddings.

    Args:
        openai_client (OpenAI): The OpenAI client instance.
        batch_input_file_id (str): The ID of the input file for the batch job.

    Returns:
        str: The ID of the created batch job.
    """
    
    batch_creation_object = openai_client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/embeddings",
        completion_window="24h",
        metadata={
          "description": "nightly eval job"
        }
    )

    return batch_creation_object.id



def initialize_batch_jobs(*, openai_client: OpenAI, batch_input_file_ids: List)->List:

    """
    Initializes batch jobs for processing embeddings using the provided input file IDs.

    Args:
        openai_client (OpenAI): The OpenAI client instance used to start batch jobs.
        batch_input_file_ids (List): A list of batch input file IDs to be processed.

    Returns:
        List: A list of batch job IDs created.
    """
    
    if not batch_input_file_ids:
        print(f"No batch Input File IDs generated.\n")
        return []
    
    batch_ids = []
    for batch_input_file_id in batch_input_file_ids:

        batch_id = batch_job_starter(openai_client = openai_client,
                                     batch_input_file_id = batch_input_file_id
                                     )
        batch_ids.append(batch_id)

    return batch_ids



async def check_job_status(openai_client: OpenAI, batch_id: str, check_idx: int)->Dict:

    """
    Checks the status of a batch job until it completes.

    Args:
        openai_client (OpenAI): The OpenAI client instance.
        batch_id (str): The ID of the batch job to check.
        check_idx (int): The current check iteration index.

    Returns:
        Dict: A dictionary containing the batch ID and output file ID upon completion.
    """
    
    while True:
        job = openai_client.batches.retrieve(batch_id)
        if job.output_file_id:
            print(f"Job {batch_id} complete. Output file ID: {job.output_file_id}")
            return {batch_id: job.output_file_id}  # Return batch_id and output file ID

        print(f"===Current status for Batch ID: {batch_id}===")
        completed = job.request_counts.completed
        failed = job.request_counts.failed
        total = completed = job.request_counts.total
        print(f"Check Number: {check_idx}")
        print(f"Completion status: \n{completed = }\n{failed = }\n{total = }\n\n")
        
        await asyncio.sleep(60)  # Wait for 60 seconds before checking again

        

async def monitor_multiple_jobs(openai_client: OpenAI, batch_ids: List):

    """
    Monitors multiple batch jobs concurrently.

    Args:
        openai_client (OpenAI): The OpenAI client instance.
        batch_ids (List): A list of batch job IDs to monitor.

    Returns:
        List: A list of results from the monitored jobs.
    """
    
    print("\n===================== Monitoring Mulitple Batch Jobs =====================")
    tasks = [check_job_status(openai_client, batch_id, check_idx) for check_idx, batch_id in enumerate(batch_ids)]
    results = await asyncio.gather(*tasks)  # Wait for all tasks to complete
    return results  # Collect and return the results



def write_enbeddings_to_file(output_file_ids: Dict, output_folder: str)->None:

    """
    Writes embeddings to files based on output file IDs.

    Args:
        output_file_ids (Dict): A dictionary mapping batch IDs to output file IDs.
        output_folder (str): The folder to save the output files.
    """
    
    print(f"\n\nWriting the embeddings to output folder {output_folder}")
    if not output_file_ids:
        print("Cannot write embeddings to file because there is no Output File ID")
        return
    
    for idx, output_file_id_map in enumerate(output_file_ids):
        output_file_id = [val for val in output_file_id_map.values()][0]
        
        file_content = client.files.content(output_file_id)

        output_filepath = os.path.join(output_folder, f"gpt_output_{idx}.jsonl")
        with open(output_filepath, "wb") as f:
            f.write(file_content.read())
        print(f"{output_filepath} file created and populated with embeddings")



def convert_jsonl_files_to_dataframe(jsonl_file_folder: str)->pd.DataFrame:

    """
    Converts JSONL files in a folder to a Pandas DataFrame.

    Args:
        jsonl_file_folder (str): The folder containing the JSONL files.

    Returns:
        pd.DataFrame: A DataFrame containing the combined data from the JSONL files.
    """
    
    
    list_of_dataframes = []
    list_of_jsonl_files: List[str] = os.listdir(jsonl_file_folder)

    if not list_of_jsonl_files:
        print("Folder {jsonl_file_folder} is empty")
        return pd.DataFrame()

    for jsonl_file in list_of_jsonl_files:
        if not jsonl_file.endswith(".jsonl"):
            print(f"File {jsonl_file} within folder {jsonl_file_folder} not a .jsonl file. Skipping file...")

            continue
        
        jsonl_filepath = os.path.join(jsonl_file_folder, jsonl_file)

        try:
            df = pd.read_json(jsonl_filepath, lines = True)
        except ValueError as e:
            print(f"Failed to read file {jsonl_filepath}.\nPython error: {e}\nSkipping file...")

        if "response" in df.columns:
            df["job_title_embeddings"] = df["response"].apply(lambda row: row["body"]["data"][0]["embedding"])
        else:
            df["job_title"] = df["body"].apply(lambda row: row["input"])
        
        list_of_dataframes.append(df)

    final_df = pd.concat(list_of_dataframes)
    
    return final_df



def create_job_embeddings_csv(job_titles_df: pd.DataFrame, embeddings_df: pd.DataFrame)-> None:

    """
    Merges job titles and their embeddings into a CSV file.

    Args:
        job_titles_df (pd.DataFrame): A DataFrame containing job titles.
        embeddings_df (pd.DataFrame): A DataFrame containing job title embeddings.
    """
    
    if len(job_titles_df) == 0 or len(embeddings_df) == 0:
        print("One or both of the final dataframes is/are empty. Returning None...")
        return
    
    merged_df = pd.merge(job_titles_df, embeddings_df, on="custom_id", how = "inner")
    trimmed_df = merged_df[["job_title", "job_title_embeddings"]]

    filename = "jobs_with_embeddings.csv"

    trimmed_df.to_csv(filename, index = False)
    print(f"\nJobs and Embeddings successfully written to file: {filename}")
    






client = OpenAI()

job_titles_folder = "job_title_batch_files"
embeddings_folder = "job_titles_with_embeddings"

create_folder(job_titles_folder)
create_folder(embeddings_folder)

df = pd.read_csv("./Job-2024-08-04.csv")

all_job_titles = df.title.dropna().unique()

create_jsonl_for_job_titles(job_titles_array = all_job_titles)

batch_input_file_ids = create_batch_input_files(openai_client = client,
                                                job_titles_folder = job_titles_folder
                                                )

batch_ids = initialize_batch_jobs(openai_client = client,
                                  batch_input_file_ids = batch_input_file_ids
                                  )

output_file_ids = asyncio.run(monitor_multiple_jobs(client, batch_ids))  # Get the output file IDs

print(f"Retrieved Ouptut File IDs:\n{output_file_ids}")


write_enbeddings_to_file(output_file_ids = output_file_ids,
                         output_folder = embeddings_folder
                         )

job_titles_df = convert_jsonl_files_to_dataframe(job_titles_folder)
embeddings_df = convert_jsonl_files_to_dataframe(embeddings_folder)

create_job_embeddings_csv(job_titles_df, embeddings_df)
