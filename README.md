## OpenAI Batch API: Efficient Text Embedding Generation

### Overview
This repository contains code and instructions for generating large-scale text embeddings using OpenAI's Batch API. The Batch API allows you to bypass rate limits and reduce API costs when processing large datasets, though it may introduce slight delays in response times.

By following the steps in this repository, you'll learn how to:

- Prepare text data for batch processing
- Send requests to OpenAI’s Batch API
- Retrieve and store embeddings for further analysis

### Key Features
- Rate Limit Handling: Bypass OpenAI’s API rate limits by using batch requests.
- Cost Efficiency: Save on API costs by processing requests in bulk at half the regular price.
- Scalable Solution: Process thousands of text entries in a single API call with ease.

### Getting Started
#### Prerequisites
To use this project, you’ll need:

**For .py file**
- `Python 3.x`
- `OpenAI API Key`
- `Pandas`
- `Numpy`
- `OpenAI`
- `Python-dotenv`
- `Asyncio`

**For .ipynb file**
- `Python 3.x`
- `OpenAI API Key`
- `Pandas`
- `OpenAI`
- `Python-dotenv`


#### Installation
1. Clone the repository:

```Python 
git clone https://github.com/Olujare-Dada/Articles.git
cd Articles
```

2. See this [article](https://medium.com/p/c9cd5f8a1961/edit) based on the code on this GitHub repository for the rest.


### Project Structure (for .ipynb File only)
`Generating_Batch_Embeddings/`<br>
│<br>
├──  `movies.csv`                            # Example dataset used for generating embeddings<br>
│<br>
├── `movie_genre_batch_files/`<br>
│   └── `movie_genre_batch_1.jsonl`          # Batch files in JSON Lines format for OpenAI API<br>
│<br>
├── `gpt_output.jsonl`                       # API response containing the embeddings<br>
│<br>
├── `.env`                                   # Environment variables<br>
├── `Generating_Batch_Embeddings.ipynb`      # Main Notebook to generate and send batch requests<br>
├── `requirements.txt`                       # Python dependencies<br>
└── `README.md`                              # Project README<br>


### Usage (for .ipynb File only)
#### Step 1: Prepare Your Dataset
Place your text dataset (e.g., `movies.csv`) inside the `root` folder. The dataset should have a column that contains the text you want to generate embeddings for.

#### Step 2: Run the Script
To generate text embeddings in batches, run the file `Generating_Batch_Embeddings.ipynb`
The script will:
1. Read your dataset.
2. Prepare the text data for batch processing.
3. Submit batches to OpenAI’s Batch API.
4. Save the results (embeddings) to the `root` folder.

#### Step 3: Monitor Batch Status
The script automatically checks the status of the batch and retrieves the embeddings once the job is complete. Results will be stored in the results/ directory as a .jsonl file.

#### Example
Here’s a snippet of how your input batch file (movie_genre_batch_1.jsonl) might look:
```json
{"custom_id": "request-1", "method": "POST", "url": "/v1/embeddings", "body": {"model": "text-embedding-ada-002", "input": "Adventure|Animation|Children"}}
{"custom_id": "request-2", "method": "POST", "url": "/v1/embeddings", "body": {"model": "text-embedding-ada-002", "input": "Comedy"}}
```
And your output file (`gpt_output.jsonl`) might contain:
```json
{"custom_id": "request-1", "embedding": [0.0012, -0.234, 0.023, ...]}
{"custom_id": "request-2", "embedding": [0.035, -0.001, 0.563, ...]}
```

### Contributing
Feel free to open issues or submit pull requests if you'd like to contribute to this project.
