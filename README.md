# ProteinLM

## Project Overview
ProteinLM is a simplified toolkit for protein sequence embedding generation and visualization using pre-trained language models (PLMs) from Hugging Face. This package allows researchers to easily load different protein language models, generate embeddings from protein sequences, and visualize them using dimensionality reduction techniques such as UMAP and PCA.

## Directory Structure
ProteinLM/
├── README.md                     # Usage documentation
├── requirements.txt              # Dependencies list
├── src/                          # Core code
│   ├── __init__.py
│   ├── model_loader.py           # Model loading functionality
│   ├── embedding_generator.py    # Generate embeddings from sequences
│   └── visualizer.py             # Visualization tools (UMAP, PCA)
├── data/                         # Example data
├── examples/                     # Example scripts
│   └── basic_usage.py            # Basic usage example
└── submit_example.sh             # Server execution script

## Installation
### Copy Files to Your Working Directory
First, copy the ProteinLM directory to your working directory:
```bash
cp -r /path/to/ProteinLM /your/working/directory/
```

## Usage
1. Select a Model and Configure Input (Optional)
Edit the examples/basic_usage.py file to select your preferred model and configure your input sequences:
```py
# Change the model to any of the supported models
tokenizer, model = load_model("facebook/esm2_t12_35M_UR50D")

# Change these sequences to your own protein sequences if needed
sequences = [
    "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHG",  # Hemoglobin alpha
    "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKV",  # Hemoglobin beta
    # Add more sequences as needed
]

# Change these labels to match your sequences
labels = ["Hemoglobin α", "Hemoglobin β"]
```

2. Submit the Job
Navigate to the ProteinLM directory and submit the job:
```bash
cd /your/working/directory/ProteinLM/
sbatch submit_example.sh
```

3. Monitor Your Job
Check the status of your job using:
```bash
squeue -u your_user_id
```

## Generated Files
After successful execution, the following files will be generated:
    1. Visualization files (PCA and UMAP plots in PNG format)
    2. Log file: plm_example_{JOBID}.log containing execution details

## Available Models
The following protein language models are supported:
### Encoder Models
* "Rostlab/prot_t5_xl_half_uniref50-enc"
* "Rostlab/prot_t5_xl_uniref50"
* "Rostlab/prot_t5_base_mt_uniref50"
### Masked Language Models
* "facebook/esm2_t6_8M_UR50D"
* "facebook/esm2_t12_35M_UR50D"
* "facebook/esm2_t30_150M_UR50D"
* "facebook/esm2_t33_650M_UR50D"
* "facebook/esm2_t36_3B_UR50D"
### Sequence-to-Sequence Models
* "ElnaggarLab/ankh-base"
* "ElnaggarLab/ankh-large"

## Troubleshooting
###　Path Configuration Issues
* Ensure all paths in the submission script are correctly set to reflect your environment.
* Make sure the Singularity image path is correct.

###　GPU Memory Requirements
* Large models like "facebook/esm2_t36_3B_UR50D" require significant GPU memory.
* If you encounter out-of-memory errors, consider:
    * Using a smaller model (e.g., ESM2 T12 instead of T36)
    * Reducing batch size in the embedding generator
    * Requesting a node with more GPU memory

###　Sequence Length Limitations
* Different models have different maximum sequence length limitations:
    * ProtT5 models: Usually up to 1024 tokens
    * ESM2 models: Usually up to 1024 tokens
    * ANKH models: Usually up to 512 tokens
* Longer sequences may need to be truncated or processed in segments

###　Container Issues
* If the container fails to start, check if the image exists and is accessible
* Ensure you have the necessary permissions to execute the container

## Contact Information
### Li-Zhong Guo
Email: glizhong89@gmail.com
@HHCLAB
For any issues or inquiries, please contact the author or create an issue in the repository.