# CheckGPT-v2

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Description

The official repository of "On the Detectability of ChatGPT Content: Benchmarking, Methodology, and Evaluation through the Lens of Academic Writing".

## Environment

### Environment used by the authors
- Memory: 128 GB
- Disk: 4TB M.2 SSD
- CPU: Intel Core i9-9900K Desktop Processor (8 Cores, 3.6 GHz)
- GPU: NVIDIA GeForce RTX 2080Ti (11 GB)
- OS: Ubuntu 22.04
- Python: 3.9.1 (pip 23.3.2)

### Recommended Hardware
- Disk: At least 10GB to store the models and datasets. An extra 52GB for each 50,000 samples of features (~2.2 TB in total for *./GPABench2*).
- GPU: **For CheckGPT**: 6 GB Memory (for training) or 2 GB Memory (for inference). Need to adjust the batch size accordingly. **For other benchmarked models in Sec 2.2**: 11 GB Memory.

### Package Installation
Run
```bash
pip install -r requirements.txt
```

We recommend using a virtual environment, docker, or VM to avoid version conflicts. For example, to set up a virtual environment using *virtualenv*, install it using pip:
```
pip install virtualenv
```
Navigate to a desired folder, create a virtual environment, activate it, and install our list of packages as provided:
```
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Table of Contents

- [Data](#data)
- [Features](#features)
- [Usage](#usage)


## Data
There are two versions of datasets:
1. GPABenchmark. 
2. GPABench2.
> We mainly use GPABench2 in our CCS 2024 submission.

### About the Artifact
The files are separated into several parts for upload convenience. Please download and extract all parts into the same host folder (e.g., ./artifact_checkgpt/).
- *CheckGPT.zip*: the main folder of the CheckGPT code (./artifact_checkgpt/CheckGPT). *embeddings* is the folder for saving features. *exp* is the folder for saving results under different experiment IDs.
- *CheckGPT_presaved_files.zip*: pre-trained models and saved experiments (./artifact_checkgpt/CheckGPT_presaved_files).
- *CS.zip*, *PHX.zip*, *HSS.zip*: GPABench2 datasets. Please download and extract them into a newly created folder, "GPABench2" (./artifact_checkgpt/GPABench2).
- *GPABenchmark.zip*: GPABenchmark datasets (./artifact_checkgpt/GPABenchmark).
- *scripts.zip*: scripts for reproducing the results in the paper. Extract them into the main folder (./artifact_checkgpt/CheckGPT).
- *README.md*: this file.

### Description of the Datasets
**GPABenchmark:**

- GPT example: ./GPABenchmark/CS_Task1/gpt.json *(Computer Science, Task 1 GPT-WRI)*
- HUM example: ./GPABenchmark/CS_Task1/hum.json
- Data structure: {PaperID}: {Abstract}

**GPABench2:**
- GPT example: ./GPABench2/PHX/gpt_task3_prompt4.json *(Physics, Task 3 GPT-POL, Prompt 4)*
- HUM example: ./GPABench2/PHX/ground.json
- Data structure: 
{Index}: 
{ 
{"id"}: {PaperID},
{"title"}: {PaperTitle},
{"abstract"}: {Abstract}
}

For GPABench2, download CS, PHX, and HSS, and put them under a created folder "./GPABench2". For HUM Task 2 GPT-CPL, use the second half of each text.

### Other Datasets used in this Paper:
Under *CheckGPT_presaved_files*:
- Other Academic Writing Purposes (Section 5.4) (Available under *CheckGPT_presaved_files/Additional_data/Other_purpose*)
- Classic NLP Datasets (Section 5.4) (Available under *CheckGPT_presaved_files/Additional_data/Classic_NLP*)
- Advanced Prompt Engineering (Section 5.7) (Available under *CheckGPT_presaved_files/Additional_data/Prompt_engineering*)
- Sanitized GPT Output (Section 5.10) (Available under *CheckGPT_presaved_files/Additional_data/Sanitized*)
- GPT4 (Section 5.6 )  (Available under *CheckGPT_presaved_files/Additional_data/GPT4*)

## Pre-trained Models:
Under *CheckGPT_presaved_files*.
- Models trained on GPABenchmark (v1) can be accessed at *Pretrained_models*.
- The experiments in Sections 5.2 and 5.3, including pre-trained models and training logs, can be found at *saved_experiments/basic*.

## Features
To train or reuse the text, please extract features from the text beforehand (For development only. Not necessary for testing).
### Feature Extraction
To turn text into features, use [*features.py*](CheckGPT/features.py). 
```bash
python features.py {DOMAIN} {TASK} {PROMPT}
```
Features will be saved in the folder named *embeddings*.
**ATTENTION: Each file of the saved features for 50,000 samples will be approximately 52GB.**

For example, to fetch the features of GPT data in **CS** on **Task 1 Prompt 3**:
```bash
python features.py CS 1 3 --gpt 1
```
The saved features are named in this format: *./embeddings/CS/gpt_CS_task1_prompt3.h5*

Likely, to fetch the features of HUM data in **CS** on **Task 1 Prompt 3**:
```bash
python features.py CS 1 3 --gpt 0
```
The saved features are named in this format: *./embeddings/CS/ground_CS.h5* (Same for Task 1 and 3)

For Task 2 GPT-CPL, the ground data will be cut into halves, and only the second half will be processed. An example of saved names is *ground_CS_task2.h5*.

You can also name the desired sample size. For example, to get the first 1000 samples:
```bash
python features.py CS 1 3 --gpt 0 --number 1000
```
The saved features are named in this format: *./embeddings/CS_1000/gpt_CS_task1_prompt3.h5*


## Usage
### On-the-fly
To evaluate any single piece of input text, run and follow the instructions:
```bash
python run_input.py
```

### Testing on text files
To directly evaluate any JSON data file, run:
```bash
python validate_text.py {FILE_PATH} {MODEL_PATH} {IS_GPT_OR_NOT}
```
For example, if you want to test pre-trained model *../CheckGPT_presaved_files/saved_experiments/basic/CS_Task3_Prompt2/Best_CS_Task3.pth* on *../GPABench2/CS/gpt_task3_prompt2.json* or *../GPABench2/CS/ground.json*:
```bash
python validate_text.py ../GPABench2/CS/gpt_task3_prompt2.json ../CheckGPT_presaved_files/saved_experiments/basic/CS_Task3_Prompt2/Best_CS_Task3.pth 1
```
or
```bash
python validate_text.py ../GPABench2/CS/ground.json ../CheckGPT_presaved_files/saved_experiments/basic/CS_Task3_Prompt2/Best_CS_Task3.pth 0
```

To run it on a special dataset like GPT4, run
```bash
python validate_text.py ../CheckGPT_presaved_files/Additional_data/GPT4/chatgpt_cs_task3.json ../CheckGPT_presaved_files/saved_experiments/basic/CS_Task3_Prompt2/Best_CS_Task3.pth 1
```

### Testing on pre-saved features
```bash
python dnn.py {DOMAIN} {TASK} {PROMPT} {EXP_ID} --pretrain 1 --test 1 --saved-model {MODEL_PATH}
```

To test the pretrained model *../CheckGPT_presaved_files/saved_experiments/basic/CS_Task3_Prompt2/Best_CS_Task3.pth* on pre-save features *./embeddings/CS/gpt_task3_prompt2.h5* and *./embeddings/CS/ground.h5*, run
```bash
python dnn.py CS 3 2 12345 --pretrain 1 --test 1 --saved-model ../CheckGPT_presaved_files/saved_experiments/basic/CS_Task3_Prompt2/Best_CS_Task3.pth
```

For features of small test data with 1000 samples:
```bash
python dnn.py CS_1000 3 2 12346 --pretrain 1 --test 1 --saved-model ../CheckGPT_presaved_files/saved_experiments/basic/CS_Task3_Prompt2/Best_CS_Task3.pth
```

### Training on pre-saved features
```bash
python dnn.py {DOMAIN} {TASK} {PROMPT} {EXP_ID}
```

To train a model from scratch on CS Task 3 Prompt 2:
```bash
python dnn.py CS 3 2 12347
```

**Ablation Study:** use --modelid to use a different model (0 for CheckGPT, 1 for RCH, 2 for MLP-Pool, 3 for CNN):

```bash
python dnn.py CS 3 2 12348 --modelid 1
python dnn.py CS 3 2 12349 --modelid 2
python dnn.py CS 3 2 12350 --modelid 3
```

### Transfer Learning
```bash
python dnn.py {DOMAIN} {TASK} {PROMPT} {EXP_ID} --trans 1 --mdomain --mtask --mprompt --mid
```
At the beginning, it will also provide cross-validation (testing) result.

For example, to transfer from CS_Task3_Prompt1 to HSS_Task1_Prompt2, run:
```bash
python dnn.py HSS 1 2 12351 --trans 1 --mdomain CS --mtask 3 --mprompt 1 --mid 12346
python dnn.py HSS_500 1 2 12352 --trans 1 --mdomain CS_500 --mtask 3 --mprompt 1 --mid 12346
```
--mid indicates the pre-trained model in previous experiments (e.g., 12346 as we did above).

## Reproducing the Results
To reproduce the results in the paper, please clone the scripts under *CheckGPT* and follow the steps below. Results will be saved in *./exp*.
### Section 5.2: 
To reproduce the results by training the models from scratch:
```bash
bash features_GPABench2_whole.sh
bash Sec5_2_fromscratch.sh
```
One example included in features_GPABench2_whole.sh is to extract features for the whole human data and GPT data in CS Task 1 Prompt 1:
```bash
python features.py CS 1 1 --gpt 0
python features.py CS 1 1 --gpt 1
```
"--gpt 0" indicates human data, which need only be processed once. The corresponding features will be saved in *./embeddings/CS/ground_CS.h5* and *./embeddings/CS/gpt_CS_task1_prompt1.h5*.
Next, one example included in Sec5_2_fromscratch.sh is to train a binary classification model from scratch on CS Task 1 Prompt 1:

```bash
python dnn.py CS 1 1 CS_Task1_Prompt1
```
The records and models will be saved in *./exp/CS_Task1_Prompt1*. *CS_Task1_Prompt1* is the experiment ID.

To validate the pre-trained models in a lightweight version with 5000 samples:
```bash
bash features_GPABench2_small.sh
bash Sec5_2_smalltest.sh
```
One example included in features_GPABench2_small.sh is to extract features for the first 5000 samples of human data and GPT data in CS Task 1 Prompt 1:
```bash
python features.py CS_5000 1 1 --gpt 0 --number 5000
python features.py CS_5000 1 1 --gpt 1 --number 5000
```
The corresponding features will be saved in *./embeddings/CS_5000/ground_CS_task1.h5* and *./embeddings/CS_5000/gpt_CS_task1_prompt1.h5*.
Next, one example included in Sec5_2_smalltest.sh is to validate the pre-trained model on the small test data:
```bash
python dnn.py CS_5000 1 1 CS_Task1_Prompt1_test5000 --pretrain 1 --test 1 --saved-model ../CheckGPT_presaved_files/saved_experiments/basic/CS_Task1_Prompt1/Best_CS_Task1.pth
```
which means testing the pre-trained model *CS_Task1_Prompt1/Best_CS_Task1.pth* on the small test data with 5000 testing samples.
The results will be saved in *./exp/CS_Task1_Prompt1_test5000*. *CS_Task1_Prompt1_test5000* is the experiment ID.

### Section 5.3
To run transferability experiments, run (To use pre-trained models, modify line 391 in *dnn.py* to load from *../CheckGPT_presaved_files/saved_experiments/basic/* instead of *./exp*. Saved model names might be slightly different.):
```bash
bash Sec5_3_transferability.sh
```
The format is like
```bash
python dnn.py ${tgt_domain}_5000 ${tgt_group} 1 src_${src_domain}${src_group}_tgt_${tgt_domain}${tgt_group} --trans 1 --mdomain ${src_domain} --mtask ${task} --mprompt 1 --mid ${src_domain}_Task${task}_Prompt1 --dataamount 2000
```
One example included in Sec5_3_transferability.sh is to transfer from CS Task 1 Prompt 1 to HSS Task 1 Prompt 2:
```bash
python dnn.py HSS_5000 3 1 src_CS1_tgt_HSS3 --trans 1 --mdomain CS --mtask 1 --mprompt 1 --mid CS_Task1_Prompt1 --dataamount 2000
```
which means to transfer from CS Task 1 Prompt 1 to HSS Task 1 Prompt 2 with 2000 tuning samples. The results will be saved in *./exp/src_CS1_tgt_HSS3*. *src_CS1_tgt_HSS3* is the experiment ID.

To train unified classifiers, run:
```bash
bash Sec5_3_unified_fromscratch.sh
```
One example included in Sec5_3_unified_fromscratch.sh is to train unified classifiers from scratch:
```bash
python dnn.py CS 1 0 CS_Task1_Prompt1234
```
*0* indicates all the prompts. The records and models will be saved in *./exp/CS_Task1_Prompt1234, where CS_Task1_Prompt1234 is the experiment ID.

To validate the pre-trained unified classifiers, run:
```bash
bash Sec5_3_unified_smalltest.sh
```
One example included in Sec5_3_unified_smalltest.sh is to validate the pre-trained unified classifiers on the small test data:
```bash
python dnn.py CS_5000 1 0 CS_Task1_Prompt1234_test5000 --pretrain 1 --test 1 --saved-model ../CheckGPT_presaved_files/saved_experiments/basic/CS_Task1_Prompt1234/Best_CS_Task1.pth
```
which means to test the pre-trained unified classifiers *CS_Task1_Prompt1234/Best_CS_Task1.pth* on the small test data with 5000 testing samples.
The results will be saved in *./exp/CS_Task1_Prompt1234_test5000*. *CS_Task1_Prompt1234_test5000* is the experiment ID.

### Section 5.4
Validate the pre-trained models on new domains:
```bash
bash Sec5_4.sh
```
The script will create a new folder, ./exp/Sec5_4, at the beginning and save the results in that Folder.
One example included in Sec5_4.sh is to validate the pre-trained models on new domains:

```bash
python validate_text.py ../CheckGPT_presaved_files/Additional_data/Other_purpose/CS/chatgpt_cs_task1_prompt1.json ../CheckGPT_presaved_files/saved_experiments/basic/ALL_Task123_Prompt1234/Best_ALL_Task0.pth 1 -v1 1 --number 100 > ./exp/Sec5_4/Other_purpose_CS_Task1.log
```
which means to test the pre-trained model *ALL_Task123_Prompt1234/Best_ALL_Task0.pth* on the "other academic writing purpose" data *chatgpt_cs_task1_prompt1.json* with 100 testing samples. The results will be saved in *./exp/Sec5_4/Other_purpose_CS_Task1.log*.
Similarly, you can test the pre-trained models on other datasets, including the ASAP (Students' Essays), BBC (News), and DBpedia (Wikipedia).

### Section 5.6
Validate the pre-trained models on new LLMs:
```bash
bash Sec5_6.sh
```
The script will create a new folder, ./exp/Sec5_6, at the beginning and save the results in that Folder.
One example included in Sec5_6.sh is to validate the pre-trained models on new LLMs:

```bash
python validate_text.py ../CheckGPT_presaved_files/Additional_data/GPT4/chatgpt_CS_task1.json ../CheckGPT_presaved_files/saved_experiments/basic/ALL_Task123_Prompt1234/Best_ALL_Task0.pth 1 > ./exp/Sec5_6/GPT4_CS_Task1.log
```
which means to test the pre-trained model *ALL_Task123_Prompt1234/Best_ALL_Task0.pth* on the GPT4 data *chatgpt_CS_task1.json*. The results will be saved in *./exp/Sec5_6/GPT4_CS_Task1.log*.

### Section 5.7
This is the script for 10 different prompt engineering techniques. Validate the pre-trained models on advanced prompt engineering:
```bash
bash Sec5_7.sh
```
The script will create a new folder, ./exp/Sec5_7, at the beginning and save the results in that Folder.
One example included in Sec5_7.sh is to validate the pre-trained models on the first kind of advanced prompt engineering (Chat-prompt-gen):

```bash
python validate_text.py ../CheckGPT_presaved_files/Additional_data/Prompt_engineering/01-Chat-prompt-gen/chatgpt_CS_task1_prompt1.json ../CheckGPT_presaved_files/saved_experiments/basic/ALL_Task123_Prompt1234/Best_ALL_Task0.pth 1 > ./exp/Sec5_7/Chat-prompt-gen_CS_Task1.log
```
which means to test the pre-trained model *ALL_Task123_Prompt1234/Best_ALL_Task0.pth* on the "Chat-prompt-gen" data *chatgpt_CS_task1_prompt1.json*. The results will be saved in *./exp/Sec5_7/Chat-prompt-gen_CS_Task1.log*.

### Section 5.10
Validate the pre-trained models on sanitized inputs:
```bash
bash Sec5_10.sh
```
The script will create a new folder, ./exp/Sec5_10, at the beginning and save the results in that folder.
One example included in Sec5_10.sh is to validate the pre-trained models on the second attack type (PromptEng):
```bash
python validate_text.py ../CheckGPT_presaved_files/Additional_data/Sanitized/02-PromptEng/CS/chatgpt_cs_task1_prompt1.json ../CheckGPT_presaved_files/saved_experiments/basic/ALL_Task123_Prompt1234/Best_ALL_Task0.pth 1 > ./exp/Sec5_10/PromptEng_CS_Task1_Prompt1.log
```
which means to test the pre-trained model *ALL_Task123_Prompt1234/Best_ALL_Task0.pth* on the "PromptEng" data *chatgpt_cs_task1_prompt1.json*. The results will be saved in *./exp/Sec5_10/PromptEng_CS_Task1_Prompt1.log*.

### Section 5.4 SOTA datasets & Section 5.5 ChatLog & Section 5.6 non-GPT LLMs
- Download the corresponding datasets. (See "Reference" below).
- Reformat them to the data structure of GPABench2. Save them under *./GPABench2*.
- Use the same commands as above for training, tuning, and inference.

## Reference
### SOTA Datasets (Section 5.4) & non-GPT LLMs (Section 5.6)
- [ArguGPT](https://github.com/huhailinguist/ArguGPT)
- [HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3)
- [M4](https://github.com/mbzuai-nlp/M4)
- [MULTITuDE](https://zenodo.org/records/10013755)
- [MGTBench](https://github.com/xinleihe/MGTBench)

### CheckGPT Performance Over Time (Section 5.5)
- [ChatLog-HC3](https://github.com/THU-KEG/ChatLog)