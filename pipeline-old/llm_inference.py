import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os
import time
import sys

start_time = time.time()
# CUDA_VISIBLE_DEVICES=1 python /workspace/username/krishivaani_inferences/krishivaani_known/inference.py

folder_path = "/workspace/username/LM/byt5-small/model"

processing_folders = [folder_path]
for x in range(len(processing_folders)):
    print("Found model:", os.path.basename(processing_folders[x]))
    mod_ = os.path.basename(processing_folders[x])
    def load_model_and_tokenizer(model_path, tokenizer_path):
        print("Loading model and tokenizer...")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        return model, tokenizer

    def run_inference(input_csv_path, model, tokenizer):
        print(f"Loading data from {input_csv_path}...")
        print("Saving inferences to", output_csv_path)

        data_df = pd.read_csv(input_csv_path, usecols=['Ourmodel'])
        data_df['Hypothesis'] = data_df['Indicconformer Hypothesis'].fillna("").astype(str)
        
        if data_df.empty:
            print("No data found in the Hypothesis column.")
            return

        dataset = Dataset.from_pandas(data_df.rename(columns={'Ourmodel': 'input'}))
        predictions = [""] * len(data_df)  # Initialize empty predictions list

        batch_size = 16

        print("Running inference...")
        for i in range(0, len(dataset), batch_size):
            batch_indices = list(range(i, min(i + batch_size, len(dataset))))
            batch = dataset.select(batch_indices)
            input_texts = batch['input']

            # Skip empty texts and track skipped indices
            valid_inputs = [(idx, text) for idx, text in zip(batch_indices, input_texts) if isinstance(text, str) and text.strip()]
            if not valid_inputs:
                continue


            indices, texts = zip(*valid_inputs)
            input_ids = tokenizer(list(texts), return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids.to(device)
            outputs = model.generate(input_ids, max_length=512, num_beams=4, early_stopping=True)
            decoded_outputs = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

            for idx, prediction in zip(indices, decoded_outputs):
                predictions[idx] = prediction

            print(f"Inference {i} done")

        data_df['Predictions'] = predictions
        return predictions
    
model_path = f"/workspace/username/LM/byt5-small/{mod_}/model"
tokenizer_path = f"/workspace/username/LM/byt5-small/{mod_}/tokenizer"

if not os.path.exists(model_path):
    model_path = f"/workspace/username/LM/byt5-small/{mod_}"
    tokenizer_path = model_path


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model, tokenizer = load_model_and_tokenizer(model_path, tokenizer_path)
input_csv = "/workspace/username/krishivaani_inferences/krishivaani_known/broken_krishivaani_known.csv"
predics = run_inference(input_csv, model, tokenizer)

