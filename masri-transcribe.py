from unsloth import FastModel, FastLanguageModel
import torch
from huggingface_hub import snapshot_download
from datasets import load_dataset
from tqdm import tqdm
import hashlib
import json
import os
import tempfile
import soundfile as sf

fourbit_models = [
    # 4bit dynamic quants for superior accuracy and low memory use
    "unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit",
    "unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit",
    # Pretrained models
    "unsloth/gemma-3n-E4B-unsloth-bnb-4bit",
    "unsloth/gemma-3n-E2B-unsloth-bnb-4bit",

    # Other Gemma 3 quants
    "unsloth/gemma-3-1b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-27b-it-unsloth-bnb-4bit",
] # More models at https://huggingface.co/unsloth

model, processor = FastModel.from_pretrained(
    model_name = "oddadmix/gemma-4b-egyptian-code-switching-b4-g2",
    dtype = None, # None for auto detection
    max_seq_length = 2048, # Choose any for long context!
    load_in_4bit = False,  # 4 bit quantization to reduce memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    # token = "hf_...", # use one if using gated models
)

from transformers import Gemma3nProcessor
processor = Gemma3nProcessor.from_pretrained("google/gemma-3n-E4B-it")

from transformers import TextStreamer

# Helper function for inference
def do_gemma_3n_inference(path, max_new_tokens = 1024):
    messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are an assistant that transcribes speech accurately.",
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "url": path},
                        {"type": "text", "text": "Please transcribe this audio."}
                    ]
                }
            ]
    
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,  # Must add for generation
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda")

    # Generate without streaming
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False
    )

    # Get only the newly generated tokens (skipping the prompt)
    generated_tokens = output[0][inputs["input_ids"].shape[-1]:]

    # Decode only the new tokens
    response = processor.decode(generated_tokens, skip_special_tokens=True)

    return response

FastLanguageModel.for_inference(model)

# Load dataset
print("Loading dataset...")
split="nedal_reads"
dataset = load_dataset("oddadmix/egyptian-youtube-single-speakers", split=split, streaming=True)
# print(f"Dataset loaded with {len(dataset)} samples")

# Create output folder
folder = split
os.makedirs(folder, exist_ok=True)

TARGET_SR = 16000

# Process each row with progress bar
for i, row in enumerate(tqdm(dataset, desc="Processing audio")):
    try:
        # Get audio and text
        audio_data = row['audio']
        chunk_id = row['chunk_id']
        
        # Generate MD5 hash of text
        text_hash = chunk_id#hashlib.md5(text.encode('utf-8')).hexdigest()
        json_path = os.path.join(folder, f"{chunk_id}.json")
        
        # Skip if already processed
        if os.path.exists(json_path):
            print(f"\nSkipping {text_hash} (already exists)")
            continue
        
        # Extract audio array and sampling rate
        audio_array = audio_data['array']
        sampling_rate = audio_data['sampling_rate']
        
        # Save audio to temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            temp_wav_path = tmp_file.name
        
        ## duration 
        duration = len(audio_array)/sampling_rate
        
        sf.write(temp_wav_path, audio_array, sampling_rate, format='WAV')
        
        # Run inference
        transcription = do_gemma_3n_inference(temp_wav_path)
        
        del row["audio"]
        
        # Save result to JSON
        result_data = {
            "transcription": transcription,
            **row
        }
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=4)
        
        # Clean up temporary file
        os.remove(temp_wav_path)
        
    except Exception as e:
        print(f"\nError processing sample {i}: {e}")
        # Clean up temp file if it exists
        try:
            if 'temp_wav_path' in locals():
                os.remove(temp_wav_path)
        except:
            pass

print(f"\nProcessing complete! Results saved in '{folder}/' folder")
