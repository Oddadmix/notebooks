from datasets import load_dataset , Audio 
import os

ds = load_dataset("oddadmix/tts-elda7ee7-16p-transcribed", split="train")

TARGET_SR = 24000
ds = ds.cast_column("audio", Audio(sampling_rate=TARGET_SR))


import hashlib

def sha256_hash_text(text):

  # Encode the text to bytes (UTF-8 is standard)
    text_bytes = text.encode('utf-8')

  # Create a new SHA256 hash object
    hash_object = hashlib.sha256(text_bytes)

  # Get the hexadecimal representation of the hash
    hash_hex = hash_object.hexdigest()

    return hash_hex


import torch
import torchaudio
from sam_audio import SAMAudio, SAMAudioProcessor

# Load model and processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "facebook/sam-audio-small"
#model_id = "facebook/sam-audio-base"
model = SAMAudio.from_pretrained(model_id).to(device).eval()
processor = SAMAudioProcessor.from_pretrained(model_id)

output_folder = "elda7ee7_sep"

def seperate_voice(path, output):

    if os.path.isfile(output_folder + "/" +  output + "_target.wav"):
        return
    
    # Load audio file
    audio_file = path

    # Describe the sound you want to isolate
    description = "A person talking"

    # Process and separate
    inputs = processor(audios=[audio_file], descriptions=[description]).to(device)
    with torch.inference_mode():
        result = model.separate(inputs, predict_spans=True)

    # Save results
    torchaudio.save(output_folder + "/" +  output + "_target.wav", result.target[0].unsqueeze(0).cpu(), processor.audio_sampling_rate)
    torchaudio.save(output_folder + "/" +  output + "_residual.wav", result.residual[0].unsqueeze(0).cpu(), processor.audio_sampling_rate)
    
    del result
    del inputs
    
from tqdm import tqdm
import time
import soundfile as sf

for i in tqdm(range(len(ds))):
    example = ds[i]
    file_path = output_folder + "/" + sha256_hash_text(example["transcription"]) + ".wav"
    audio = example["audio"]["array"]
    sf.write(file_path, audio, TARGET_SR)
    seperate_voice(file_path, sha256_hash_text(example["transcription"]))
