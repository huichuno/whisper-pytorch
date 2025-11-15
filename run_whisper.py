import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import librosa
import time
from dotenv import load_dotenv
import os

load_dotenv()

model_id = os.getenv('MODEL_ID')

def read_wav(filepath):
    raw_speech, samplerate = librosa.load(filepath, sr=16000)
    return raw_speech

device = "xpu" if torch.xpu.is_available() else "cpu"
if torch.xpu.is_available():
    dtype = torch.bfloat16 if torch.xpu.is_bf16_supported() else torch.float16
else:
    dtype = torch.float32

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, 
    dtype=dtype, 
    low_cpu_mem_usage=True, 
    use_safetensors=True,
    
)

model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    dtype=dtype,
    device=device,
)

raw_speech = read_wav('audio/how_r_u.wav')

start = time.time()
result = pipe(raw_speech)
end = time.time()
elapsed_time = end - start

print(result["text"])
print(f"Elapsed time: {elapsed_time:.3f} seconds\n") 
