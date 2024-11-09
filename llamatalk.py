import torch
from transformers import LlamaForCausalLM, AutoTokenizer

# Load model and tokenizer (ensure path_to_llama_model points to your downloaded model path)
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

# Enable mixed-precision on Apple Silicon if available
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device).half()  # Using FP16 for memory efficiency

# Prepare input
input_text = "Who is kunal?"
inputs = tokenizer(input_text, return_tensors="pt").to(device)

# Generate output (change parameters if needed)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=50,
        do_sample=True,
        temperature=0.7
    )

# Decode output
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated text:", output_text)
