import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "models/EDIATH-merged"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔥 Using device: {device}")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto",
    trust_remote_code=True
)

model.eval()

def evaluate(prompt):

    formatted_prompt = (
        "### Instruction:\n"
        f"{prompt}\n\n"
        "### Response:\n"
    )

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.6,
            top_p=0.9,
            do_sample=True
        )

    response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    print("\n=== EDIATH RESPONSE ===\n")
    print(response)


if __name__ == "__main__":
    while True:
        prompt = input("\nEnter test prompt (or 'exit'): ")
        if prompt.lower() == "exit":
            break
        evaluate(prompt)