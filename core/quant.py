from datasets import load_dataset
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.utils import dispatch_for_generation

def quantize_multimodal_model(model_name: str, hf_token: str = None):
    # 1) Load model + processor
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",               # or torch.float16 if needed
        trust_remote_code=True,
        use_auth_token=hf_token
    )
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_auth_token=hf_token
    )

    # 2) Calibration dataset
    DATASET_ID = "lmms-lab/flickr30k"
    DATASET_SPLIT = "test"
    NUM_CAL_SAMPLES = 1
    MAX_SEQ_LEN = 1024

    # load the entire split, shuffle, then select the first NUM_CAL_SAMPLES
    ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
    ds = ds.shuffle(seed=42).select(range(NUM_CAL_SAMPLES))

    # 3) Preprocess: example["image"] is already a PIL.Image
        # Apply the correct chat template format
    def preprocess(example):
        # Format using the turn-based template you provided
        prompt = "<start_of_turn>user\n<|image_1|>\nWhat does this image show?\n<end_of_turn>\n"
        response = "<start_of_turn>model\n" + " ".join(example["caption"]) + "\n<end_of_turn>"
        
        return {
            "text": prompt + response,
            "images": example["image"],
        }

    ds = ds.map(preprocess)

    # 4) Tokenize
    def tokenize(sample):
        return processor(
            text=sample["text"],
            images=sample["images"],
            padding=False,
            max_length=MAX_SEQ_LEN,
            truncation=True,
        )

    ds = ds.map(tokenize, writer_batch_size=1, remove_columns=ds.column_names)

    # 5) Data collator
    def data_collator(batch):
        # oneshot expects a single-sample batch
        return {k: torch.tensor(v) for k, v in batch[0].items()}

    # 6) GPTQ recipe
    recipe = GPTQModifier(
        targets="Linear",
        scheme="W4A16",
        ignore=["lm_head", "re:model.vision_embed_tokens.*"],
    )

    # 7) Run oneshot quantization
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=MAX_SEQ_LEN,
        num_calibration_samples=NUM_CAL_SAMPLES,
        trust_remote_code_model=True,
        data_collator=data_collator,
    )

    # 8) Test and save
    print("===== SAMPLE GENERATION =====")
    dispatch_for_generation(model)

    prompt = "The future of artificial intelligence is"
    inputs = processor(text=prompt, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=50, disable_compile=True)
    print(processor.decode(out[0], skip_special_tokens=True))

    save_dir = f"{model_name.split('/')[-1]}-W4A16"
    model.save_pretrained(save_dir, save_compressed=True)
    processor.save_pretrained(save_dir)
    print(f"✅ Model quantized and saved to {save_dir}")

    return model, processor
# Usage
if __name__ == "__main__":
    import os

    model_name = "google/gemma-3n-E2b-it"
    hf_token = os.getenv("HF_TOKEN")

    model, processor = quantize_multimodal_model(model_name, hf_token)