import torch
from transformers import CLIPProcessor, CLIPModel
from peft import PeftConfig, PeftModel

def load_lora_model(pretrained_lora_path):
    dtype = torch.float16   # Use fp16 for evaluation
    device = torch.device('cuda:0') # Use single GPU for evaluation
    
    peft_config = PeftConfig.from_pretrained(pretrained_lora_path)
    pretrained_clip_path = peft_config.base_model_name_or_path
    
    # Load pretrained CLIP model
    processor = CLIPProcessor.from_pretrained(pretrained_clip_path)
    model = CLIPModel.from_pretrained(pretrained_clip_path)
    tokenizer = processor.tokenizer
    image_processor = processor.image_processor

    print("loading lora...")
    model = PeftModel.from_pretrained(model, pretrained_lora_path, is_trainable=False)
    print("finished")
    
    model.to(device, dtype=dtype)
    model.requires_grad_(False)
    
    return model, processor, tokenizer, image_processor

def main():
    pretrained_lora_path = "./output/clip_ft_lora_all/pretrained"
    model, processor, tokenizer, image_processor = load_lora_model(pretrained_lora_path)
    print("Process Finished")

if __name__ == "__main__":
    main()
