import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
from model_loader import get_model, get_processor

def vlm(prompt: str = None, img_path: str = None, image=None, max_tokens: int = 500, temperature = 0.1) -> str:
    model = get_model()
    processor = get_processor()
    
    if image is not None: 
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    elif img_path is not None:
        try:
            image = Image.open(img_path)
        except Exception as e:
            raise ValueError(f"Wrong image path provided: {e}")
    else:
        raise ValueError("Must provide either img_path or image.")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image  
                },
                {"type": "text", "text": prompt}
            ]
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            temperature=temperature,
            do_sample=True if temperature > 0.1 else False
        )

    generated_ids = outputs.sequences[0][len(inputs.input_ids[0]):]
    content = processor.decode(generated_ids, skip_special_tokens=True).strip()

    # Calculate average token probability (Confidence)
    probs = [F.softmax(score, dim=-1) for score in outputs.scores]
    token_probs = [prob[0, gen_id].item() for prob, gen_id in zip(probs, generated_ids)]
    avg_confidence = sum(token_probs) / len(token_probs) if token_probs else 0.0

    return {"content": content, "confidence": avg_confidence}
