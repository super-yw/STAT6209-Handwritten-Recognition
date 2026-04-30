# model_loader.py
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
from config import MODEL_DIR, LORA_DIR
from download_models import download_model

# 全局变量
processor = None
model = None

def load_model():
    """下载并加载模型（含LoRA），已加载则跳过"""
    global processor, model

    if processor is not None and model is not None:
        print("模型已加载，跳过")
        return processor, model

    # Step 1: 下载模型
    download_model()

    # Step 2: 加载 processor
    print("正在加载 processor...")
    processor = AutoProcessor.from_pretrained(MODEL_DIR)

    # Step 3: 加载基础模型
    print("正在加载基础模型...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_DIR,
        dtype=torch.bfloat16,
        device_map="auto"
    )

    # Step 4: 加载 LoRA adapter
    print("正在加载 LoRA adapter...")
    model = PeftModel.from_pretrained(
        model,
        LORA_DIR
    )

    print("模型加载完成 ✓")
    return processor, model


def get_processor():
    if processor is None:
        load_model()
    return processor


def get_model():
    if model is None:
        load_model()
    return model
