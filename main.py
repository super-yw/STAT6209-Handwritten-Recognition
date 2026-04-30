import os
from download_models import download_model
from model_loader import load_model  
from config import MODEL_NAME, MODEL_DIR, LORA_DIR, OUT_DIR, INPUT_IMG_DIR, INPUT_JSON_DIR, DN_GRID_PATH, DSP_GRID_PATH, INV_GRID_PATH  
from generate_grid import merge_ranges, generate_grid
from graph.graph import agent
from graph.state import AgentState
import json
from huggingface_hub import snapshot_download
from pathlib import Path
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoModelForCausalLM, AutoTokenizer
import torch
import json
from PIL import Image
import re
import cv2
import copy
import os
from langgraph.graph import StateGraph, END, START
import numpy as np
import ast
import torch.nn.functional as F
from peft import PeftModel


def main():
    # Step 1: 下载并加载模型
    load_model()

    # Step 2: 检查数据
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    mod_img_files= sorted(os.listdir(INPUT_IMG_DIR))
    ori_json_files = sorted(os.listdir(INPUT_JSON_DIR))

    if not os.path.exists(DN_GRID_PATH):
        grid = generate_grid(DN_REPRESENTATIVE_PATH)
        with open(DN_GRID_PATH, 'w', encoding='utf-8') as f:
            json.dump(grid, f)
    if not os.path.exists(DSP_GRID_PATH):
        grid = generate_grid(DSP_REPRESENTATIVE_PATH)
        with open(DSP_GRID_PATH, 'w', encoding='utf-8') as f:
            json.dump(grid, f)
    if not os.path.exists(INV_GRID_PATH):
        grid = generate_grid(INV_REPRESENTATIVE_PATH)
        with open(INV_GRID_PATH, 'w', encoding='utf-8') as f:
            json.dump(grid, f)

    # Step 3:  执行业务逻辑
    for i, (img_path, json_path) in enumerate(zip(mod_img_files, ori_json_files)):
        
        print("="*30, f"{i}th data", "="*30)
        
        img_path = os.path.join(INPUT_IMG_DIR, img_path)
        json_path = os.path.join(INPUT_JSON_DIR, json_path)
        print(img_path)
        print(json_path)

        with open(json_path, 'r', encoding='utf-8') as f:
            ori_cont = json.load(f)
        if "sales_order_no" in ori_cont.keys():
            with open(DN_GRID_PATH, 'r', encoding='utf-8') as f:
                cut_grid = json.load(f)
            inv_type = 0
            columns = {"Material No.":"material_no", 
            "Material Description": "description", 
            "Case Factor": "case_factor", 
            "Unit": "unit", 
            "Sold Uni Qty": "sold_qty", 
            " Delivery Qty Unit": "del_qty"}
        elif "po_no" in ori_cont.keys():
            with open(DSP_GRID_PATH, 'r', encoding='utf-8') as f:
                cut_grid = json.load(f)
            inv_type = 1
            columns = {"Item No.": "seq_no", 
            "Description": "desc_en", 
            "Case Pk": "case_pk", 
            "TUC/Bar Code": "barcode", 
            "Prin Qty": "qty", 
            "Invoice Pri": "price_str"}
        else:
            with open(INV_GRID_PATH, 'r', encoding='utf-8') as f:
                cut_grid = json.load(f)
            inv_type = 2
            columns = {"ITEM": "item_code", 
            "DESCRIPTION": "desc_en", 
            "CASE FACTOR": "case_factor", 
            "UNIT QTY SOLD": "sold_qty", 
            "UNIT QTY REP": "rep_qty", 
            "UNIT QTY FOC": "foc_qty", 
            "UNIT QTY TOTAL": "total_qty", 
            "DELIVERY QTY CASE": "del_case", 
            "DELIVERY QTY UNIT": "del_unit", 
            "UNIT PRICE": "unit_price", 
            "TOTAL AMOUNT": "total_amount"}
        
        initial_state = {"inv_type": inv_type,
        "columns": columns,
        "cut_grid": cut_grid,
        "img_path": img_path,
        "ori_cont": ori_cont,
        "mods": {},
        "mod_num_qly":None,
        "loc_qly": None,
        "uncertain": None,
        "wrong_info": [],
        "wrong_num":[],
        "wrong_loc":[],
        "success": None,            
        "rounds": 0,        
        "history_mods": []
        }

        final_state = agent.invoke(initial_state)


        pred_json_path = os.path.join(OUT_DIR, json_path)
        os.makedirs(os.path.dirname(pred_json_path), exist_ok=True)
        print(f"{pred_json_path}")
        with open(pred_json_path, "w") as f:
            json.dump(final_state, f, indent=2)

 
    print(f"\n✅ All Done!")
    return


if __name__ == "__main__":
    main()