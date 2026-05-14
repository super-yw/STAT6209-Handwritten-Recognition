'''
04.16: 【model是没微调的Qwen3-VL-4B-Instruct】
    1. ================= STATE DEFINITION =================
        critique: Dict             # 还没写
        revision_round: int        # 还没写
        history_scores: List[Dict] # 还没写

    2. ================= TOOL: REAL ARXIV SEARCH =================
        def cut_img(): 因为seg_cells_0416.json的seg列数和state["columns"]没对齐，所以此处是只针对dsp的一个特殊版本。
        待更新seg_cells.json & 更新后删掉if语句块。
        目前为了方便观察cut的效果，保存每个裁出来的图， cut_grid测试稳定后可删掉cv2.imwrite().

    3. ================= NODES =================
        def chck_node(): 只写了dsp的验证修改。
        待补全dn & invoice 的验证修改。
**************************************************************************************************************************
**************************************************************************************************************************
04.19: 【model是没微调的Qwen3-VL-4B-Instruct】
    1. 04.16 【2&3】已解决

    1. ================= NODES =================
        def get_json_node(): dn未修改的image对应的label中缺后半部分的信息，所以没法在ori_cont基础上改"milk_crate_deposit" & "milk_crate_refund".
        暂时注释掉了部分代码
        if "MILK CRATE DEPOSIT" in state["mods"]:
            cont["transporter_information"]["milk_crate_deposit"] = state["mods"]["MILK CRATE DEPOSIT"]
        elif "MILK CRATE REFUND" in state["mods"]:
            cont["transporter_information"]["milk_crate_refund"] = state["mods"]["MILK CRATE REFUND"]
    2. ================= NODES =================
        revse node还没拼过来

'''

from huggingface_hub import snapshot_download
from pathlib import Path
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
import json
from PIL import Image
from typing import TypedDict, List, Dict, Literal, Optional
import re
import cv2
import copy
import os
from langgraph.graph import StateGraph, END, START
import numpy as np
import ast

# ================= CONFIGURATION =================
model_path = Path(r"Qwen3-VL-4B-Instruct0")
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    dtype=torch.bfloat16, 
    device_map="auto" 
)


# ================= STATE DEFINITION =================
class AgentState(TypedDict):
    inv_type: int # 初始化给
    columns: Dict # 初始化给
    cut_grid: List[List[Dict]] # 初始化给
    mod_num: Optional[int] # count_node √
    img_path: str # 初始化给
    ori_cont: Dict # 初始化给
    mods: Dict # loc_node & number_node
    mod_cont: Dict # merge_node
    uncertain: Optional[bool] # 初始化, count_node, loc_node & number_node
    '''
    critique: Dict             # 还没写
    revision_round: int        # 还没写
    history_scores: List[Dict] # 还没写
    '''

# ================= TOOL: REAL ARXIV SEARCH =================
def vlm(prompt: str = None, img_path: str = None, image=None, max_tokens: int = 500, temperature = 0.1) -> str:
    if image is not None: # 直接传入了图像对象（PIL Image 或 numpy array）
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

    outputs = model.generate(**inputs, max_new_tokens=max_tokens)
    res = processor.decode(outputs[0])

    start_tag = "<|im_start|>assistant\n"
    end_tag = "<|im_end|>"
    start_idx = res.rfind(start_tag)           # rfind 从右边找，定位最后一个
    content_start = start_idx + len(start_tag)  # 跳过标签本身
    end_idx = res.find(end_tag, content_start) # 从content_start往后找第一个结束标签
    content = res[content_start:end_idx]

    return content


def cut_img(img_path, row_id, col_id, cut_grid):
    img = cv2.imread(img_path)
    print(f"row_id:{row_id}; col_id:{col_id}")
    node = cut_grid[row_id][col_id]
    x1, y1 = node["x1"], node["y1"] # 左上顶点
    x2, y2 = node["x2"], node["y2"] # 右下顶点
    crop = img[y1:y2, x1:x2]
    out_dir='crops'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print("New directory has been built.")

    img_path_clean = str(img_path).replace("/", "_").replace("\\", "_").replace(":", "_").replace(".", "_")
    save_path = f"crops/{img_path_clean}_{row_id}_{col_id}.png"

    if crop is None or crop.size == 0:
        print(f"❌ crop invalid, not save.")
    else:
        success = cv2.imwrite(save_path, crop)
        if success:
            print(f"✅ Successfully saved: {save_path}")
        else:
            print(f"❌ Failed to save: {save_path}")
    return crop

# ================= NODES =================
def pre_process_node(state: AgentState):
    print(f"\n🔹 Pre process node...")
    mod_ls = copy.deepcopy(state["mods"])
    img = cv2.imread(img_path)
    x1 = 450; y1 = 1510 # 左上顶点
    x2 = 1120; y2 = 1880 # 右下顶点
    bottom = img[y1:y2, x1:x2]

    prompt = """The image shows an invoice.
    There are 2 items, their names are: "MILK CRATE DEPOSIT" and "MILK CRATE REFUND".

    Find out which item has a handwritten number value (on the right of the item name).

    The return MUST be in the format: [{"item name": [value]}, {"item name": [value]}].
    
    If neither has a handwritten value, return [].

    Follow the following steps:
    1. Find out how many handwritten numbers are in the picture. If no handwritten numbers, skip the following steps and return [] derectly.
    2. Indentify what are these handwritten numbers.
    3. Identify which item each handwritten number belongs to. The upper handwritten number should belong to "MILK CRATE DEPOSIT", and he lower number belongs to "MILK CRATE REFUND".
    4. Return in the required format: [{"item name": [value]}, {"item name": [value]}].
    """

    for n in range(5):
        print(f"Round {n}.")
        msg = vlm(prompt=prompt, image=bottom)
        try:
            mod_btm = ast.literal_eval(msg)
            if not mod_btm:
                break
            else:
                for mod_i in mod_btm:
                    item = mod_i["item name"]
                    value = mod_i["value"]
                    mod_ls[item] = value
                print(f"mod_cont (with bottom): {mod_ls}")
                break
        except ValueError:
            print(f"VLM output not required: {msg}")
    
    new_img = img[0:y1, :]
    out_dir='dn_top'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print("New directory has been built.")
    img_path_clean = str(state["img_path"]).replace("/", "_").replace("\\", "_").replace(":", "_").replace(".", "_")
    save_path = f"{out_dir}/{img_path_clean}.png"
    if new_img is None or new_img.size == 0:
        print(f"❌ top img invalid, not save.")
    else:
        success = cv2.imwrite(save_path, new_img)
        if success:
            print(f"✅ Successfully saved: {save_path}")
        else:
            print(f"❌ Failed to save: {save_path}")
    
    return {"mod_cont": mod_ls, "img_path": save_path}


def count_node(state: AgentState):
    print(f"\n🔹 Count node...")
    prompt = """How many handwritten modifications are there in ths image? Retrn the number(eg, 1) ONLY."""

    num = None

    for n in range(5):
        print(f"Round {n}.")
        msg = vlm(prompt=prompt, img_path=state["img_path"]).strip()
        try:
            num = int(msg)
            if num >= 0:
                print(f"{num} modifications.")
                break
            else:
                print(f"Number < 0: {msg}")
        except ValueError:
            print(f"VLM output not number: {msg}")

    if num is None:
        print("⚠️ Failed to get valid number after 5 rounds.")
        return {"mod_num": num, "uncertain": True}

    return {"mod_num": num}


def loc_node(state: AgentState):
    mod_ls = copy.deepcopy(state["mods"])
    print(f"\n🔹 Loc node...")
    num = state["mod_num"]
    if num is None:
        num = "uncertain"

    prompt = f"""
    The image shows an invoice.
    There are {len(state["columns"])} columns, denoted by {list(state["columns"].keys())}.
    Each row records an item, uniquelly represented by an integer showing which row it is(row number starts from 1. eg, the first row: 1, the second row: 2).

    There are {num} modifications in total.
    Find out which item and which column have been modified.

    The return MUST in the format: [{{"row number": [row number 1], "column": [column name 1]}}, {{"row number": [row number 1], "column": [column name 2]}},...].

    Column names must be chosen in {list(state["columns"].keys())}. Can NOT be other names.
    """
    

    json_col_names = [x for x in state["columns"].values()]

    for n in range(5):
        print(f"Round {n}.")
        msg = vlm(prompt=prompt, img_path=state["img_path"])
        #et each loc
        raw_blocks = re.findall(r'\{[^{}]*\}', msg)
        mods = []
        for block in raw_blocks:
            try:
                d = json.loads(block)
            except json.JSONDecodeError:
                print(f"Unable to parse: {block}")
                continue
            # check validation for each dictionary
            print(f"d['row number']: {d['row number']}")
            if isinstance(d['row number'], list):
                d['row number'] = str(d['row number'][0])
            row_no = int(float(str(d['row number']).replace(" ", "")))

            print(f"d['column']: {d['column']}")
            if isinstance(d["column"], list):
                d["column"] = str(d["column"][0])
            col_name = str(d["column"])
            
            # check row_no
            if row_no > len(state["ori_cont"]["items"]) or row_no <= 0:
                print(f"Row number invalid: {row_no}.")
                continue

            # check col_name
            if col_name not in state["columns"].keys():
                print(f"Colmn invalid: column {col_name} not valid.")
                continue

            # add to the result
            json_col_name = state["columns"][col_name]
            mods.append({"row_index": row_no - 1, "json_column": json_col_name, "moded num": None})
            print(f"mods:{mods}")

        if len(mods) == num:
            break
        else:
            mods = []
            print(f"Wrong number of modifications found: rquired {num}, but got {len(mods)}: {mods}")

    if not mods:
        print("⚠️ Failed to get valid modifications after 5 rounds.")
        mod_ls["items"] = mods
        return {"mods": mod_ls, "uncertain": True}

    mod_ls["items"] = mods

    return {"mods": mod_ls}


def get_number_node(state: AgentState):
    print(f"\n🔹 Number node...")

    json_col_names = [x for x in state["columns"].values()]
    mods_lss = copy.deepcopy(state["mods"])
    mods_ls = mods_lss["items"]
    flag = None
    for i, mod in enumerate(mods_ls):
        try:
            col_idx = json_col_names.index(mod["json_column"])
        except ValueError as e:
            raise ValueError(f"Invalid values: {mod["row_index"]}, {mod['json_column']}")

        img = cut_img(img_path = state["img_path"], row_id=mod["row_index"], col_id=col_idx, cut_grid=state["cut_grid"])

        prompt = """
        There are handwritten numbers and printed numbers in the picture. What's the handwritten number in the picture?
        The return must be handwritten, not printed. And it must be a real number, contents like "x", "/", "\\", "(1+2)" are not allowed.
        Return the number ONLY, no explanation.
        """

        num = None
        for n in range(5):
            print(f"Round {n}.")
            msg = vlm(prompt=prompt, image=img).strip().replace("/", "").replace("\\", "").replace("<", "").replace(">", "")
            print(msg)
            try:
                if isinstance(ast.literal_eval(msg), list):
                    msg = str(ast.literal_eval(msg)[0])
            except SyntaxError:
                print(f"Strage msg: {msg}.")
            try:
                num = int(msg)
                break
            except ValueError:
                print(f"The modified number is not an integer: {msg}")
            try:
                num = float(msg)
                break
            except ValueError:
                print(f"The modified number is not a float number: {msg}")
        if num is None:
            print(f"⚠️ Failed to get valid number after 5 rounds: {mod}")
            flag = True
        mods_ls[i]["moded num"] = num
    
        mods_lss["items"] = mods_ls

    return {"mods": mods_lss, "uncertain": flag}

# 如果有 conflict node 应该放在这


def get_json_node(state: AgentState):
    print(f"\n🔹 Get_mod_json node...")
    if not state["mod_num"]:
        cont = state["ori_cont"]
    else:
        mods_ls = state["mods"]["items"]
        cont = copy.deepcopy(state["ori_cont"])
        # change format for "items"
        for i, mod in enumerate(mods_ls):
            idx = mod["row_index"]
            cont["items"][idx][mod["json_column"]] = mod["moded num"]
            
            if state["inv_type"] == 0: 
                if mod["json_column"] == "sold_qty":
                    cont["items"][idx]["sold_qty"] = str(int(cont["items"][idx]["sold_qty"]))
                    if int(cont["items"][idx]["sold_qty"]) < int(cont["items"][idx]["del_qty"]):
                        cont["items"][idx]["del_qty"] = cont["items"][idx]["sold_qty"]
                elif mod["json_column"] == "del_qty":
                    cont["items"][idx]["del_qty"] = str(int(cont["items"][idx]["del_qty"]))
                # 暂时不以del_qty为准【0419】

            elif state["inv_type"] == 1:
                if mod["json_column"] == "qty":
                    cont["items"][idx]["qty"] = str(int(cont["items"][idx]["qty"]))
                    breakdown_str = cont["items"][idx]["qty_breakdown"].strip()
                    numbers = re.findall(r'\d+', breakdown_str)
                    buy, free = int(numbers[0]), int(numbers[1])
                    if buy + free != cont["items"][idx]["qty"]:
                        new_buy = int(cont["items"][idx]["qty"]) - free
                        new_bd_str = f"({new_buy}+{free})"
                        cont["items"][idx]["qty_breakdown"] = new_bd_str
                elif mod["json_column"] == "price_str":
                    cont["items"][idx]["price_str"] = f"{float(str(cont["items"][idx]["price_str"]).split(" ")[0]):.2f} E"

            elif state["inv_type"] == 2:
                if mod["json_column"] == "sold_qty":
                    cont["items"][idx]["sold_qty"] = str(int(cont["items"][idx]["sold_qty"]))
                    total = int(cont["items"][idx]["sold_qty"]) + int(cont["items"][idx]["rep_qty"]) + int(cont["items"][idx]["foc_qty"])
                    if total != int(cont["items"][idx]["del_unit"]):
                        cont["items"][idx]["del_unit"] = str(total)
                elif mod["json_column"] == "total_qty":
                    cont["items"][idx]["total_qty"] = str(int(cont["items"][idx]["total_qty"]))
                elif mod["json_column"] == "del_unit":
                    cont["items"][idx]["del_unit"] = str(int(cont["items"][idx]["del_unit"]))
                # 不全但对现有数据够用【0419】
        '''
        if "MILK CRATE DEPOSIT" in state["mods"]:
            cont["transporter_information"]["milk_crate_deposit"] = state["mods"]["MILK CRATE DEPOSIT"]
        elif "MILK CRATE REFUND" in state["mods"]:
            cont["transporter_information"]["milk_crate_refund"] = state["mods"]["MILK CRATE REFUND"]
        '''
    return {"mod_cont": cont}
    
# ================= EDGES / LOGIC =================
def pre_process_or_mot(state: AgentState) -> Literal["pre process", "no pre-process"]:

    if state["inv_type"] == 0:
        print("Type 1 invoice, pre-process needed.")
        return "pre process"
    print(f"No pre-process needed.")
    return "no pre-process"

def mod_or_mot(state: AgentState) -> Literal["no modif", "modif exist"]:
    if not state["mod_num"]:
        print("No modifications detected.")
        return "no modif"
    print(f"Found {state["mod_num"]} modifications.")
    return "modif exist"

# ================= GRAPH BUILD =================
workflow = StateGraph(AgentState)
workflow.add_node("pre_process", pre_process_node)
workflow.add_node("count", count_node)
workflow.add_node("loc", loc_node)
workflow.add_node("get_number", get_number_node)
workflow.add_node("get_json", get_json_node)


workflow.add_conditional_edges(START, pre_process_or_mot, {
    "pre process": "pre_process",
    "no pre-process": "count"
    })
workflow.add_edge("pre_process", "count")
workflow.add_conditional_edges("count", mod_or_mot, {
    "no modif": "get_json",
    "modif exist": "loc"
    })
workflow.add_edge("loc", "get_number")
workflow.add_edge("get_number", "get_json")
workflow.add_edge("get_json", END)

app = workflow.compile()

# ================= MAIN EXECUTION =================
if __name__ == "__main__":
    out_dir='pred_labels'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_dir2='pred_mod_labels'
    if not os.path.exists(out_dir2):
        os.makedirs(out_dir2)

    mod_img_path = "Invoice/marked/invoice"
    mod_img_files= sorted(os.listdir(mod_img_path))
    ori_json_path = "Invoice/labels/invoice"
    ori_json_files = sorted(os.listdir(ori_json_path))
    cut_grid_path = "invoice_seg_cells.json"

    with open(cut_grid_path, 'r', encoding='utf-8') as f:
        cut_grid = json.load(f)

    for i, (img_path, json_path) in enumerate(zip(mod_img_files, ori_json_files)):
        print("="*30, f"{i}th data", "="*30)
        
        img_path = os.path.join(mod_img_path, img_path)
        json_path = os.path.join(ori_json_path, json_path)
        print(img_path)
        print(json_path)

        with open(json_path, 'r', encoding='utf-8') as f:
            ori_cont = json.load(f)
        if "sales_order_no" in ori_cont.keys():
            inv_type = 0
            columns = {"Material No.":"material_no", 
            "Material Description": "description", 
            "Case Factor": "case_factor", 
            "Unit": "unit", 
            "Sold Uni Qty": "sold_qty", 
            " Delivery Qty Unit": "del_qty"}
        elif "po_no" in ori_cont.keys():
            inv_type = 1
            columns = {"Item No.": "seq_no", 
            "Description": "desc_en", 
            "Case Pk": "case_pk", 
            "TUC/Bar Code": "barcode", 
            "Prin Qty": "qty", 
            "Invoice Pri": "price_str"}
        else:
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
        "uncertain": None}

        final_state = app.invoke(initial_state)


        pred_json_path = os.path.join(out_dir, json_path)
        with open(pred_json_path, "w") as f:
            json.dump({"invoice content": final_state["mod_cont"], "uncertain": final_state["uncertain"]}, f, indent=2)

        pred_mod_json_path = os.path.join(out_dir2, json_path)
        with open(pred_mod_json_path, "w") as f:
            json.dump(final_state["mods"], f, indent=2)
 
    print(f"\n✅ All Done!")