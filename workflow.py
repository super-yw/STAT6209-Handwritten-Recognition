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
    columns: List[str] # 初始化给
    major_key: str # 初始化给
    cut_grid: List[List[Dict]] # 初始化给
    mod_num: Optional[int] # count_node √
    img_path: str # 初始化给
    ori_cont: Dict # 初始化给
    mods: List[Dict] # loc_node & number_node
    mod_cont: Dict # merge_node
    uncertain: Optional[bool] # 初始化, count_node, loc_node & number_node
    '''
    critique: Dict             # 还没写
    revision_round: int        # 还没写
    history_scores: List[Dict] # 还没写
    '''

# ================= TOOL: REAL ARXIV SEARCH =================
def vlm(prompt: str = None, img_path: str = None, image=None, max_tokens: int = 500) -> str:
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
    if col_id >= 4: # 更新分割后删掉
        node = cut_grid[row_id][col_id-3]
    else:
        node = cut_grid[row_id][col_id]
    x1, y1 = node["x1"], node["y1"] # 左上顶点
    x2, y2 = node["x2"], node["y2"] # 右下顶点
    crop = img[y1:y2, x1:x2]
    out_dir='crops'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print("New directory has been built.")
    # 2. 清理文件名非法字符
    img_path_clean = str(img_path).replace("/", "_").replace("\\", "_").replace(":", "_").replace(".", "_")
    save_path = f"crops/{img_path_clean}_{row_id}_{col_id}.png"
    
    # 3. 检查 crop 是否有效
    if crop is None or crop.size == 0:
        print(f"❌ crop无效, 跳过保存")
    else:
        # 4. 保存并验证结果
        success = cv2.imwrite(save_path, crop)
        if success:
            print(f"✅ 保存成功: {save_path}")
        else:
            print(f"❌ 保存失败: {save_path}")
    return crop

# ================= NODES =================
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
    print(f"\n🔹 Loc node...")
    num = state["mod_num"]
    if num is None:
        num = "uncertain"
    item_id = state["major_key"]
    prompt = f"""
    The image shows an invoice.
    There are {len(state["columns"])} columns, denoted by {state["columns"]}.
    Each row records an item, uniquely represented by {item_id}.

    There are {num} modifications in total.
    Find out which item and which column have been modified.

    The return MUST be in the format: [{{"{item_id}": [{item_id} 1], "column": [column name 1]}}, {{"{item_id}": [{item_id} 2], "column": [column name 2]}}, ...].
    
    Column names must be chosen in {state["columns"]}. Can NOT be other names.
    """
    
    valid_item_nos = [int(float(item["seq_no"])) for item in state["ori_cont"]["items"]]
    print(f"valid_item_nos:{valid_item_nos}")
    valid_col_names = [x for x in state["columns"] if x != state["major_key"]]

    for n in range(5):
        print(f"Round {n}.")
        msg = vlm(prompt=prompt, img_path=state["img_path"])

        # Step1: 用正则从字符串中提取所有 {...} 块
        raw_blocks = re.findall(r'\{[^{}]*\}', msg)
        mods = []
        for block in raw_blocks:
            # Step2: 尝试解析为字典
            try:
                d = json.loads(block)
            except json.JSONDecodeError:
                print(f"Unable to parse: {block}")
                continue
            # Step3: 验证字段存在
            print(f"item_id: {item_id}")
            print(f"d[item_id]: {d[item_id]}")
            if isinstance(d[item_id], list):
                d[item_id] = str(d[item_id][0])
            item_no = int(float(str(d[item_id]).replace(" ", "")))
            col_name  = d["column"]
            # Step4: 验证 item_id 是否在已知列表中
            if item_no not in valid_item_nos:
                print(f"Item_id invalid: {item_id} {item_no} not valid.")
                continue
            # Step5: 验证 column 是否在已知列表中
            if col_name not in valid_col_names:
                print(f"Colmn invalid: column {col_name} not valid.")
                continue
            # Step6: 拼成字典加入结果
            mods.append({item_id: item_no, "column": col_name, "moded num": None})
            print(f"mods:{mods}")

        if len(mods) == num:
            break
        else:
            mods = []
            print(f"Wrong number of modifications found: rquired {num}, but got {len(mods)}: {mods}")

    if not mods:
        print("⚠️ Failed to get valid modifications after 5 rounds.")
        return {"mods": mods, "uncertain": True}

    return {"mods": mods}


def number_node(state: AgentState):
    print(f"\n🔹 Number node...")

    item_id = state["major_key"]
    valid_item_nos = [int(float(item["seq_no"])) for item in state["ori_cont"]["items"]]
    # valid_item_nos = [item[item_id] for item in state["ori_cont"]["items"]]
    valid_col_names = state["columns"]
    mods_ls = copy.deepcopy(state["mods"])
    flag = None
    for i, mod in enumerate(mods_ls):
        try:
            row_idx = valid_item_nos.index(mod[item_id])
            col_idx = valid_col_names.index(mod["column"])
        except ValueError as e:
            raise ValueError(f"Invalid values: {mod[item_id]}, {mod['column']}")

        img = cut_img(img_path = state["img_path"], row_id=row_idx, col_id=col_idx, cut_grid=state["cut_grid"])

        prompt = """
        There are handwritten numbers and printed numbers in the picture. What's the handwritten number in the picture?
        The return must be handwritten, not printed. And it must be a real number, contents like "x", "/", "\\", "(1+2)" are not allowed.
        Return the number ONLY, no explanation.
        """

        num = None
        for n in range(5):
            print(f"Round {n}.")
            msg = vlm(prompt=prompt, image=img).strip()
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
    
    return {"mods": mods_ls, "uncertain": flag}

# 如果有 conflict node 应该放在这

def merge_node(state: AgentState):
    print(f"\n🔹 Merge node...")
    if not state["mod_num"]:
        cont = state["ori_cont"]
    else:
        item_id = state["major_key"]
        valid_item_nos = [int(float(item["seq_no"])) for item in state["ori_cont"]["items"]]
        mods_ls = state["mods"]
        cont = copy.deepcopy(state["ori_cont"])
        for i, mod in enumerate(mods_ls):
            idx = valid_item_nos.index(mod[item_id])
            cont["items"][idx][mod["column"]] = mod["moded num"]

    return {"mod_cont": cont}


def check_node(state: AgentState):
    print(f"\n🔹 Check node...")

    item_id = state["major_key"]
    valid_item_nos = [int(float(item["seq_no"])) for item in state["ori_cont"]["items"]]
    mods_ls = state["mods"]
    cont = copy.deepcopy(state["mod_cont"])
    for i, mod in enumerate(mods_ls):
        print(f"valid_item_nos:{valid_item_nos}")
        print(f"mod:{mod}")
        print(f"item_id:{item_id}")
        idx = valid_item_nos.index(mod[item_id])
        if state["inv_type"] == 0: # 还没写
            print("\n")

        elif state["inv_type"] == 1:
            if mod["column"] == "qty":
                cont["items"][idx]["qty"] = str(int(cont["items"][idx]["qty"]))
                breakdown_str = cont["items"][idx]["qty_breakdown"].strip()
                numbers = re.findall(r'\d+', breakdown_str)
                buy, free = int(numbers[0]), int(numbers[1])
                if buy + free != cont["items"][idx]["qty"]:
                    new_buy = int(cont["items"][idx]["qty"]) - free
                    new_bd_str = f"({new_buy}+{free})"
                    cont["items"][idx]["qty_breakdown"] = new_bd_str
            elif mod["column"] == "price_str":
                cont["items"][idx]["price_str"] = f"{float(cont["items"][idx]["price_str"]):.2f} E"

        else: # 还没写
            print("\n")

    return {"mod_cont": cont}
    
# ================= EDGES / LOGIC =================
def mod_or_mot(state: AgentState) -> Literal["merge", "loc"]:

    # Stop when average score > 8.5 or max rounds reached (5)
    if not state["mod_num"]:
        print("No modifications detected.")
        return "merge"
    print(f"Found {state["mod_num"]} modifications.")
    return "loc"

# ================= GRAPH BUILD =================
workflow = StateGraph(AgentState)
workflow.add_node("count", count_node)
workflow.add_node("loc", loc_node)
workflow.add_node("number", number_node)
workflow.add_node("merge", merge_node)
workflow.add_node("check", check_node)

workflow.add_edge(START, "count")
workflow.add_conditional_edges("count", mod_or_mot, {
    "merge": "merge",
    "loc": "loc"
    })
workflow.add_edge("loc", "number")
workflow.add_edge("number", "merge")
workflow.add_edge("merge", "check")
workflow.add_edge("check", END)


app = workflow.compile()

# ================= MAIN EXECUTION =================
if __name__ == "__main__":
    out_dir='pred_labels'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_dir2='pred_mod_labels'
    if not os.path.exists(out_dir2):
        os.makedirs(out_dir2)

    mod_img_path = "Invoice/marked/dsp"
    mod_img_files= sorted(os.listdir(mod_img_path))
    ori_json_path = "Invoice/labels/dsp"
    ori_json_files = sorted(os.listdir(ori_json_path))
    cut_grid_path = "seg_cells_0416.json"

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
            columns = ["material_no", "Material Description", "case_factor", "unit", "sold_qty", "del_qty"]
            major_key = "material_no"
        elif "po_no" in ori_cont.keys():
            inv_type = 1
            columns = ["Item No.", "Dscription", "case_pk", "barcode", "qty", "price_str"]
            major_key = "Item No."
        else:
            inv_type = 2
            columns = ["item_code", "DESCRIPTION", "case_factor", "sold_qty", "rep_qty", "foc_qty", "total_qty", "del_case", "del_unit", "unit_price", "total_amount"]
            major_key = "item_code"
        
        initial_state = {"inv_type": inv_type,
        "columns": columns,
        "major_key": major_key,
        "cut_grid": cut_grid,
        "img_path": img_path,
        "ori_cont": ori_cont,
        "uncertain": None}

        final_state = app.invoke(initial_state)


        pred_json_path = os.path.join(out_dir, json_path)
        with open(pred_json_path, "w") as f:
            json.dump({"invoice content": final_state["mod_cont"], "uncertain": final_state["uncertain"]}, f, indent=2)

        pred_mod_json_path = os.path.join(out_dir2, json_path)
        with open(pred_mod_json_path, "w") as f:
            json.dump(final_state["mods"], f, indent=2)
 
    print(f"\n✅ All Done!")