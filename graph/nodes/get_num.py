from ..state import AgentState  
from config import MAX_RETRY_NUM
from utils import vlm, cut_img
import copy
import re
import os

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

        prompt = """The image may contain handwritten numbers and printed numbers.
        
        Your task:
        1. If there is any handwritten number, return the handwritten number.
        2. If there is no handwritten number, return the most prominent printed number.
        3. If there is a crossed-out mark (e.g., "x", "/", "\\") but no handwritten number, return 0.
        
        Output requirements:
        - Return a number only.
        - Do not include any explanation.
        - Do not return symbols or expressions (e.g., "x", "/", "\\", "(1+2)").
        """
        if i in state["wrong_info"]:
            prompt += f"\nThe correct answer is NOT {mod["moded num"]}."

        final_num = None
        final_conf = 0.0
        for n in range(MAX_RETRY_NUM):
            print(f"Round {n}.")
            res = vlm(prompt=prompt, image=img, max_tokens=15, temperature=0.1)
            msg = res["content"].strip().replace("/", "").replace("\\", "").replace("<", "").replace(">", "").split()[0].split("+")[0]
            conf = res["confidence"]
            print(msg)
            try:
                if isinstance(ast.literal_eval(msg), list):
                    msg = str(ast.literal_eval(msg)[0])
            except SyntaxError:
                print(f"Strage msg: {msg}.")
            try:
                final_num = int(msg)
                final_conf = conf
                break
            except ValueError:
                print(f"The modified number is not an integer: {msg}")
            try:
                final_num = float(msg)
                final_conf = conf
                break
            except ValueError:
                print(f"The modified number is not a float number: {msg}")
        if final_num is None:
            print(f"⚠️ Failed to get valid number after 5 rounds: {mod}")
            flag = True

        mods_ls[i]["moded num"] = final_num
        mods_ls[i]["confidence"] = final_conf 

        mods_lss["items"] = mods_ls

    return {"mods": mods_lss, "uncertain": flag, "rounds": state["rounds"] + 1}