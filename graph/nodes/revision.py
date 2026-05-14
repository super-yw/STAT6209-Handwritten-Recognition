from ..state import AgentState  
from utils import vlm, cut_img
import copy
import re
import os

def revise_mod_node(state: AgentState):
    print(f"\n🔹 Revise node (Self-Correction)... ")
    if "items" not in state["mods"] or not state["mods"]["items"]:
        return {"mods": state["mods"]}

    mods_all = copy.deepcopy(state["mods"])
    items_mods = mods_all["items"]
    col_names = list(state["columns"].values())

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

    success_flag = None
    wrong_info = []
    for i, mod in enumerate(items_mods):
        current_val = mod["moded num"]
        confidence = mod.get("confidence", 0)

        # Step 1: 概率初筛 (筛选置信度小于0.90的)
        if confidence >= 0.90:
            print(f"  ✅ Item {i + 1} highly reliable ({confidence:.4f}), skipping revise.")
            continue

        print(f"  ⚠️ Item {i+1} Low confidence ({confidence:.4f}), start auditing...")
        
        # Step 3: 多重采样与判决
        print(f"  ❌ Discrepancy. Multi-sampling...")
        # 准备小图（只有需要改的时候才抠图）
        col_idx = col_names.index(mod["json_column"])
        img_crop = cut_img(state["img_path"], mod["row_index"], col_idx, state["cut_grid"])
        
        success_flag = False
        s02 = vlm(prompt=prompt, image=img_crop, temperature=0.2, max_tokens=15)["content"]
        s07 = vlm(prompt=prompt, image=img_crop, temperature=0.7, max_tokens=15)["content"]
        print(f"Candidates: {current_val}, {s02}, {s07}.")

        judge_prompt = f"""The image may contain handwritten numbers and printed numbers.

        Identify the correct number by applying these rules in priority order:
        1. If there is a handwritten number → return the handwritten number.
        2. If there is a crossed-out mark (e.g., "x", "/", "\\") and no handwritten number → return 0.
        3. If neither applies → return the most prominent printed number.

        The 3 candidate answers are: {current_val}, {s02}, {s07}.
        Choose the one that best matches the correct number you identified.

        Output requirements:
        - Return one number only, chosen from the 3 candidates above.
        - No explanation, no extra text.
        """

        final_res = vlm(prompt=judge_prompt, image=img_crop, max_tokens=50, temperature=0.1)
        final_str = final_res["content"]
        match = re.search(r"[-+]?\d*\.\d+|\d+", final_str)
        selected_val = match.group() if match else final_str.strip()
        items_mods[i]["moded num"] = int(selected_val)
        print(f"  📝 Discrepancy resolved. Selected: {selected_val} (from candidates: {current_val}, {s02}, {s07})")
        
        wrong_info.append(i)
        state["history_mods"].append(state["mods"])
    if success_flag is None:
        success_flag = True
    mods_all["items"] = items_mods
    if state["rounds"] <= MAX_REVISON_ROUND and state["success"] != False:
        return {"mods": mods_all, 
        "success": success_flag, 
        "history_mods":state["history_mods"],
        "wrong_info": wrong_info
        }
    elif state["rounds"] <= MAX_REVISON_ROUND and state["success"] == False:
        return {"mods": mods_all, 
        "history_mods":state["history_mods"],
        "wrong_info": wrong_info
        }
    else:
        if state["success"] != False:
            return {"mods": mods_all, 
            "success": success_flag, 
            "history_mods":state["history_mods"],
            "wrong_info": wrong_info,
            }
        else:
            return {"mods": mods_all, 
            "history_mods":state["history_mods"],
            "wrong_info": wrong_info,
            }



def revise_count_node(state: AgentState):
    print(f"\n🔹 Revise node (Self-Correction)... ")
    
    prompt = "How many handwritten modifications are there in this image? Retrn the number(eg, 1) ONLY."

    confidence = state["mode_num_conf"]
    current_val = state["mod_num"]

    # Step 1: 概率初筛 (筛选置信度小于0.90的)
    if confidence >= 0.90:
        print(f"  ✅ The number is highly reliable ({confidence:.4f}), skipping revise.")

        return {"mod_num_qly": True}


    print(f"  ⚠️ The number has low confidence ({confidence:.4f}), start auditing...")
    state["wrong_num"].append(current_val)
    if state["rounds"] <= MAX_REVISON_ROUND:
        return {"mod_num_qly": False, "wrong_num":list(set(state["wrong_num"])),"success": False}



def revise_loc_node(state: AgentState):
    print(f"\n🔹 Revise node (Self-Correction)... ")

    confidence = state["loc_conf"]
    current_val = state["mods"]

    # Step 1: 概率初筛 (筛选置信度小于0.90的)
    if confidence >= 0.90:
        print(f"  ✅The number is  highly reliable ({confidence:.4f}), skipping revise.")

        return {"loc_qly": True}

    print(f"  ⚠️ The number has low confidence ({confidence:.4f}), start auditing...")
    state["wrong_loc"].append(current_val)
    if state["rounds"] <= MAX_REVISON_ROUND:
        return {"loc_qly": False,"wrong_loc":list(set(state["wrong_loc"]))}
    else:
        return {"loc_qly": False,"wrong_loc":list(set(state["wrong_loc"])),"success": False}