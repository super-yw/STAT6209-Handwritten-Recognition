from ..state import AgentState  
from prompts import CHECK_PROMPT, FIND_ADDON_MODIF_PROMPT
from config import MAX_RETRY_NUM
from utils import vlm
import cv2
import copy
import re
import os
import json
import ast

def pre_process_node(state: AgentState):
    print(f"\n🔹 Pre process node...")
    mod_ls = copy.deepcopy(state["mods"])
    img = cv2.imread(state["img_path"])
    img_height, img_width = img.shape[:2]
    # Use VLM to check whether extra structure exists, and get the coordinates if exists.
    check_prompt = CHECK_PROMPT
    for n in range(MAX_RETRY_NUM):
        print(f"Round {n}.")
        res = vlm(prompt=check_prompt, image=img)
        msg = res["content"]
        try:
            result = json.loads(msg)
            break
        except ValueError:
            print(f"VLM output not required: {msg}")

    if result["has_extra_section"] and result["bbox_relative"]:
        x_min_r, y_min_r, x_max_r, y_max_r = result["bbox_relative"]
        
        # 换算为像素坐标
        x_min_px = int(x_min_r * img_width)
        y_min_px = int(y_min_r * img_height)
        x_max_px = int(x_max_r * img_width)
        y_max_px = int(y_max_r * img_height)
        
        print(f"Pixel Coordinates: [{x_min_px}, {y_min_px}, {x_max_px}, {y_max_px}]")
        # 输出示例: 像素坐标: [16, 864, 776, 1116]

        extra_img = img[y_min_px:y_max_px, x_min_px:x_max_px]

        prompt = FIND_ADDON_MODIF_PROMPT

        for n in range(MAX_RETRY_NUM):
            print(f"Round {n}.")
            res = vlm(prompt=prompt, image=extra_img)
            msg = res["content"]
            try:
                mod_btm = ast.literal_eval(msg)
                if not mod_btm:
                    break
                else:
                    for mod_i in mod_btm:
                        item = mod_i["item name"]
                        value = mod_i["value"]
                        mod_ls[item] = value
                    break
            except ValueError:
                print(f"VLM output not required: {msg}")
        
        new_img = img[0:y_min_px, :]
        out_dir = "pre-process_result"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            print(f"New directory has been built: {out_dir}")
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
        return {"mods": mod_ls, "img_path": save_path}

    else:
        return