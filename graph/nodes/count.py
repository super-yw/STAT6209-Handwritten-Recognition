from ..state import AgentState  
from config import MAX_RETRY_NUM
from utils import vlm
import copy
import re

def count_node(state: AgentState):
    print(f"\n🔹 Count node...")
    prompt = """How many handwritten modifications are there in this image? Retrn the number(eg, 1) ONLY."""
    if state["mod_num_qly"] == False:
        print(f"Wrong num: {state["wrong_num"]}")
        prompt += f"\n The answer is not {state["wrong_num"]}"

    num = None

    for n in range(MAX_RETRY_NUM):
        print(f"Round {n}.")
        res = vlm(prompt=prompt, img_path=state["img_path"])
        msg = res["content"].strip()
        try:
            # 使用正则提取数字，防止 int() 报错
            match = re.search(r'\d+', msg)
            # num = int(msg)
            if match:
                num = int(match.group())
                print(f"{num} modifications.")
                break
            else:
                print(f"Number < 0: {msg}")
        except ValueError:
            print(f"VLM output not number: {msg}")

    if num is None:
        print("⚠️ Failed to get valid number after 5 rounds.")
        return {"mod_num": num, "uncertain": True, "mode_num_conf": res["confidence"], "rounds": state["rounds"] + 1}
    return {"mod_num": num, "mode_num_conf": res["confidence"], "rounds": state["rounds"] + 1}
