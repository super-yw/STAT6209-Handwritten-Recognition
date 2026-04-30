from ..state import AgentState  
from config import MAX_RETRY_NUM
from utils import vlm
import copy

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
    if state["loc_qly"] == False:
        print(f"Wrong loc: {state["wrong_loc"]}")
        prompt += f"\n\n {state["wrong_loc"]} contains some wrong answers."
    

    json_col_names = [x for x in state["columns"].values()]

    for n in range(MAX_RETRY_NUM):
        print(f"Round {n}.")
        res = vlm(prompt=prompt, img_path=state["img_path"])
        msg = res["content"]

        #get each loc
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
            mods.append({"row_index": row_no - 1, "json_column": json_col_name, "moded num": None, "confidence": None})
            print(f"mods:{mods}")

        if len(mods) <= num:
            break
        else:
            mods = []
            print(f"Wrong number of modifications found: rquired {num}, but got {len(mods)}: {mods}")

    if not mods:
        print("⚠️ Failed to get valid modifications after 5 rounds.")
        mod_ls["items"] = mods
        return {"mods": mod_ls, "uncertain": True, "loc_conf": res["confidence"], "rounds": state["rounds"] + 1}

    mod_ls["items"] = mods


    return {"mods": mod_ls, "loc_conf": res["confidence"], "rounds": state["rounds"] + 1}