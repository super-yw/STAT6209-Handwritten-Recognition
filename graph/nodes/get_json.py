from ..state import AgentState  
import copy
import re
import os

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
                
        if "MILK CRATE DEPOSIT" in state["mods"]:
            cont["transporter_information"]["milk_crate_deposit"] = state["mods"]["MILK CRATE DEPOSIT"]
        if "MILK CRATE REFUND" in state["mods"]:
            cont["transporter_information"]["milk_crate_refund"] = state["mods"]["MILK CRATE REFUND"]

    return {"mod_cont": cont}