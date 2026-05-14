from typing import Literal
from .state import AgentState   
from config import MAX_REVISON_ROUND

def after_count(state: AgentState) -> Literal["bad qly", "no modif", "modif exist"]:
    if state["mod_num_qly"] == True:
        if not state["mod_num"]:
            print("No modifications detected.")
            return "no modif"
        print(f"Found {state["mod_num"]} modifications.")
        return "modif exist"
    else:
        if state["rounds"] <= MAX_REVISON_ROUND:
            print(f"Bad qly: {state["rounds"]}.")
            return "bad qly"
        else:
            print("TOO MANY TIMES!!!!!!!!!!!!!!!!!!")
            if not state["mod_num"]:
                print("No modifications detected.")
                return "no modif"
            print(f"Found {state["mod_num"]} modifications.")
            return "modif exist"

def after_loc(state: AgentState) -> Literal["bad qly", "good qly"]:
    if state["loc_qly"] == True:
        print("Good qly.")
        return "good qly"
    else:
        if state["rounds"] <= MAX_REVISON_ROUND:
            print("Bad qly.")
            return "bad qly"
        else:
            return "good qly"

def after_num(state: AgentState) -> Literal["bad qly", "good qly"]:
    if state["success"] == True:
        print("Good qly.")
        return "good qly"
    else:
        if state["rounds"] <= MAX_REVISON_ROUND:
            print("Bad qly.")
            return "bad qly"
        else:
            return "good qly"