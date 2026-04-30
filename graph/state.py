from typing import TypedDict, List, Dict, Optional

class AgentState(TypedDict):
    inv_type: int 
    columns: Dict 
    cut_grid: List[List[Dict]] 
    mod_num: Optional[int] # count_node √
    mode_num_conf: float
    mod_num_qly: Optional[bool]
    wrong_num: List[int]
    img_path: str 
    ori_cont: Dict 
    mods: Dict # loc_node & number_node
    loc_conf: float
    loc_qly: Optional[bool]
    wrong_loc = List[Dict]
    mod_cont: Dict # merge_node
    uncertain: Optional[bool] # count_node, loc_node & number_node
    success: Optional[bool]            
    rounds: int        
    wrong_info: List[int]
    history_mods: List[Dict] 