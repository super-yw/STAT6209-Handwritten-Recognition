# ================= Inference Pipeline =================
MAX_REVISON_ROUND = 8
MAX_RETRY_NUM = 5

# ================= Path =================
MODEL_NAME = "Qwen/Qwen3-VL-4B-Instruct"   
MODEL_DIR = "./Qwen3-VL-4B-Instruct" 
LORA_DIR = "./LoRA"
OUT_DIR = "./Outputs"
INPUT_IMG_DIR = "./data/provided_img"
INPUT_JSON_DIR = "./data/provided_json"
DN_GRID_PATH = "./data/segments/dn_seg_cells.json"
DSP_GRID_PATH = "./data/segments/dsp_seg_cells.json"
INV_GRID_PATH = "./data/segments/invoice_seg_cells.json"
DN_REPRESENTATIVE_PATH = "./data/provided_img/mock_dn_po_025.png"
DSP_REPRESENTATIVE_PATH = "./data/provided_img/mock_dsp_po_033.png"
INV_REPRESENTATIVE_PATH = "./data/provided_img/mock_invoice_po_055.png"
