import os
import cv2
import numpy as np

def cut_img(img_path, row_id, col_id, cut_grid):
    img = cv2.imread(img_path)
    print(f"row_id:{row_id}; col_id:{col_id}")
    node = cut_grid[row_id][col_id]
    x1, y1 = node["x1"], node["y1"] # left-top
    x2, y2 = node["x2"], node["y2"] # right-bottom
    crop = img[y1:y2, x1:x2]
    out_dir='crops'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print("New directory has been built.")

    img_path_clean = str(img_path).replace("/", "_").replace("\\", "_").replace(":", "_").replace(".", "_")
    save_path = f"crops/{img_path_clean}_{row_id}_{col_id}.png"

    if crop is None or crop.size == 0:
        print(f"❌ crop invalid, not save.")
    else:
        success = cv2.imwrite(save_path, crop)
        if success:
            print(f"✅ Successfully saved: {save_path}")
        else:
            print(f"❌ Failed to save: {save_path}")
    return crop