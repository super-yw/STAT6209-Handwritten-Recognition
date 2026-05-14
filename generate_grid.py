import cv2
import numpy as np
import json
import argparse

def merge_ranges(ranges, gap_threshold=10):
    """
    将间距小于 gap_threshold 的相邻区间合并
    避免一行文字被断成多个区间
    """
    if not ranges:
        return []
    merged = [ranges[0]]
    for current in ranges[1:]:
        prev = merged[-1]
        gap = current[0] - prev[1]
        if gap <= gap_threshold:
            # 合并
            merged[-1] = (prev[0], current[1])
        else:
            merged.append(current)
    return merged


def generate_grid(img_path):
    # Step 1: 读取图像 + 灰度化
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Step 2: 二值化（文字为白，背景为黑）
    # 自适应阈值，应对光照不均
    binary = cv2.adaptiveThreshold(
        gray,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,  # 反转：线条为白，背景为黑
        blockSize=15,
        C=10
    )
    # Step 3: 水平投影 → 检测行边界
    # 每行像素求和
    h_projection = np.sum(binary, axis=1)  # shape: (height,)

    # 找有文字的行（投影值 > 阈值）
    h_threshold = np.max(h_projection) * 0.05  # 最大值的5%作为阈值
    h_mask = h_projection > h_threshold        # True=有文字，False=空白

    # 找行的起始和结束位置（从空白到有文字 = 起始，从有文字到空白 = 结束）
    row_ranges = []
    in_text = False
    start = 0
    for i, val in enumerate(h_mask):
        if val and not in_text:       # 进入文字区域
            start = i
            in_text = True
        elif not val and in_text:     # 离开文字区域
            row_ranges.append((start, i))
            in_text = False
    if in_text:                       # 处理最后一行未结束的情况
        row_ranges.append((start, len(h_mask)))

    # Step 4: 垂直投影 → 检测列边界
    # 方法：对所有文字行区域的垂直投影求和（避免行间距干扰）
    # 先将所有行区域合并到一张掩码图
    row_mask_img = np.zeros_like(binary)
    for (r_start, r_end) in row_ranges:
        row_mask_img[r_start:r_end, :] = binary[r_start:r_end, :]

    # 每列像素求和
    v_projection = np.sum(row_mask_img, axis=0)  # shape: (width,)

    # 找有文字的列（投影值 > 阈值）
    v_threshold = np.max(v_projection) * 0.05
    v_mask = v_projection > v_threshold

    # 找列的起始和结束位置
    col_ranges = []
    in_text = False
    start = 0
    for i, val in enumerate(v_mask):
        if val and not in_text:
            start = i
            in_text = True
        elif not val and in_text:
            col_ranges.append((start, i))
            in_text = False
    if in_text:
        col_ranges.append((start, len(v_mask)))

    row_ranges2 = merge_ranges(row_ranges, gap_threshold=13)
    col_ranges2 = merge_ranges(col_ranges, gap_threshold=9.8)
    # Step 5: 构建单元格网格
    cells = []
    for row_idx, (r_start, r_end) in enumerate(row_ranges2):
        row_cells = []
        for col_idx, (c_start, c_end) in enumerate(col_ranges2):
            cell = {
                'row': row_idx,
                'col': col_idx,
                'x1': c_start,
                'y1': r_start,
                'x2': c_end,
                'y2': r_end,
            }
            row_cells.append(cell)
        cells.append(row_cells)

    print(f"\nGrid: {len(cells)} Rows × {len(cells[0])} Columns.")

    return cells

def parse_args():
    parser = argparse.ArgumentParser(description="Grid generation code...")
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="The file path of the representative image of its category."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=OUTPUT_DIR,         # 默认值从 config 读
        help="The output json path."
    )
    return parser.parse_args()
    
if __name__ == "__main__":
    args = parse_args()

    print(f"Input path: {args.input}")
    print(f"Output path: {args.output}")

    grid = generate_grid(args.input)
    # 保存到文件
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(grid, f)
