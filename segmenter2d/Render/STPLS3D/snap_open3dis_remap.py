"""
在 snap_open3dis.py 渲染图像、深度、相机内外参之后, 按照相机连续顺序重新命名保存的文件。

功能简介：
    - 批量重命名 STPLS3D_Open3DIS 渲染结果数据集中的多视角文件索引。
    - 按预设映射表（REMAP_INDEX）将原视角索引重映射到新的索引，并保持四位数零填充格式。
    - 支持演习（dry run）模式，仅打印计划操作以便安全检查；关闭后才实际移动文件。

适用目录结构（每个场景目录包含以下子目录）：
    scene_folder/
        ├─ color/      （彩色渲染 PNG）
        ├─ depth/      （深度图 TIF、可选 NPY、可视化 PNG）
        ├─ intrinsic/  （相机内参 NPY）
        ├─ pose/       （相机外参 NPY）
        ├─ valid_mask/ （有效像素区域 PNG）
        ├─ coverage/   （每像素点覆盖计数 PNG）
        └─ meta/       （每帧元数据 NPZ）

运行逻辑：
    1) 遍历数据集根目录下的每个场景文件夹；
    2) 对指定子目录（TARGET_SUBDIRS）中的所有文件：
       - 第一次遍历：解析“文件名前四位数字”为旧索引，根据 REMAP_INDEX 映射到新索引；
         仅替换这四位数字，保留后续后缀（如 "_color"）与扩展名（.png/.tif等），
         临时改名为 新索引前缀+原后缀+原扩展名+".tmp"，以避免命名冲突；
       - 第二次遍历：将所有以 ".tmp" 结尾的临时文件去掉 ".tmp" 后缀，完成最终重命名；
    3) dry_run=True 时仅打印“计划”，dry_run=False 时实际移动文件。

关键参数：
    - REMAP_INDEX：旧索引 -> 新索引的字典映射；
    - TARGET_SUBDIRS：需要重命名的子目录列表；
    - DATASET_ROOT_PATH：数据集根目录路径；
    - IS_DRY_RUN：是否演习模式（强烈建议先用 True 检查）。

使用示例：
    直接运行本文件：
        python snap_open3dis_remap.py
    或在代码中调用：
        from snap_open3dis_remap import remap_dataset_files
        remap_dataset_files(root_dir="/path/to/STPLS3D_Open3DIS", dry_run=False)

注意事项：
    - 文件名需以四位数字开头（如 0000.png、0007_color.png），否则将被跳过；
    - 若某个子目录不存在，将跳过并提示；
    - 建议先在小样本上 dry run 验证映射是否符合预期，再批量执行。
"""

import os
import shutil
import re

def remap_dataset_files(root_dir, dry_run=True):
    """
    根据预设的映射规则，重命名指定目录结构下的文件。

    Args:
        root_dir (str): 数据集的根目录路径。
        dry_run (bool): 是否为演习模式。如果为True，则只打印操作，不实际修改文件。
    """
    # 定义旧索引到新索引的映射关系
    # REMAP_INDEX = {
    #     0: 0,
    #     6: 1,
    #     1: 2,
    #     3: 3,
    #     5: 4,
    #     7: 5,
    #     4: 6,
    #     2: 7
    # } # 8张图片的映射关系

    REMAP_INDEX = {
        0: 0,
        10: 1,
        12: 2,
        14: 3,
        1: 4,
        3: 5,
        5: 6,
        7: 7,
        9: 8,
        15: 9,
        13: 10,
        11: 11,
        8: 12,
        6: 13,
        4: 14,
        2: 15
    } # 16张图片的映射关系

    # 定义需要处理的二级子目录名称
    TARGET_SUBDIRS = ["color", "depth", "intrinsic", "pose", "valid_mask", "coverage", "meta"]

    print(f"--- 开始处理，根目录: {root_dir} ---")
    if dry_run:
        print("--- 当前为演习模式 (Dry Run)，不会对文件进行任何修改 ---")

    # 检查根目录是否存在
    if not os.path.isdir(root_dir):
        print(f"错误：目录 '{root_dir}' 不存在。")
        return

    # 遍历根目录下的所有一级子目录 (e.g., '10_points_GTv3_19')
    for scene_folder in os.listdir(root_dir):
        scene_path = os.path.join(root_dir, scene_folder)
        if not os.path.isdir(scene_path):
            continue

        print(f"\n正在处理场景: {scene_folder}")

        # 遍历二级子目录 (color, depth, etc.)
        for sub_dir_name in TARGET_SUBDIRS:
            current_path = os.path.join(scene_path, sub_dir_name)
            if not os.path.isdir(current_path):
                print(f"  - 警告: 在 {scene_folder} 中未找到子目录 {sub_dir_name}，跳过。")
                continue
            
            print(f"  - 正在处理子目录: {sub_dir_name}")

            # --- 第一步：重命名为带 .tmp 后缀的临时文件，避免冲突 ---
            files_to_process = [f for f in os.listdir(current_path) if not f.endswith('.tmp')]
            for filename in files_to_process:
                try:
                    # 分离文件名和扩展名，并解析文件名前四位数字作为旧索引
                    basename, extension = os.path.splitext(filename)
                    match = re.match(r"^(\d{4})(.*)$", basename)
                    if not match:
                        raise ValueError("文件名不以四位数字开头")
                    old_index_str, suffix = match.group(1), match.group(2)
                    old_index = int(old_index_str)

                    # 如果当前文件索引在映射表中
                    if old_index in REMAP_INDEX:
                        new_index = REMAP_INDEX[old_index]
                        
                        # 替换前缀索引，保留后缀（如 _color）
                        new_basename = f"{new_index:04d}{suffix}"
                        
                        old_filepath = os.path.join(current_path, filename)
                        # 临时文件名，例如 0001.png -> 0002.png.tmp
                        temp_filepath = os.path.join(current_path, f"{new_basename}{extension}.tmp")

                        print(f"    [PASS 1] 计划: {filename} -> {os.path.basename(temp_filepath)}")
                        if not dry_run:
                            shutil.move(old_filepath, temp_filepath)

                except ValueError:
                    # 如果文件名不是以四位数字开头，则忽略
                    print(f"    - 警告: 文件 '{filename}' 名称不是以四位数字开头，已跳过。")
                    continue
                except Exception as e:
                    print(f"    - 错误: 处理文件 {filename} 时发生未知错误: {e}")

            # --- 第二步：去掉 .tmp 后缀，完成重命名 ---
            temp_files = [f for f in os.listdir(current_path) if f.endswith('.tmp')]
            for temp_filename in temp_files:
                temp_filepath = os.path.join(current_path, temp_filename)
                # 最终文件名，例如 0002.png.tmp -> 0002.png
                final_filename = temp_filename.replace('.tmp', '')
                final_filepath = os.path.join(current_path, final_filename)
                
                print(f"    [PASS 2] 计划: {temp_filename} -> {final_filename}")
                if not dry_run:
                    shutil.move(temp_filepath, final_filepath)

    print("\n--- 所有操作已完成 ---")


# --- 主程序入口 ---
if __name__ == "__main__":
    # *******************************************************************
    # ** 重要：请在这里设置您的数据集根目录！**
    # *******************************************************************
    DATASET_ROOT_PATH = "/data1/wangcl/dataset/open_3d/STPLS3D_Open3DIS/2D"

    # *******************************************************************
    # ** 安全开关：设置为 True 进行演习，设置为 False 以实际执行文件重命名 **
    # *******************************************************************
    # 第一次运行时，强烈建议保持为 True，检查输出是否正确
    IS_DRY_RUN =  False 

    # 调用主函数
    remap_dataset_files(root_dir=DATASET_ROOT_PATH, dry_run=IS_DRY_RUN)



# 场景名称补零脚本 (仅运行一次即可)
# import os
# import re

# # 进入目标目录
# os.chdir('/home/Data/data2/wcl/DataSet/STPLS3D/Synthetic_v3_Instance_nosample_ply_processed_all/validation')

# # 获取所有.npy文件
# files = [f for f in os.listdir('.') if f.endswith('.npy')]

# for filename in files:
#     # 使用正则表达式匹配并补零
#     new_name = re.sub(r'^(\d)_points_', r'0\1_points_', filename)
#     new_name = re.sub(r'_(\d)\.npy$', r'_0\1.npy', new_name)
    
#     if filename != new_name:
#         print(f"重命名: {filename} -> {new_name}")
#         os.rename(filename, new_name)

# print("重命名完成！")
