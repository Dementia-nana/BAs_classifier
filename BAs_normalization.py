import os
import numpy as np
import pandas as pd
import glob
import re
from openpyxl import load_workbook


def extract_parameters_from_filename(filename):
    """
    从文件名中提取实际开孔与实际环糊精电流（第三段为“环糊精差值”）
    例：CA2_124.2398_73.8609.xlsx
        - 开孔幅值 = 124.2398  -> 实际开孔 opening_value = -124.2398
        - 差值幅值 = 73.8609
        - 实际环糊精电流 cd_current_value = -(124.2398 - 73.8609) = -50.3789
    """
    base_name = os.path.splitext(filename)[0]
    pattern = r'^.+\_(\d+\.?\d*)\_(\d+\.?\d*)$'
    match = re.match(pattern, base_name)

    if not match:
        return None, None

    try:
        opening_mag = float(match.group(1))  # 开孔的正幅值
        delta_mag = float(match.group(2))    # 差值的正幅值（开孔幅值 - 环糊精电流幅值）
        opening_value = -opening_mag
        cd_current_value = -(opening_mag - delta_mag)
        return opening_value, cd_current_value
    except ValueError:
        return None, None


def process_excel_files(input_dir, output_dir):
    """批量处理Excel文件，工作表名称包含实际比值与归一化因子（支持递归树状目录）"""
    STANDARD_OPENING = -120.1389  # 标准开孔
    STANDARD_CD_CURRENT = -47.3542  # 标准环糊精电流
    STANDARD_RATIO = (STANDARD_OPENING - STANDARD_CD_CURRENT) / STANDARD_OPENING  # 标准比值（未直接使用，保持一致）

    os.makedirs(output_dir, exist_ok=True)

    # === 递归搜集所有 Excel 文件 ===
    excel_files = glob.glob(os.path.join(input_dir, "**", "*.xlsx"), recursive=True)

    if not excel_files:
        print(f"在目录 {input_dir} 及其子目录中未找到任何Excel文件")
        return

    print(f"在 {input_dir} 及其子目录中找到 {len(excel_files)} 个Excel文件，开始处理...")

    for file_path in excel_files:
        try:
            file_name = os.path.basename(file_path)
            actual_opening, actual_cd_current = extract_parameters_from_filename(file_name)

            if actual_opening is None or actual_cd_current is None:
                print(f"警告: 文件名 {file_name} 格式不正确，已跳过")
                continue

            print(f"\n处理文件: {file_name}")
            print(f"提取的参数 - 实际开孔: {actual_opening:.4f}, 实际环糊精电流(由差值计算): {actual_cd_current:.4f}")

            # 计算实际比值
            actual_ratio = (actual_opening - actual_cd_current) / actual_opening
            print(f"计算的实际比值: {actual_ratio:.6f}")

            # 计算归一化因子（保持原公式）
            normalization_factor = (STANDARD_OPENING / actual_opening + STANDARD_CD_CURRENT / actual_cd_current) / 2
            print(f"计算的归一化因子: {normalization_factor:.6f}")

            # 读取Excel文件
            df = pd.read_excel(file_path)

            # ★★ 现在 KDE 输出是 9 列（多了“精修事件编号”和“原始区间索引”）★★
            if len(df.columns) != 9:
                print(f"警告: 文件 {file_name} 列数不是9列，已跳过")
                continue

            # 数据处理（结构与之前相同，只是列索引整体右移一列）
            new_df = pd.DataFrame()

            # 前两列保持原样：精修事件编号 & 原始区间索引
            new_df[df.columns[0]] = df.iloc[:, 0]
            new_df[df.columns[1]] = df.iloc[:, 1]

            # ★ 把“峰个数”这一列也保留下来（第 2 列）★
            new_df[df.columns[2]] = df.iloc[:, 2]

            # 计算DT列（事件持续时间 lg 值）——现在在第 4 列（索引 3）
            duration = df.iloc[:, 3].abs()
            new_df["DT"] = np.log10(duration.where(duration > 0, np.nan))

            # 计算 M1 和 S1 —— mean1 在索引 4，SD1 在索引 5
            mean1 = df.iloc[:, 4]
            new_df["M1"] = ((actual_cd_current - mean1) / actual_cd_current)
            new_df["S1"] = df.iloc[:, 5]

            # 计算 M2 和 S2 —— mean2 在索引 6，SD2 在索引 7
            mean2 = df.iloc[:, 6]
            new_df["M2"] = ((actual_cd_current - mean2) / actual_cd_current)
            new_df["S2"] = df.iloc[:, 7]

            # 最后一列 IsReal 保持不变（现在是索引 8）
            new_df[df.columns[8]] = df.iloc[:, 8]

            # === 输出路径：保留输入的相对目录结构 ===
            rel_dir = os.path.relpath(os.path.dirname(file_path), start=input_dir)
            rel_dir = "" if rel_dir == "." else rel_dir  # 根目录文件不额外加一层
            out_subdir = os.path.join(output_dir, rel_dir)
            os.makedirs(out_subdir, exist_ok=True)

            output_path = os.path.join(out_subdir, f"processed_{file_name}")

            # 格式化工作表名称：包含实际比值和归一化因子（比值保留6位、小数点替换为下划线；长度<=31）
            sheet_name = f"{actual_ratio:.6f}_{normalization_factor:.3f}".replace('.', '_')[:31]

            # 使用ExcelWriter指定工作表名称
            with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
                new_df.to_excel(writer, sheet_name=sheet_name, index=False)

            print(f"已处理并保存: {output_path}")
            print(f"工作表名称: {sheet_name}")

        except Exception as e:
            print(f"处理文件 {file_name} 时出错: {str(e)}")
            continue

    print("\n所有文件处理完成!")


if __name__ == "__main__":
    # 用户配置参数
    INPUT_DIRECTORY = r"D:\BA\Data_processing\signal_reading\Normalization/Normalization_final_v2/bbp"  # 根目录（包含树状子目录）
    OUTPUT_DIRECTORY = r"D:\BA\Data_processing\signal_reading\Normalization/final_result_v2/bbp"  # 输出根目录（将保留相对结构）

    process_excel_files(INPUT_DIRECTORY, OUTPUT_DIRECTORY)
