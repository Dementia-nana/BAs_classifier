import os
import re
import pyabf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime

# 解决Matplotlib后端问题（后台出图）
plt.switch_backend('Agg')

# ======================
# 可配置参数
# ======================

# 事件时间所在列名（请按你的 Excel 实际列名修改）
EVENT_START_COL = "T1"  # 比如改成 "T1"
EVENT_END_COL   = "T2"  # 比如改成 "T2"

# 输入根目录：批量模式时，ABF 和 Excel 都放在这里
INPUT_ROOT = r"D:\BA\Data_processing\signal_reading\extract_event/bbp"  # 自己改成你的目录

# 单文件模式时用到（下面 main() 里有开关）
SINGLE_ABF_PATH   = r"D:\BA\Data_processing\signal_reading\lvbo\final_result_v2\example.abf"
SINGLE_EXCEL_PATH = r"D:\BA\Data_processing\signal_reading\extract_event\extract_event_final_v2\精修事件时间_example.xlsx"

# 输出根目录：所有精修 Excel 和图片子文件夹都放这里
OUTPUT_ROOT = r"D:\BA\Data_processing\signal_reading\extract_event/bbp_result"

# 图片最多生成多少张
MAX_IMG_COUNT = 20

# 单文件 / 批量模式
BATCH_MODE = True   # True = 批量处理 INPUT_ROOT；False = 处理 SINGLE_ABF_PATH + SINGLE_EXCEL_PATH


# ======================
# 工具函数：从 Excel 文件名中抽取 key（去掉时间戳）
# 如：精修事件时间_HS-1_filtered_20251203_133000.xlsx -> key="HS-1_filtered"
# ======================
def extract_key_from_excel(excel_path: str) -> str:
    fn = os.path.basename(excel_path)
    stem, _ = os.path.splitext(fn)
    m = re.match(r"精修事件时间_(?P<key>.+?)(?:_[0-9]{8}_[0-9]{6})?$", stem)
    if m:
        return m.group("key")
    # 不符合上面模式时，就用去掉前缀后的整个名字
    if stem.startswith("精修事件时间_"):
        return stem[len("精修事件时间_"):]
    return stem


# ======================
# 核心处理：一对 ABF + Excel
# ======================
def process_single_pair(abf_file_path: str, event_excel_path: str):
    """
    处理单个 ABF + 事件时间 Excel（两者一一对应）。
    区间来源改为：从 Excel 里读取 EVENT_START_COL / EVENT_END_COL。
    之后的过滤、精修、画图、导出精修时间 Excel 的逻辑保持不变。
    """
    try:
        # ---------------------- 1. 读取 ABF ----------------------
        abf = pyabf.ABF(abf_file_path)
        Current_signal = abf.data[0]
        time_axis = abf.sweepX
        signal_units = abf.sweepUnitsY
        print(f"\n✅ 成功读取 ABF 文件：{os.path.basename(abf_file_path)}")

        # ---------------------- 2. 读取事件时间 Excel ----------------------
        print(f"   使用事件时间 Excel：{os.path.basename(event_excel_path)}")
        df_events = pd.read_excel(event_excel_path)

        if EVENT_START_COL not in df_events.columns or EVENT_END_COL not in df_events.columns:
            raise ValueError(
                f"Excel 中未找到指定的事件时间列：'{EVENT_START_COL}' / '{EVENT_END_COL}'。\n"
                f"当前列名：{list(df_events.columns)}"
            )

        # 根据时间列在 ABF 中找到对应的索引，构造粗略事件区间 ranges
        ranges = []
        invalid_rows = 0
        for i in range(len(df_events)):
            try:
                start_time_ms = float(df_events.iloc[i][EVENT_START_COL])
                end_time_ms = float(df_events.iloc[i][EVENT_END_COL])

                start_time = start_time_ms / 1000.0
                end_time = end_time_ms / 1000.0
            except Exception:
                print(f"   ⚠️ 第 {i+1} 行时间无法转换为 float，跳过。")
                invalid_rows += 1
                continue

            start_idx = int(np.argmin(np.abs(time_axis - start_time)))
            end_idx   = int(np.argmin(np.abs(time_axis - end_time)))

            if start_idx >= end_idx:
                print(f"   ⚠️ 第 {i+1} 行事件时间范围无效：start_idx={start_idx}, end_idx={end_idx}，已跳过。")
                invalid_rows += 1
                continue

            ranges.append((start_idx, end_idx))

        if not ranges:
            print("❌ 根据 Excel 提供的时间未获得任何有效事件区间，终止该文件处理。")
            return

        print(f"事件时间总行数：{len(df_events)}，有效区间数：{len(ranges)}，无效行数：{invalid_rows}")

        # ---------------------- 3. 中文字体设置 ----------------------
        plt.rcParams["font.family"] = ["SimHei"]
        plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

        # ---------------------- 4. 过滤区间（逻辑保持不变） ----------------------
        threshold = -40  # 仅用于画图中的参考线

        filtered_ranges = []
        filter_reasons = []   # 每个原始区间的过滤原因
        filtered_indices = set()  # 通过过滤的区间（按 ranges 的索引）

        for idx, (start, end) in enumerate(ranges):
            segment_length = end - start + 1
            segment = Current_signal[start:end + 1]
            current_avg = np.mean(segment) if len(segment) > 0 else 0.0

            reasons = []
            if len(segment) == 0:
                reasons.append("区间长度为 0")
            else:
                if np.max(segment) >= 5:
                    reasons.append(f"最大值≥5 (实际: {np.max(segment):.2f})")
                if segment_length <= 150:
                    reasons.append(f"点数≤150 (实际: {segment_length})")
                if segment_length >= 300000:
                    reasons.append(f"点数≥150000 (实际: {segment_length})")
                if current_avg < -35:
                    reasons.append(f"平均值<-35 (实际: {current_avg:.2f})")
                if current_avg > 0:
                    reasons.append(f"平均值>-10 (实际: {current_avg:.2f})")

            # 无过滤原因 -> 通过
            if not reasons:
                filtered_ranges.append((start, end))
                filtered_indices.add(idx)

            filter_reasons.append(reasons)

        # ---------------------- 5. 精修区间（逻辑保持不变） ----------------------
        refined_events = {}       # {原始区间索引: (new_start, new_end)}
        refined_event_data = []   # 用于导出精修事件时间
        refined_event_id = 1      # 精修事件编号（从1开始）

        for idx, (start, end) in enumerate(filtered_ranges):
            # 找出它在原始 ranges 里的索引 original_idx
            original_idx = [i for i, (s, e) in enumerate(ranges) if s == start and e == end][0]

            new_start = start
            while new_start < end and Current_signal[new_start + 1] >= Current_signal[new_start]:
                new_start += 1

            new_end = end
            while new_end > new_start and Current_signal[new_end - 1] >= Current_signal[new_end]:
                new_end -= 1

            if new_end > new_start:
                refined_events[original_idx] = (new_start, new_end)
                refined_event_data.append({
                    "精修事件编号": refined_event_id,
                    "原始区间索引": original_idx + 1,  # 原始区间从1开始显示
                    "精修开始时间(s)": round(time_axis[new_start], 6),
                    "精修结束时间(s)": round(time_axis[new_end], 6),
                    "精修事件时长(s)": round(time_axis[new_end] - time_axis[new_start], 6)
                })
                refined_event_id += 1

        print(f"\n事件统计（{os.path.basename(abf_file_path)}）：")
        print(f"  原始区间数: {len(ranges)}")
        print(f"  过滤后区间数: {len(filtered_ranges)}")
        print(f"  精修后区间数: {len(refined_events)}")

        # ---------------------- 6. 导出精修事件时间 Excel（去掉时间戳） ----------------------
        os.makedirs(OUTPUT_ROOT, exist_ok=True)
        key = extract_key_from_excel(event_excel_path)
        excel_filename = f"精修事件时间_{key}.xlsx"   # 无时间戳
        excel_path_out = os.path.join(OUTPUT_ROOT, excel_filename)

        if refined_event_data:
            refined_df = pd.DataFrame(refined_event_data)
            refined_df.to_excel(excel_path_out, index=False, engine="openpyxl")
            print(f"\n✅ 精修事件时间已导出至 Excel：{excel_path_out}")
        else:
            print(f"\n⚠️ 无有效精修事件，未生成精修 Excel 文件")

        # ---------------------- 7. 生成事件示意图（最多 MAX_IMG_COUNT 张，逻辑保持不变） ----------------------
        img_output_folder = os.path.join(OUTPUT_ROOT, key)  # 图片文件夹名就是 key
        os.makedirs(img_output_folder, exist_ok=True)
        print(f"\n图片将保存至：{img_output_folder}（最多 {MAX_IMG_COUNT} 张）")

        generated_img_count = 0
        total_ranges = len(ranges)

        for idx, (start, end) in enumerate(ranges, 1):
            if generated_img_count >= MAX_IMG_COUNT:
                print(f"\n⚠️ 已达到最大图片数量（{MAX_IMG_COUNT} 张），停止生成后续图片")
                break

            plt.figure(figsize=(10, 6), dpi=100)

            # 绘制全程原始信号
            plt.plot(time_axis, Current_signal, color='lightgray', linewidth=0.7, label='原始信号')

            # 判断是否通过过滤
            is_filtered = (idx - 1) in filtered_indices

            if is_filtered:
                # 通过过滤：显示精修区间（红色实线）+ 原始区间（红色虚线）
                refined_start, refined_end = refined_events.get(idx - 1, (start, end))
                plt.plot(
                    time_axis[refined_start:refined_end + 1],
                    Current_signal[refined_start:refined_end + 1],
                    color='red',
                    linewidth=2.0,
                    label='精修后事件（通过过滤）'
                )
                plt.plot(
                    time_axis[start:end + 1],
                    Current_signal[start:end + 1],
                    color='red',
                    linestyle='--',
                    linewidth=1.0,
                    label='原始区间'
                )
                status_text = "通过过滤"
            else:
                # 被过滤：只显示原始区间（绿色实线）
                plt.plot(
                    time_axis[start:end + 1],
                    Current_signal[start:end + 1],
                    color='green',
                    linewidth=2.0,
                    label='原始区间（被过滤）'
                )
                status_text = "被过滤"

            # 添加阈值线，仅用于参考
            plt.axhline(y=threshold, color='gray', linestyle='--', linewidth=1.0, label=f'检测阈值: {threshold}')

            # x 轴范围聚焦事件附近
            event_start_time = time_axis[start]
            event_end_time = time_axis[end]
            event_duration = event_end_time - event_start_time
            expand = max(event_duration * 1, 0.01)
            plt.xlim(event_start_time - expand, event_end_time + expand)

            # y 轴范围固定
            plt.ylim(-130, 0)

            # 被过滤事件时标注过滤原因
            if not is_filtered and filter_reasons[idx - 1]:
                reason_text = "过滤原因:\n" + "\n".join(filter_reasons[idx - 1])
                plt.text(
                    0.02, 0.02, reason_text,
                    transform=plt.gca().transAxes,
                    fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.8)
                )

            # 标题和标签
            plt.title(
                f'原始事件 {idx}/{total_ranges} - {status_text}\n'
                f'时间范围: {event_start_time:.6f}s - {event_end_time:.6f}s',
                fontsize=12
            )
            plt.xlabel('时间 (s)', fontsize=10)
            plt.ylabel(f'信号幅度 ({signal_units})', fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(loc='upper right', fontsize=9)
            plt.tight_layout()

            # 保存图片（文件名无时间戳，只包含事件序号和时间）
            status_flag = "通过" if is_filtered else "过滤"
            img_filename = f'原始事件_{idx}_{status_flag}_时间_{event_start_time:.3f}-{event_end_time:.3f}s.png'
            img_path = os.path.join(img_output_folder, img_filename)
            plt.savefig(img_path, format='png', bbox_inches='tight')
            plt.close()

            generated_img_count += 1
            if (generated_img_count % 10 == 0
                or generated_img_count == MAX_IMG_COUNT
                or generated_img_count == total_ranges):
                print(f"  已生成 {generated_img_count}/{min(total_ranges, MAX_IMG_COUNT)} 张图片")

        actual_img_count = min(total_ranges, MAX_IMG_COUNT)
        print(f"\n✅ 图片生成完成！共生成 {generated_img_count}/{actual_img_count} 张图片")
        print(f"  其中通过过滤的事件图片: {sum(1 for i in range(actual_img_count) if i in filtered_indices)} 张")
        print(f"  被过滤的事件图片: {actual_img_count - sum(1 for i in range(actual_img_count) if i in filtered_indices)} 张")

    except FileNotFoundError:
        print(f"错误：找不到指定的 ABF 或 Excel 文件，请检查路径：{abf_file_path} / {event_excel_path}")
    except Exception as e:
        print(f"\n发生未知错误（处理 {os.path.basename(abf_file_path)} 时）: {str(e)}")
        import traceback
        traceback.print_exc()


# ======================
# 批量匹配 ABF 与 Excel
# ======================
def find_pairs_in_root(root_dir: str):
    """
    在 root_dir 中寻找 .abf 与 .xlsx 文件，一一配对：
    - ABF key = 去掉扩展名的文件名，例如 HS-1_filtered
    - Excel key = extract_key_from_excel() 得到的 key
    返回列表 [(abf_path, excel_path, key), ...]
    """
    files = os.listdir(root_dir)
    abf_files = [os.path.join(root_dir, f) for f in files if f.lower().endswith(".abf")]
    xls_files = [os.path.join(root_dir, f) for f in files if f.lower().endswith((".xlsx", ".xls"))]

    abf_map = {}
    for p in abf_files:
        stem = os.path.splitext(os.path.basename(p))[0]
        abf_map.setdefault(stem, []).append(p)

    excel_map = {}
    for p in xls_files:
        key = extract_key_from_excel(p)
        excel_map.setdefault(key, []).append(p)

    print(f"\n在 {root_dir} 找到 ABF：{len(abf_files)} 个，Excel：{len(xls_files)} 个")
    print("ABF keys（前 10 个）：", list(abf_map.keys())[:10])
    print("Excel keys（前 10 个）：", list(excel_map.keys())[:10])

    pairs = []
    keys = sorted(set(abf_map.keys()) & set(excel_map.keys()))
    for k in keys:
        # 如果同 key 有多个文件，随便取一个（一般你是一一对应的）
        abf_path = sorted(abf_map[k])[0]
        excel_path = sorted(excel_map[k])[0]
        pairs.append((abf_path, excel_path, k))

    miss_abf = set(abf_map.keys()) - set(excel_map.keys())
    miss_xls = set(excel_map.keys()) - set(abf_map.keys())
    if miss_abf:
        print("⚠️ 下列 ABF key 未找到对应 Excel：", miss_abf)
    if miss_xls:
        print("⚠️ 下列 Excel key 未找到对应 ABF：", miss_xls)

    print(f"成功配对：{len(pairs)} 对（示例前 5 对 key）：{[k for _,_,k in pairs[:5]]}")
    return pairs


# ======================
# 主流程
# ======================
def main():
    if BATCH_MODE:
        if not os.path.isdir(INPUT_ROOT):
            print(f"错误：批量模式下 INPUT_ROOT 不是有效目录：{INPUT_ROOT}")
            return
        pairs = find_pairs_in_root(INPUT_ROOT)
        if not pairs:
            print("❌ 未找到任何 ABF–Excel 配对，程序结束。")
            return

        print(f"\n🔎 开始批量处理，共 {len(pairs)} 对文件")
        for i, (abf_path, excel_path, key) in enumerate(pairs, start=1):
            print("\n" + "=" * 80)
            print(f"[{i}/{len(pairs)}] key={key}")
            print(f"ABF : {abf_path}")
            print(f"Excel: {excel_path}")
            process_single_pair(abf_path, excel_path)

        print("\n🎉 批量处理完成。")
        print("输出目录：", os.path.abspath(OUTPUT_ROOT))
    else:
        # 单文件模式
        if not os.path.isfile(SINGLE_ABF_PATH):
            print(f"错误：SINGLE_ABF_PATH 不是有效 ABF 文件：{SINGLE_ABF_PATH}")
            return
        if not os.path.isfile(SINGLE_EXCEL_PATH):
            print(f"错误：SINGLE_EXCEL_PATH 不是有效 Excel 文件：{SINGLE_EXCEL_PATH}")
            return
        print("\n👉 单文件模式：")
        print("ABF  :", SINGLE_ABF_PATH)
        print("Excel:", SINGLE_EXCEL_PATH)
        process_single_pair(SINGLE_ABF_PATH, SINGLE_EXCEL_PATH)
        print("\n🎉 单文件处理完成。")
        print("输出目录：", os.path.abspath(OUTPUT_ROOT))


if __name__ == "__main__":
    main()
