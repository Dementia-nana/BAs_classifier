import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyabf
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
import matplotlib.gridspec as gridspec
import traceback
from glob import glob

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号


# ---------------------- 批量匹配工具 ----------------------
def extract_key(path, mode='stem', regex=None):
    """
    从文件路径抽取匹配用的 key。
    - mode='stem'：使用文件名主干（不含扩展名）
    - mode='regex'：使用自定义正则（需包含命名捕获组 'key'）
    """
    fn = os.path.basename(path)
    stem, _ = os.path.splitext(fn)
    if mode == 'stem':
        return stem
    elif mode == 'regex' and regex:
        m = re.match(regex, fn)
        return m.group('key') if m else None
    else:
        return stem


def find_file_pairs(
    abf_root, excel_root,
    abf_glob="**/*.abf",
    excel_glob="**/*.xlsx",
    key_mode="regex",
    abf_key_regex=r'(?P<key>.+)\.abf$',  # LCA_2_filtered.abf -> key=LCA_2_filtered
    excel_key_regex=r'精修事件时间_(?P<key>.+?)(?:\.xlsx)?$' # 精修事件时间_LCA_2_filtered_YYYYMMDD_HHMMSS.xlsx -> key=LCA_2_filtered
):
    """
    在两个根目录下递归找文件，并按 key 匹配成对。
    返回列表：[ (key, abf_path, excel_path) , ... ]
    """
    abf_files = glob(os.path.join(abf_root, abf_glob), recursive=True)
    excel_files = glob(os.path.join(excel_root, excel_glob), recursive=True)

    abf_map = {}
    for p in abf_files:
        k = extract_key(os.path.basename(p), key_mode, abf_key_regex)
        if k:
            abf_map.setdefault(k, []).append(p)

    excel_map = {}
    for p in excel_files:
        k = extract_key(os.path.basename(p), key_mode, excel_key_regex)
        if k:
            excel_map.setdefault(k, []).append(p)

    keys = sorted(set(abf_map.keys()) & set(excel_map.keys()))
    pairs = []
    for k in keys:
        # 若同 key 多文件：取“路径最深 + 文件名最长”的那个
        choose = lambda lst: sorted(lst, key=lambda s: (len(s.split(os.sep)), len(os.path.basename(s))), reverse=True)[0]
        abf_path = choose(abf_map[k])
        excel_path = choose(excel_map[k])
        pairs.append((k, abf_path, excel_path))

    print(f"在 {abf_root} 找到 ABF：{len(abf_files)} 个；在 {excel_root} 找到 Excel：{len(excel_files)} 个")
    print(f"匹配成功的成对 key：{len(pairs)} 个（示例前5个） -> {[p[0] for p in pairs[:5]]}")
    miss_e = list(set(abf_map.keys()) - set(excel_map.keys()))
    miss_a = list(set(excel_map.keys()) - set(abf_map.keys()))
    if miss_e:
        print(f"⚠️ 这些 ABF key 未找到 Excel：{miss_e[:10]}{' ...' if len(miss_e)>10 else ''}")
    if miss_a:
        print(f"⚠️ 这些 Excel key 未找到 ABF：{miss_a[:10]}{' ...' if len(miss_a)>10 else ''}")
    return pairs


# ---------------------- 保持你的峰合并/边界查找逻辑 ----------------------
def filter_close_peaks(peak_positions, peak_densities, min_distance=2.5, max_distance=7.0):
    """保持现状：相邻峰距离<min 或 >max 都合并，仅保留密度最高的峰"""
    if len(peak_positions) <= 1:
        return peak_positions, peak_densities, False

    sorted_indices = np.argsort(peak_positions)
    sorted_pos = peak_positions[sorted_indices]
    sorted_dens = peak_densities[sorted_indices]

    filtered_pos = []
    filtered_dens = []
    current_group_pos = [sorted_pos[0]]
    current_group_dens = [sorted_dens[0]]
    has_filtered = False

    for i in range(1, len(sorted_pos)):
        distance = sorted_pos[i] - sorted_pos[i - 1]
        if distance < min_distance or distance > max_distance:
            current_group_pos.append(sorted_pos[i])
            current_group_dens.append(sorted_dens[i])
            has_filtered = True
        else:
            max_idx = np.argmax(current_group_dens)
            filtered_pos.append(current_group_pos[max_idx])
            filtered_dens.append(current_group_dens[max_idx])
            current_group_pos = [sorted_pos[i]]
            current_group_dens = [sorted_dens[i]]

    max_idx = np.argmax(current_group_dens)
    filtered_pos.append(current_group_pos[max_idx])
    filtered_dens.append(current_group_dens[max_idx])

    if len(peak_positions) > len(filtered_pos):
        has_filtered = True

    return np.array(filtered_pos), np.array(filtered_dens), has_filtered


def find_density_boundary(y, density, peak_pos, target_density=0.01, direction='up'):
    """寻找指定峰值附近密度为目标值的边界点"""
    peak_idx = np.argmin(np.abs(y - peak_pos))
    if direction == 'up':
        for i in range(0 if direction == 'down' else peak_idx, len(y)):
            if density[i] <= target_density:
                return y[i]
        return y[-1]
    else:
        for i in range(peak_idx, -1, -1):
            if density[i] <= target_density:
                return y[i]
        return y[0]


# ---------------------- 事件处理（3σ清洗 + 清洗可视化叠加） ----------------------
def process_event(abf_data, start_time, end_time, event_index, save_dir, params, sampling_rate):
    try:
        time_axis = np.arange(len(abf_data)) / sampling_rate
        start_idx = np.argmin(np.abs(time_axis - start_time))
        end_idx = np.argmin(np.abs(time_axis - end_time))
        if start_idx >= end_idx:
            print(f"事件 {event_index} 时间范围无效，跳过")
            return None

        event_data = abf_data[start_idx:end_idx + 1]
        event_duration = (end_idx - start_idx + 1) / sampling_rate * 1000  # ms

        # 一轮清洗：移除 > -10pA
        valid_mask = event_data <= -10
        cleaned_data = event_data[valid_mask]
        if len(cleaned_data) == 0:
            print(f"事件 {event_index} 清洗后无有效数据，跳过")
            return None

        # KDE（带宽修正）
        kde = gaussian_kde(cleaned_data, bw_method=params['bandwidth'])
        y_kde = np.linspace(-35, 0, 500)
        density = kde(y_kde)

        # 峰检测 + 条件筛选（-30~-13 pA 且 密度≥0.02）
        peak_indices, _ = find_peaks(density, prominence=params['prominence'])
        all_peak_y = y_kde[peak_indices]
        all_peak_density = density[peak_indices]
        valid_peak_mask = (all_peak_y >= -30) & (all_peak_y <= -13) & (all_peak_density >= 0.01)
        all_peak_y = all_peak_y[valid_peak_mask]
        all_peak_density = all_peak_density[valid_peak_mask]
        print(f"事件 {event_index}：原始峰{len(peak_indices)}，有效峰{len(all_peak_y)}（-30~-13pA & 密度≥0.02）")

        # 峰合并 + 最多保留2个
        if len(all_peak_y) > 0:
            filtered_peak_y, filtered_peak_density, _ = filter_close_peaks(
                all_peak_y, all_peak_density, params['peak_min_distance'], params['peak_max_distance']
            )
            sorted_indices = np.argsort(filtered_peak_density)[::-1]
            filtered_peak_y = filtered_peak_y[sorted_indices][:2]
            filtered_peak_density = filtered_peak_density[sorted_indices][:2]
        else:
            filtered_peak_y, filtered_peak_density = np.array([]), np.array([])

        # 谷值 / 分界线
        valley_y = []
        if len(filtered_peak_y) >= 2:
            valley_indices, _ = find_peaks(-density, prominence=params['prominence'])
            valley_y = y_kde[valley_indices]

        upper_bound, lower_bound = np.nan, np.nan
        if len(filtered_peak_y) >= 1:
            max_peak = filtered_peak_y[0] if len(filtered_peak_y) == 1 else filtered_peak_y[np.argmax(filtered_peak_y)]
            min_peak = filtered_peak_y[0] if len(filtered_peak_y) == 1 else filtered_peak_y[np.argmin(filtered_peak_y)]
            upper_bound = find_density_boundary(y_kde, density, max_peak, 0.01, 'up')
            lower_bound = find_density_boundary(y_kde, density, min_peak, 0.01, 'down')

        divider = np.nan
        if len(valley_y) > 0 and len(filtered_peak_y) >= 2:
            mean_peak = np.mean(filtered_peak_y)
            divider = valley_y[np.argmin(np.abs(valley_y - mean_peak))]

        # ===== 3σ 清洗可视化准备 =====
        def sigma3_stats(arr):
            if arr.size == 0:
                return np.nan, np.nan, np.nan
            mu = float(np.mean(arr))
            sd = float(np.std(arr))
            if np.isnan(sd) or sd == 0.0:
                return mu, mu, mu
            return mu - 2.0 * sd, mu + 2.0 * sd, sd

        indices = np.arange(len(cleaned_data))
        high_level = np.zeros_like(cleaned_data, dtype=bool)
        low_level = np.zeros_like(cleaned_data, dtype=bool)
        single_level = np.zeros_like(cleaned_data, dtype=bool)

        if not np.isnan(upper_bound) and not np.isnan(lower_bound):
            if len(filtered_peak_y) == 1:
                single_level = (cleaned_data <= upper_bound) & (cleaned_data >= lower_bound)
            elif len(filtered_peak_y) >= 2 and not np.isnan(divider):
                high_level = (cleaned_data <= upper_bound) & (cleaned_data >= divider)
                low_level = (cleaned_data < divider) & (cleaned_data >= lower_bound)

        high_lo = high_hi = low_lo = low_hi = single_lo = single_hi = np.nan
        high_out = low_out = single_out = np.zeros_like(cleaned_data, dtype=bool)

        if np.any(high_level):
            h_lo, h_hi, _ = sigma3_stats(cleaned_data[high_level])
            high_lo, high_hi = h_lo, h_hi
            high_out = high_level & ((cleaned_data < high_lo) | (cleaned_data > high_hi))

        if np.any(low_level):
            l_lo, l_hi, _ = sigma3_stats(cleaned_data[low_level])
            low_lo, low_hi = l_lo, l_hi
            low_out = low_level & ((cleaned_data < low_lo) | (cleaned_data > low_hi))

        if np.any(single_level):
            s_lo, s_hi, _ = sigma3_stats(cleaned_data[single_level])
            single_lo, single_hi = s_lo, s_hi
            single_out = single_level & ((cleaned_data < single_lo) | (cleaned_data > single_hi))

        # ===== 可视化：上下两个子图 =====
        fig = plt.figure(figsize=(12, 10))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])

        # 上子图：核密度
        ax_kde = plt.subplot(gs[0, 0])
        ax_kde.fill_betweenx(y_kde, 0, density, alpha=0.7, color='#3498db')
        ax_kde.plot(density, y_kde, 'b-', linewidth=2, label='核密度曲线')
        ax_kde.axvline(x=0.01, color='purple', linestyle='-.', linewidth=1, label='边界密度阈值 (0.01)')
        ax_kde.axvline(x=0.02, color='red', linestyle='-.', linewidth=1, label='峰值密度阈值 (0.02)')
        ax_kde.axhline(y=-30, color='orange', linestyle=':', linewidth=1, label='峰值范围 (-30~-13pA)')
        ax_kde.axhline(y=-13, color='orange', linestyle=':', linewidth=1)

        for i, (py, pdens) in enumerate(zip(filtered_peak_y, filtered_peak_density)):
            ax_kde.axhline(y=py, color='red', linestyle='--', linewidth=1.5)
            ax_kde.text(pdens + 0.01, py, f'有效峰{i + 1}: {py:.1f}pA', ha='left', va='center', color='red', fontsize=8)

        if not np.isnan(divider):
            ax_kde.axhline(y=divider, color='green', linestyle='--', linewidth=1.5)
            ax_kde.text(kde(divider) + 0.01, divider, '分界线', ha='left', va='center', color='green', fontsize=8)

        if not np.isnan(upper_bound):
            ax_kde.axhline(y=upper_bound, color='orange', linestyle='-', linewidth=1.5, label='上边界')
        if not np.isnan(lower_bound):
            ax_kde.axhline(y=lower_bound, color='blue', linestyle='-', linewidth=1.5, label='下边界')

        ax_kde.set_ylim(-35, 0)
        ax_kde.set_title(f'事件 {event_index} 核密度分布（含{len(filtered_peak_y)}个有效峰）', fontsize=12)
        ax_kde.set_xlabel('概率密度', fontsize=10)
        ax_kde.set_ylabel('电流 (pA)', fontsize=10)
        ax_kde.grid(alpha=0.3, linestyle='--')
        ax_kde.legend(fontsize=8)

        # 下子图：分类 + 3σ可视化
        ax_classified = plt.subplot(gs[1, 0])
        ax_classified.set_title('分类后的电流信号（含3σ清洗可视化）', fontsize=12)
        ax_classified.set_ylabel('电流 (pA)', fontsize=10)
        ax_classified.set_xlabel('数据点', fontsize=10)
        ax_classified.set_ylim(-35, 0)
        ax_classified.grid(alpha=0.3, linestyle=':')

        if not np.isnan(upper_bound) and not np.isnan(lower_bound):
            if np.any(single_level):
                ax_classified.scatter(indices[single_level], cleaned_data[single_level],
                                      color='green', alpha=0.7, label='单水平')
            if np.any(high_level):
                ax_classified.scatter(indices[high_level], cleaned_data[high_level],
                                      color='green', alpha=0.7, label='高水平')
            if np.any(low_level):
                ax_classified.scatter(indices[low_level], cleaned_data[low_level],
                                      color='blue', alpha=0.7, label='低水平')

            if not np.isnan(divider) and np.any(high_level | low_level):
                ax_classified.axhline(y=divider, color='green', linestyle='--', linewidth=1.5, label='分界线')

            ax_classified.axhline(y=upper_bound, color='orange', linestyle='-', linewidth=1.5, label='上边界')
            ax_classified.axhline(y=lower_bound, color='blue', linestyle='-', linewidth=1.5, label='下边界')
            ax_classified.axhline(y=-30, color='orange', linestyle=':', linewidth=1, label='峰值范围 (-30~-13pA)')
            ax_classified.axhline(y=-13, color='orange', linestyle=':', linewidth=1)

        # 3σ 外点
        if np.any(high_out):
            ax_classified.scatter(indices[high_out], cleaned_data[high_out], color='gray', alpha=0.9, s=10, label='3σ外点(高)')
        if np.any(low_out):
            ax_classified.scatter(indices[low_out], cleaned_data[low_out], color='gray', alpha=0.9, s=10, label='3σ外点(低)')
        if np.any(single_out):
            ax_classified.scatter(indices[single_out], cleaned_data[single_out], color='gray', alpha=0.9, s=10, label='3σ外点(单)')

        # 3σ范围线
        if not np.isnan(high_lo) and not np.isnan(high_hi) and np.any(high_level):
            ax_classified.axhline(y=high_lo, color='green', linestyle='-.', linewidth=1, label='高 3σ下界')
            ax_classified.axhline(y=high_hi, color='green', linestyle='-.', linewidth=1, label='高 3σ上界')
        if not np.isnan(low_lo) and not np.isnan(low_hi) and np.any(low_level):
            ax_classified.axhline(y=low_lo, color='blue', linestyle='-.', linewidth=1, label='低 3σ下界')
            ax_classified.axhline(y=low_hi, color='blue', linestyle='-.', linewidth=1, label='低 3σ上界')
        if not np.isnan(single_lo) and not np.isnan(single_hi) and np.any(single_level):
            ax_classified.axhline(y=single_lo, color='green', linestyle='-.', linewidth=1, label='单 3σ下界')
            ax_classified.axhline(y=single_hi, color='green', linestyle='-.', linewidth=1, label='单 3σ上界')

        # 去重图例项
        handles, labels = ax_classified.get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        ax_classified.legend(uniq.values(), uniq.keys(), fontsize=9, loc='best')

        # 统计信息文本框
        stats_text = (
            f"事件 {event_index} 统计信息：\n"
            f"总数据点: {len(event_data)} | 清洗后数据点: {len(cleaned_data)}\n"
            f"事件持续时间: {event_duration:.4f} 毫秒 | 有效峰数量: {len(filtered_peak_y)}\n"
            f"峰筛选条件：峰值-30~-13pA 且 密度≥0.02\n"
            f"上边界: {upper_bound:.1f}pA | 下边界: {lower_bound:.1f}pA | 分界线: {divider:.1f}pA\n"
        )
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"事件_{event_index}_核密度分析图.png")
        plt.figtext(0.02, 0.01, stats_text, fontsize=9, bbox=dict(facecolor='white', alpha=0.8, pad=5))
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"事件 {event_index} 分析图已保存: {save_path}")

        # ===== 3σ清洗 + 重新计算特征（Excel 用） =====
        def clean_by_3sigma(arr):
            if arr.size == 0:
                return arr, np.nan, np.nan, np.nan, np.nan
            mu = float(np.mean(arr))
            sd = float(np.std(arr))
            if np.isnan(sd) or sd == 0.0:
                return arr, mu, sd, mu, mu
            lo = mu - 2.0 * sd
            hi = mu + 2.0 * sd
            mask = (arr >= lo) & (arr <= hi)
            cleaned_arr = arr[mask]
            return cleaned_arr, float(np.mean(cleaned_arr)) if cleaned_arr.size > 0 else np.nan, \
                   float(np.std(cleaned_arr)) if cleaned_arr.size > 0 else np.nan, lo, hi

        high_pts = cleaned_data[high_level] if np.any(high_level) else np.array([])
        low_pts = cleaned_data[low_level] if np.any(low_level) else np.array([])
        single_pts = cleaned_data[single_level] if np.any(single_level) else np.array([])

        if len(filtered_peak_y) >= 2:
            print(f"事件 {event_index}：双峰进入 3σ 清洗（用于Excel统计）")
            _, high_mu, high_sd, high_lo2, high_hi2 = clean_by_3sigma(high_pts)
            _, low_mu, low_sd, low_lo2, low_hi2 = clean_by_3sigma(low_pts)
            print(f"  高水平 3σ区间: [{high_lo2:.3f}, {high_hi2:.3f}]")
            print(f"  低水平 3σ区间: [{low_lo2:.3f}, {low_hi2:.3f}]")

            result = {
                '事件编号': event_index,
                '峰个数': len(filtered_peak_y),
                '事件持续时间(毫秒)': event_duration,
                '高水平电流平均': high_mu if high_pts.size > 0 else np.nan,
                '高水平SD平均': high_sd if high_pts.size > 0 else np.nan,
                '低水平电流平均': low_mu if low_pts.size > 0 else np.nan,
                '低水平SD平均': low_sd if low_pts.size > 0 else np.nan,
                '单水平电流平均': np.nan,
                '单水平SD平均': np.nan,
            }

        elif len(filtered_peak_y) == 1:
            print(f"事件 {event_index}：单峰进入 3σ 清洗（用于Excel统计）")
            _, single_mu, single_sd, s_lo2, s_hi2 = clean_by_3sigma(single_pts)
            print(f"  单水平 3σ区间: [{s_lo2:.3f}, {s_hi2:.3f}]")

            result = {
                '事件编号': event_index,
                '峰个数': len(filtered_peak_y),
                '事件持续时间(毫秒)': event_duration,
                '高水平电流平均': np.nan,
                '高水平SD平均': np.nan,
                '低水平电流平均': np.nan,
                '低水平SD平均': np.nan,
                '单水平电流平均': single_mu if single_pts.size > 0 else np.nan,
                '单水平SD平均': single_sd if single_pts.size > 0 else np.nan,
            }
        else:
            result = {
                '事件编号': event_index,
                '峰个数': 0,
                '事件持续时间(毫秒)': event_duration,
                '高水平电流平均': np.nan,
                '高水平SD平均': np.nan,
                '低水平电流平均': np.nan,
                '低水平SD平均': np.nan,
                '单水平电流平均': np.nan,
                '单水平SD平均': np.nan,
            }

        return result

    except Exception as e:
        print(f"处理事件 {event_index} 时出错: {str(e)}")
        traceback.print_exc()
        return None


# ---------------------- 单对文件运行 ----------------------
def run_single_pair(excel_path, abf_path, output_dir, params, key=None):
    """
    新增参数 key：用于输出 Excel 文件名（不带时间戳）。
    如果未传 key，则使用 ABF 的主干名。
    """
    try:
        print(f"读取Excel：{excel_path}")
        df_events = pd.read_excel(excel_path)
        # 这里要求四列：精修事件编号、原始区间索引、精修开始时间(s)、精修结束时间(s)
        required_columns = ['精修事件编号', '原始区间索引', '精修开始时间(s)', '精修结束时间(s)']
        if not all(col in df_events.columns for col in required_columns):
            print(f"错误: Excel必须包含列: {required_columns}")
            return False

        print(f"读取ABF：{abf_path}")
        abf = pyabf.ABF(abf_path)
        current_signal = abf.data[0]
        sampling_rate = abf.dataRate
        print(f"ABF长度: {len(current_signal)} 点, 采样率: {sampling_rate:.1f} Hz")

        # —— 输出目录：按 key 建子文件夹（此处的 output_dir 已是 <output_root>/<key>）——
        os.makedirs(output_dir, exist_ok=True)

        total_events = len(df_events)
        print(f"共 {total_events} 个事件，开始处理...")
        results = []
        for i in range(total_events):
            event_index = i + 1
            start_time = df_events.iloc[i]['精修开始时间(s)']
            end_time = df_events.iloc[i]['精修结束时间(s)']
            print(f"\n处理事件 {event_index}/{total_events}: {start_time:.6f}s - {end_time:.6f}s")
            res = process_event(current_signal, start_time, end_time, event_index, output_dir, params, sampling_rate)
            if res:
                # 把输入Excel中的“精修事件编号”和“原始区间索引”带进结果里
                res['精修事件编号'] = df_events.iloc[i]['精修事件编号']
                res['原始区间索引'] = df_events.iloc[i]['原始区间索引']
                results.append(res)

        if results:
            excel_rows = []
            for res in results:
                # 输出Excel前两列：精修事件编号、原始区间索引
                row = {
                    '精修事件编号': res['精修事件编号'],
                    '原始区间索引': res['原始区间索引'],
                    '峰个数': res['峰个数'],
                    '事件持续时间': res['事件持续时间(毫秒)'],
                    'mean1': np.nan,
                    'SD1': np.nan,
                    'mean2': np.nan,
                    'SD2': np.nan,
                    'IsReal': res['峰个数'] != 0
                }
                if res['峰个数'] >= 2:
                    row['mean1'] = res['高水平电流平均']
                    row['SD1'] = res['高水平SD平均']
                    row['mean2'] = res['低水平电流平均']
                    row['SD2'] = res['低水平SD平均']
                elif res['峰个数'] == 1:
                    row['mean1'] = res['单水平电流平均']
                    row['SD1'] = res['单水平SD平均']
                    row['mean2'] = res['单水平电流平均']
                    row['SD2'] = res['单水平SD平均']
                excel_rows.append(row)

            # 输出 Excel 文件名包含 key（不带时间戳）
            if key is None or not str(key).strip():
                key = os.path.splitext(os.path.basename(abf_path))[0]
            out_xlsx = os.path.join(output_dir, f"{key}_事件特征参数统计.xlsx")

            pd.DataFrame(excel_rows).to_excel(out_xlsx, index=False, engine='openpyxl')
            print(f"\n✅ 统计Excel已保存：{out_xlsx}")
            return True
        else:
            print("⚠️ 无有效事件结果")
            return False

    except Exception as e:
        print(f"单对文件处理失败：{e}")
        traceback.print_exc()
        return False


# ---------------------- 主函数：批量/单文件 ----------------------
def main():
    try:
        # ======= 选择模式 =======
        BATCH_MODE = True  # True=批量；False=单文件

        # ======= 分析参数（保持你的原设定） =======
        params = {
            'bandwidth': 0.3,
            'prominence': 0.001,
            'peak_min_distance': 2.5,
            'peak_max_distance': 8
        }

        if not BATCH_MODE:
            # --- 单文件模式（把以下路径改成你的） ---：
            excel_path = r"D:\BA\Data_processing\signal_reading\KDE\bbp\bbp-filtered.xlsx"
            abf_path = r"D:\BA\Data_processing\signal_reading\KDE\bbp\bbp-filtered.abf"
            output_root = r"D:\BA\Data_processing\signal_reading\KDE\bbp_result"

            # —— 从文件名自动提取 key：例如 CA_3.7_filtered ——
            key_from_abf = extract_key(os.path.basename(abf_path), mode='regex', regex=r'(?P<key>.+)\.abf$')
            key = key_from_abf or extract_key(os.path.basename(excel_path), mode='regex',
                                              regex=r'精修事件时间_(?P<key>.+)_[0-9]{8}_[0-9]{6}(?:\.xlsx)?$')
            if not key:
                key = os.path.splitext(os.path.basename(abf_path))[0]

            # —— 输出目录固定为 <output_root>/<key> ——（满足你的新需求）
            output_dir = os.path.join(output_root, key)
            os.makedirs(output_dir, exist_ok=True)

            run_single_pair(excel_path, abf_path, output_dir, params, key=key)

        else:
            # --- 批量模式（把三个根目录改成你的） ---
            abf_root = r"D:\BA\Data_processing\signal_reading\KDE/bbp"
            excel_root = r"D:\BA\Data_processing\signal_reading\KDE/bbp"
            output_root = r"D:\BA\Data_processing\signal_reading\KDE/bbp_result"

            pairs = find_file_pairs(
                abf_root=abf_root,
                excel_root=excel_root,
                abf_glob="**/*.abf",
                excel_glob="**/*.xlsx",
                key_mode="regex",
                abf_key_regex=r'(?P<key>.+)\.abf$',
                excel_key_regex=r'精修事件时间_(?P<key>.+?)(?:\.xlsx)?$'
            )

            print(f"\n开始批量处理，共 {len(pairs)} 对文件")
            success, fail = 0, 0
            for key, abf_path, excel_path in pairs:
                # —— 输出目录固定为 <output_root>/<key> ——（扁平输入也会按 key 建 11 个文件夹）
                out_dir = os.path.join(output_root, key)
                print("\n" + "=" * 80)
                print(f"处理 key: {key}")
                print(f"ABF : {abf_path}")
                print(f"Excel: {excel_path}")
                print(f"输出: {out_dir}")
                ok = run_single_pair(excel_path, abf_path, out_dir, params, key=key)
                success += int(ok)
                fail += (0 if ok else 1)

            print("\n" + "=" * 80)
            print(f"批量完成：成功 {success} / 失败 {fail}（总 {len(pairs)}）")
            print(f"所有结果位于：{os.path.abspath(output_root)}")

    except Exception as e:
        print(f"程序运行出错: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
