import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import re
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# 设置matplotlib后端为TkAgg以解决显示问题
plt.switch_backend('TkAgg')
# 设置中文字体和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False

# 定义多种不同的颜色，用于总散点图中区分不同类别
COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
    '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'
]


def load_data(file_path):
    """加载Excel数据"""
    try:
        return pd.read_excel(file_path)
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到。")
        raise
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {str(e)}")
        raise


def preprocess_data(data):
    """数据预处理"""
    scaler = StandardScaler()
    return scaler.fit_transform(data)


def dbscan_clustering(X, eps, min_samples, metric='euclidean', algorithm='auto'):
    """
    执行DBSCAN聚类算法，返回标签和核心点标记
    标签：-1表示噪声，其他为簇ID
    核心点：True表示核心点，False表示边界点
    """
    n_samples = X.shape[0]
    labels = np.full(n_samples, -1, dtype=np.int32)  # 初始化标签为-1(噪声)
    is_core = np.full(n_samples, False, dtype=bool)  # 标记是否为核心点
    visited = np.zeros(n_samples, dtype=bool)  # 访问标记

    # 计算邻居矩阵
    if algorithm == 'brute' or n_samples < 1000:
        from scipy.spatial.distance import cdist
        dist_matrix = cdist(X, X, metric=metric)
        neighbors_matrix = dist_matrix <= eps
    else:
        nbrs = NearestNeighbors(radius=eps, algorithm=algorithm, metric=metric).fit(X)
        neighbors_matrix = nbrs.radius_neighbors_graph(X).toarray().astype(bool)

    # 计算每个点的邻居数并标记核心点
    neighbor_counts = np.sum(neighbors_matrix, axis=1)
    core_points = neighbor_counts >= min_samples
    is_core[core_points] = True

    # 主循环进行聚类
    cluster_id = 0
    for i in range(n_samples):
        if visited[i] or not core_points[i]:
            continue

        # 开始新聚类
        visited[i] = True
        labels[i] = cluster_id
        seeds = np.where(neighbors_matrix[i])[0].tolist()

        # 扩展聚类
        j = 0
        while j < len(seeds):
            current_point = seeds[j]

            if not visited[current_point]:
                visited[current_point] = True

                if core_points[current_point]:
                    # 添加新的核心点邻居到种子集
                    new_neighbors = np.where(neighbors_matrix[current_point])[0]
                    for neighbor in new_neighbors:
                        if not visited[neighbor] and neighbor not in seeds:
                            seeds.append(neighbor)

            # 标记为当前聚类
            if labels[current_point] == -1:
                labels[current_point] = cluster_id

            j += 1

        cluster_id += 1

    return labels, is_core


# ===== 新增：团簇数量约束（密度去噪第一步）=====
def enforce_max_clusters(labels: np.ndarray, max_clusters: int = 2, verbose_prefix: str = "") -> np.ndarray:
    """
    若非噪声簇数量 > max_clusters，则反复把“点数最少”的簇整体标记为噪声（-1），直到≤max_clusters。
    仅修改 labels，不改其它逻辑。
    """
    lab = labels.copy()
    removed = 0
    while True:
        # 统计非噪声簇
        uniq = [c for c in np.unique(lab) if c != -1]
        if len(uniq) <= max_clusters:
            break
        # 找到最小簇
        counts = [(c, int(np.sum(lab == c))) for c in uniq]
        counts.sort(key=lambda x: x[1])  # 按簇大小升序
        smallest_cluster = counts[0][0]
        lab[lab == smallest_cluster] = -1  # 标为噪声
        removed += 1
    if removed > 0:
        print(f"{verbose_prefix}DBSCAN得到 {len(uniq)+removed} 个簇，按规则剔除 {removed} 个最小簇 -> 至多 {max_clusters} 个簇")
    return lab


def extract_type_from_filename(filename):
    """从文件名中提取类型，如从"processed_CDCA_120.9_48.19"提取"CDCA" """
    match = re.search(r'processed_(.*?)_', filename)
    if match:
        return match.group(1)
    return "Unknown"


def process_single_file(file_path, sd_image_folder, dt_image_folder, eps=0.2, min_samples=10):
    """处理单个Excel文件，返回结果数据框和正常点数据"""
    # 加载数据
    df = load_data(file_path)
    file_basename = os.path.splitext(os.path.basename(file_path))[0]

    # ---------------------- 只保留IsReal列为TRUE的事件 ----------------------
    if 'IsReal' not in df.columns:
        raise KeyError(f"文件 {file_basename} 中缺少'IsReal'列，无法筛选有效事件")

    df['IsReal'] = df['IsReal'].fillna(False)
    total_samples_before = len(df)
    df_filtered = df[df['IsReal'] == True].reset_index(drop=True)
    total_samples_after = len(df_filtered)

    print(
        f"文件 {file_basename}：筛选前总样本数={total_samples_before}，筛选后有效样本数（IsReal=True）={total_samples_after}")
    if total_samples_after == 0:
        print(f"警告：文件 {file_basename} 筛选后无有效事件（IsReal=True），跳过后续处理")
        return df, extract_type_from_filename(file_basename), np.array([]), np.array([])

    # 提取文件名和类型
    data_type = extract_type_from_filename(file_basename)

    # 准备SD散点图数据点 (M, S) 与 DT散点图数据点 (M, DT)
    sd_points, dt_points, event_ids = [], [], []
    try:
        for idx, row in df_filtered.iterrows():
            event_id = row[df_filtered.columns[0]]  # 事件编号
            peak_count = row.get('峰个数', 1)
            dt_value = row['DT']

            if peak_count == 2:
                m1, s1 = row['M1'], row['S1']
                m2, s2 = row['M2'], row['S2']
                sd_points.append([m1, s1]); sd_points.append([m2, s2])
                dt_points.append([m1, dt_value]); dt_points.append([m2, dt_value])
                event_ids.append(event_id); event_ids.append(event_id)
            else:
                m1, s1 = row['M1'], row['S1']
                sd_points.append([m1, s1])
                dt_points.append([m1, dt_value])
                event_ids.append(event_id)
    except KeyError as e:
        print(f"文件 {file_basename} 中缺少必要的列: {str(e)}")
        raise
    except Exception as e:
        print(f"处理数据点时出错: {str(e)}")
        raise

    sd_points = np.array(sd_points) if sd_points else np.array([])
    dt_points = np.array(dt_points) if dt_points else np.array([])
    event_ids = np.array(event_ids) if event_ids else np.array([])

    if len(sd_points) == 0 or len(dt_points) == 0:
        print(f"文件 {file_basename} 无有效点数据，跳过聚类和绘图")
        df['IsNoise'] = df.apply(
            lambda row: False if row['IsReal'] else True,
            axis=1
        )
        df['type'] = data_type
        return df, data_type, np.array([]), np.array([])

    # ===== SD：预处理与聚类 =====
    X_sd_scaled = preprocess_data(sd_points)
    sd_labels, sd_is_core = dbscan_clustering(X_sd_scaled, eps, min_samples)
    # —— 密度去噪第一步：限制最多2个簇（其余最小簇改为噪声）——
    sd_labels = enforce_max_clusters(sd_labels, max_clusters=2, verbose_prefix="SD：")

    # ===== DT：预处理与聚类 =====
    X_dt_scaled = preprocess_data(dt_points)
    dt_labels, dt_is_core = dbscan_clustering(X_dt_scaled, eps, min_samples)
    # —— 密度去噪第一步：限制最多2个簇（其余最小簇改为噪声）——
    dt_labels = enforce_max_clusters(dt_labels, max_clusters=2, verbose_prefix="DT：")

    # 确定异常事件（任何点是噪声的事件）
    sd_noise_points = sd_labels == -1
    dt_noise_points = dt_labels == -1
    abnormal_events = set(event_ids[sd_noise_points]) | set(event_ids[dt_noise_points])

    # ---------------------- 生成SD散点图（去噪前）----------------------
    plt.figure(figsize=(10, 6))
    sd_core_mask = sd_is_core & (sd_labels != -1)
    plt.scatter(sd_points[sd_core_mask, 0], sd_points[sd_core_mask, 1],
                color='darkblue', marker='o', s=5, label='核心点')
    sd_border_mask = ~sd_is_core & (sd_labels != -1)
    plt.scatter(sd_points[sd_border_mask, 0], sd_points[sd_border_mask, 1],
                color='lightblue', marker='o', s=5, label='边界点')
    plt.scatter(sd_points[sd_noise_points, 0], sd_points[sd_noise_points, 1],
                color='red', marker='x', s=5, label='噪声点')
    plt.title(f'{file_basename} - SD散点图（去噪前，仅IsReal=True事件）')
    plt.xlabel('M'); plt.ylabel('S')
    plt.xlim(0.4,0.75); plt.ylim(0, 4)
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.7); plt.tight_layout()
    sd_pre_denoise_path = os.path.join(sd_image_folder, f"{file_basename}_SD_去噪前.png")
    plt.savefig(sd_pre_denoise_path); plt.close()
    print(f"SD去噪前图像已保存至: {sd_pre_denoise_path}")

    # ---------------------- 生成SD散点图（去噪后）----------------------
    plt.figure(figsize=(10, 6))
    sd_normal_mask = (sd_labels != -1) & ~np.isin(event_ids, list(abnormal_events))
    plt.scatter(sd_points[sd_normal_mask, 0], sd_points[sd_normal_mask, 1],
                color='green', marker='o', s=5, label='正常点')
    sd_abnormal_normal_mask = (sd_labels != -1) & np.isin(event_ids, list(abnormal_events))
    plt.scatter(sd_points[sd_abnormal_normal_mask, 0], sd_points[sd_abnormal_normal_mask, 1],
                color='gray', marker='o', s=5, label='异常事件的正常点')
    plt.title(f'{file_basename} - SD散点图（去噪后，仅IsReal=True事件）')
    plt.xlabel('M'); plt.ylabel('S')
    plt.xlim(0.4,0.75); plt.ylim(0, 4)
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.7); plt.tight_layout()
    sd_post_denoise_path = os.path.join(sd_image_folder, f"{file_basename}_SD_去噪后.png")
    plt.savefig(sd_post_denoise_path); plt.close()
    print(f"SD去噪后图像已保存至: {sd_post_denoise_path}")

    # ---------------------- 生成DT散点图（去噪前）----------------------
    plt.figure(figsize=(10, 6))
    dt_core_mask = dt_is_core & (dt_labels != -1)
    plt.scatter(dt_points[dt_core_mask, 0], dt_points[dt_core_mask, 1],
                color='darkblue', marker='o', s=5, label='核心点')
    dt_border_mask = ~dt_is_core & (dt_labels != -1)
    plt.scatter(dt_points[dt_border_mask, 0], dt_points[dt_border_mask, 1],
                color='lightblue', marker='o', s=5, label='边界点')
    plt.scatter(dt_points[dt_noise_points, 0], dt_points[dt_noise_points, 1],
                color='red', marker='x', s=5, label='噪声点')
    plt.title(f'{file_basename} - DT散点图（去噪前，仅IsReal=True事件）')
    plt.xlabel('M'); plt.ylabel('DT')
    plt.xlim(0.4,0.75); plt.ylim(0, 6)
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.7); plt.tight_layout()
    dt_pre_denoise_path = os.path.join(dt_image_folder, f"{file_basename}_DT_去噪前.png")
    plt.savefig(dt_pre_denoise_path); plt.close()
    print(f"DT去噪前图像已保存至: {dt_pre_denoise_path}")

    # ---------------------- 生成DT散点图（去噪后）----------------------
    plt.figure(figsize=(10, 6))
    dt_normal_mask = (dt_labels != -1) & ~np.isin(event_ids, list(abnormal_events))
    plt.scatter(dt_points[dt_normal_mask, 0], dt_points[dt_normal_mask, 1],
                color='green', marker='o', s=5, label='正常点')
    dt_abnormal_normal_mask = (dt_labels != -1) & np.isin(event_ids, list(abnormal_events))
    plt.scatter(dt_points[dt_abnormal_normal_mask, 0], dt_points[dt_abnormal_normal_mask, 1],
                color='gray', marker='o', s=5, label='异常事件的正常点')
    plt.title(f'{file_basename} - DT散点图（去噪后，仅IsReal=True事件）')
    plt.xlabel('M'); plt.ylabel('DT')
    plt.xlim(0.4,0.75); plt.ylim(0, 6)
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.7); plt.tight_layout()
    dt_post_denoise_path = os.path.join(dt_image_folder, f"{file_basename}_DT_去噪后.png")
    plt.savefig(dt_post_denoise_path); plt.close()
    print(f"DT去噪后图像已保存至: {dt_post_denoise_path}")

    # 添加新列（基于原始df，保持所有行的输出完整性）
    try:
        df['IsNoise'] = df.apply(
            lambda row: True if (not row.get('IsReal', True)) or (row[df.columns[0]] in abnormal_events) else False,
            axis=1
        )
        df['type'] = data_type
    except Exception as e:
        print(f"添加新列时出错: {str(e)}")
        raise

    # 提取正常点数据，用于总散点图
    sd_normal_points = sd_points[sd_normal_mask] if 'sd_normal_mask' in locals() else np.array([])
    dt_normal_points = dt_points[dt_normal_mask] if 'dt_normal_mask' in locals() else np.array([])

    return df, data_type, sd_normal_points, dt_normal_points


def batch_process_excel_files(folder_path, output_folder, eps=0.2, min_samples=10):
    """批量处理文件夹中的所有Excel文件，支持选择类别生成总散点图"""
    # 确保输出文件夹和图像子文件夹存在
    sd_image_folder = os.path.join(output_folder, "SD散点图")
    dt_image_folder = os.path.join(output_folder, "DT散点图")

    for folder in [output_folder, sd_image_folder, dt_image_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # 获取文件夹中所有Excel文件
    excel_files = [f for f in os.listdir(folder_path) if f.endswith(('.xlsx', '.xls'))]

    if not excel_files:
        print(f"在 {folder_path} 中未找到Excel文件")
        return

    # 收集所有类别的正常点数据，用于绘制总散点图
    all_sd_normal_points = []  # 存储(类别名称, 点数据, 颜色)
    all_dt_normal_points = []  # 存储(类别名称, 点数据, 颜色)
    unique_types = []  # 存储所有独特的类别

    for i, file in enumerate(excel_files):
        file_path = os.path.join(folder_path, file)
        print(f"\n处理文件: {file_path}")

        try:
            color = COLORS[i % len(COLORS)]

            result_df, data_type, sd_normal_points, dt_normal_points = process_single_file(
                file_path, sd_image_folder, dt_image_folder, eps, min_samples)

            # 保存结果
            output_filename = f"final_{os.path.basename(file)}"
            output_path = os.path.join(output_folder, output_filename)
            result_df.to_excel(output_path, index=False)
            print(f"处理结果已保存至: {output_path}")

            # 合并类别点
            if len(sd_normal_points) > 0 and len(dt_normal_points) > 0:
                if data_type not in unique_types:
                    unique_types.append(data_type)
                    all_sd_normal_points.append((data_type, sd_normal_points, color))
                    all_dt_normal_points.append((data_type, dt_normal_points, color))
                else:
                    for idx, (t, p, c) in enumerate(all_sd_normal_points):
                        if t == data_type:
                            all_sd_normal_points[idx] = (t, np.vstack((p, sd_normal_points)), c)
                            all_dt_normal_points[idx] = (
                                t, np.vstack((all_dt_normal_points[idx][1], dt_normal_points)), c)
                            break

        except Exception as e:
            print(f"处理文件 {file} 时出错: {str(e)}，跳过该文件继续处理")
            continue

    if not all_sd_normal_points or not all_dt_normal_points:
        print("\n无有效类别数据用于生成总散点图，总散点图生成步骤跳过")
        return

    print("\n检测到以下可用类别（仅含IsReal=True且有正常点数据的类别）：")
    for i, (data_type, _, _) in enumerate(all_sd_normal_points):
        print(f"{i + 1}. {data_type}")

    while True:
        try:
            selection = input("\n请输入要显示的类别序号（用逗号分隔，如：1,3,5 或输入'all'显示全部）：")
            if selection.lower() == 'all':
                selected_indices = list(range(len(all_sd_normal_points)))
                break
            else:
                selected_indices = [int(idx.strip()) - 1 for idx in selection.split(',')]
                for idx in selected_indices:
                    if idx < 0 or idx >= len(all_sd_normal_points):
                        raise ValueError(f"无效的序号：{idx + 1}")
                break
        except ValueError as e:
            print(f"输入错误：{e}，请重新输入")

    selected_sd_points = [all_sd_normal_points[i] for i in selected_indices]
    selected_dt_points = [all_dt_normal_points[i] for i in selected_indices]

    # ---------------------- 生成SD总散点图 ----------------------
    plt.figure(figsize=(12, 8))
    for data_type, points, color in selected_sd_points:
        if len(points) > 0:
            plt.scatter(points[:, 0], points[:, 1],
                        color=color, marker='o', s=5, label=data_type, alpha=1)

    plt.title('所选类别的SD正常点总散点图（仅IsReal=True事件）')
    plt.xlabel('M'); plt.ylabel('S')
    plt.xlim(0.4,0.75); plt.ylim(0, 4)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=1)
    plt.tight_layout()
    sd_total_plot_path = os.path.join(sd_image_folder, "所选类别SD总散点图.png")
    plt.savefig(sd_total_plot_path, bbox_inches='tight')
    plt.close()
    print(f"SD总散点图已保存至: {sd_total_plot_path}")

    # ---------------------- 生成DT总散点图 ----------------------
    plt.figure(figsize=(12, 8))
    for data_type, points, color in selected_dt_points:
        if len(points) > 0:
            plt.scatter(points[:, 0], points[:, 1],
                        color=color, marker='o', s=5, label=data_type, alpha=1)

    plt.title('所选类别的DT正常点总散点图（仅IsReal=True事件）')
    plt.xlabel('M'); plt.ylabel('DT')
    plt.xlim(0.4,0.75); plt.ylim(0, 6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    dt_total_plot_path = os.path.join(dt_image_folder, "所选a类别DT总散点图.png")
    plt.savefig(dt_total_plot_path, bbox_inches='tight')
    plt.close()
    print(f"DT总散点图已保存至: {dt_total_plot_path}")


if __name__ == "__main__":
    # 设置文件夹路径和参数
    input_folder = r"D:\BA\Data_processing\signal_reading\Normalization/final_result_v2/bbp"  # 输入文件夹路径
    output_folder = r"D:\BA\Data_processing\signal_reading\draw/训练集bbp-弱聚类"  # 输出文件夹路径
    eps = 1  # DBSCAN邻域半径，可根据实际数据调整
    min_samples = 10  # DBSCAN最小样本数，可根据实际数据调整

    # 批量处理
    batch_process_excel_files(input_folder, output_folder, eps, min_samples)
    print("\n所有文件处理完成！")
