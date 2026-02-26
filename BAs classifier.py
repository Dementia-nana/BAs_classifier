from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    StratifiedKFold, learning_curve, StratifiedShuffleSplit
)
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.base import clone
from sklearn.utils.class_weight import compute_sample_weight
import traceback

# PDF
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle

# =========================
# 字体（解决中文显示空心方块）
# =========================
import matplotlib
from matplotlib import font_manager


def _setup_matplotlib_chinese_font():
    candidates = [
        "Microsoft YaHei", "Microsoft YaHei UI",
        "SimHei", "SimSun", "NSimSun",
        "PingFang SC", "Heiti SC", "Songti SC",
        "Noto Sans CJK SC", "Noto Sans CJK",
        "WenQuanYi Micro Hei",
        "Arial Unicode MS",
    ]

    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = None
    for name in candidates:
        if name in available:
            chosen = name
            break

    if chosen:
        matplotlib.rcParams["font.sans-serif"] = [chosen] + ["DejaVu Sans"]
        matplotlib.rcParams["font.family"] = "sans-serif"
        print(f"✅ Matplotlib font set to: {chosen}")
    else:
        matplotlib.rcParams["font.family"] = "sans-serif"
        matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
        print("⚠️ 未检测到常见中文字体")

    matplotlib.rcParams["axes.unicode_minus"] = False


_setup_matplotlib_chinese_font()

_SIMHEI_OK = False
styles = getSampleStyleSheet()
try:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont

    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    for k in ("Normal", "Title", "Heading1", "Heading2", "Heading3"):
        styles[k].fontName = "STSong-Light"
    _SIMHEI_OK = True
    print("✅ ReportLab font set to: STSong-Light")
except Exception:
    _SIMHEI_OK = False
    print("⚠️ ReportLab 中文字体设置失败（仅影响 PDF 模块）。")

# =========================
# 全局配置
# =========================
DATA_DIR = Path("machine_learning/practice_v2")  # 训练数据目录
PREDICT_PATH = Path("machine_learning/预测集v2调整后/final_processed_HS_filtered_v2_调整后.xlsx")  # 预测数据（可为目录或单个文件）
OUTPUT_DIR = Path("machine_learning/outputs")  # 输出目录
MAX_CV_SPLITS = 10

# ======= 运行模块开关（按需调 True/False）=======
RUN_MODEL_SELECTION = True  # 训练并评估9个模型、选择最佳
RUN_CONFUSION_MATS = True  # 绘制并导出混淆矩阵（Train & OOF）
RUN_LEARNING_CURVES = False  # 学习曲线（时间最耗，调试时可关）
RUN_FEATURE_IMPORT = False  # 导出特征重要性/系数
RUN_PDF_REPORT = False  # 生成PDF报告
RUN_PREDICTION = True  # 执行预测（强制归类至11类之一）
RUN_EXPORT_OOF = False  # 导出OOF逐样本CSV + CM矩阵CSV
RUN_DEBUG_SNAPSHOTS = False  # 导出各阶段数据快照CSV（便于排错）

# ======= 平行坐标数据导出（训练集本身数据，不含任何预测）=======
EXPORT_PARALLEL_DATA = True  # True=导出；False=不导出

# ======= 新增：额外绘图功能（你提出的 3 项）=======
RUN_EXTRA_PLOTS = True  # True=额外输出散点图/表格图/平行坐标图

# ======= 学习曲线细化开关/速度档 =======
LC_IMPL = "balanced"  # "balanced"（自定义类均衡）或 "sklearn"
LC_FAST_MODE = False  # True=轻量快速，False=全面精细
PLOT_LEARNING_CURVES_FOR = "best_only"  # "best_only" 或 "all"

# —— 学习曲线参数（按快慢模式自动调整）——
USE_CLASS_BALANCING = True
if LC_FAST_MODE:
    LC_SSS_SPLITS = 4
    LC_SSS_TEST_SIZE = 0.2
    LC_TRAIN_SIZES_ABS = (80, 200, 400)
    LC_BAL_TRAIN_FRACS = [0.05, 0.10, 0.20, 0.35, 0.60, 0.85, 1.00]
    LC_BAL_REPEATS = 3
else:
    LC_SSS_SPLITS = 8
    LC_SSS_TEST_SIZE = 0.2
    LC_TRAIN_SIZES_ABS = (80, 120, 200, 350, 600, 900, 1300)
    LC_BAL_TRAIN_FRACS = [
        0.02, 0.03, 0.04, 0.05, 0.06,
        0.07, 0.08, 0.09, 0.10, 0.12,
        0.14, 0.16, 0.18, 0.20, 0.22,
        0.25, 0.28, 0.30, 0.33, 0.36,
        0.40, 0.45, 0.50, 0.55, 0.60,
        0.70, 0.80, 0.90, 0.95, 1.00
    ]
    LC_BAL_REPEATS = 5

# 固定特征/目标（你已经固定）
PRESET_FEATURE_COLS = ['DT', 'M1', 'S1', 'M2', 'S2']
PRESET_TARGET_COL = 'type'

MODEL_WHITELIST = None


# =========================
# 小工具
# =========================
def _slugify(name: str) -> str:
    keep = "-_().[] 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return "".join(ch if ch in keep else "_" for ch in str(name))


def _save_df_indexed(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False, encoding='utf-8-sig')


def _fixed_val_split(y, test_size=0.2, random_state=42):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    (tr_idx, val_idx), = sss.split(np.zeros_like(y), y)
    return tr_idx, val_idx


def _balanced_sample_indices(y_subset, n_total, n_classes, rng, with_replacement=True):
    per_cls = int(np.ceil(n_total / n_classes))
    idx_list = []
    unique_classes = np.unique(y_subset)
    for c in unique_classes:
        pool = np.where(y_subset == c)[0]
        if len(pool) == 0:
            continue
        if len(pool) >= per_cls or not with_replacement:
            take = rng.choice(pool, size=min(per_cls, len(pool)), replace=False)
        else:
            take = rng.choice(pool, size=per_cls, replace=True)
        idx_list.append(take)
    if not idx_list:
        return np.array([], dtype=int)
    sel = np.concatenate(idx_list)
    if len(sel) > n_total:
        sel = rng.choice(sel, size=n_total, replace=False)
    return sel


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _save_df(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False, encoding='utf-8-sig')


def _append_log(msg: str, log_path: Path):
    print(msg)
    with log_path.open('a', encoding='utf-8') as f:
        f.write(msg + '\n')


# =========================
# 平行坐标数据导出（训练集本身数据）
# =========================
def export_parallel_coordinates_data_from_training_itself(
        X_raw: pd.DataFrame,
        X_encoded: pd.DataFrame,
        y_true_str: np.ndarray,
        feature_cols: list,
        out_dir: Path,
        timestamp: str,
        source_df: pd.DataFrame = None
):
    try:
        df_out = pd.DataFrame()
        for c in feature_cols:
            if c in X_raw.columns:
                df_out[c] = X_raw[c].values
            else:
                df_out[c] = X_encoded[c].values

        df_out["true_label"] = np.asarray(y_true_str, dtype=str)

        if source_df is not None:
            for extra_col in ["source_file", "file_group"]:
                if extra_col in source_df.columns:
                    df_out[extra_col] = source_df[extra_col].values

        out_path1 = out_dir / f"parallel_coords_data_train_itself_{timestamp}.csv"
        _save_df_indexed(df_out, out_path1)

        df_enc = X_encoded[feature_cols].copy()
        df_enc["true_label"] = np.asarray(y_true_str, dtype=str)
        if source_df is not None:
            for extra_col in ["source_file", "file_group"]:
                if extra_col in source_df.columns:
                    df_enc[extra_col] = source_df[extra_col].values

        out_path2 = out_dir / f"parallel_coords_data_train_itself_encoded_{timestamp}.csv"
        _save_df_indexed(df_enc, out_path2)

        print(f"✅ 已导出平行坐标图数据（训练集本身，不含预测）：\n  - {out_path1}\n  - {out_path2}")
    except Exception as e:
        print(f"⚠️ 导出平行坐标图数据失败：{e}")
        traceback.print_exc()


# =========================
# 调试/快照
# =========================
def debug_dataset_snapshot(df: pd.DataFrame, out_dir: Path, stage: str, feature_cols=None, target_col=None):
    if not RUN_DEBUG_SNAPSHOTS:
        return
    log_path = out_dir / "debug_log.txt"
    _append_log(f"\n=== [{stage}] ===", log_path)
    _append_log(f"shape={df.shape}", log_path)

    if 'IsNoise' in df.columns:
        vc = df['IsNoise'].value_counts(dropna=False)
        _append_log("IsNoise counts:", log_path)
        for k, v in vc.items():
            _append_log(f"  {repr(k)} -> {v}", log_path)
        counts_df = vc.rename_axis('IsNoise').reset_index(name='count')
        _save_df(counts_df, out_dir / f"debug_{stage}_IsNoise_counts.csv")

    if target_col and target_col in df.columns:
        y = df[target_col].astype(str)
        vc = y.value_counts()
        _append_log(f"Target `{target_col}` classes={len(vc)}", log_path)
        for k, v in vc.items():
            _append_log(f"  {repr(k)} -> {v}", log_path)
        _save_df(pd.DataFrame({'label': vc.index, 'count': vc.values}), out_dir / f"debug_{stage}_target_counts.csv")
        _save_df(pd.DataFrame({'unique_labels': sorted(y.unique().tolist())}),
                 out_dir / f"debug_{stage}_target_unique.csv")

    if feature_cols:
        X = df[feature_cols].copy()
        nan_counts = X.isna().sum()
        inf_counts = np.isinf(X.to_numpy()).sum(axis=0)
        info = pd.DataFrame({
            'feature': feature_cols,
            'nan_count': [int(nan_counts.get(c, 0)) for c in feature_cols],
            'inf_count': [int(inf_counts[i]) for i in range(len(feature_cols))]
        })
        _save_df(info, out_dir / f"debug_{stage}_feature_nan_inf.csv")


def debug_cv_split_distribution(X: np.ndarray, y: np.ndarray, cv: StratifiedKFold,
                                class_names: list, out_dir: Path):
    if not RUN_DEBUG_SNAPSHOTS:
        return
    rows = []
    for fold_idx, (tr, te) in enumerate(cv.split(X, y), start=1):
        cnt_tr = Counter(y[tr]);
        cnt_te = Counter(y[te])
        for i, cname in enumerate(class_names):
            rows.append({
                'fold': fold_idx,
                'class': cname,
                'train_count': cnt_tr.get(i, 0),
                'valid_count': cnt_te.get(i, 0),
            })
    _save_df(pd.DataFrame(rows), out_dir / "debug_cv_class_distribution.csv")


# =========================
# 1. 读数据
# =========================
def load_excel_folder(directory: Path) -> pd.DataFrame:
    d = Path(directory)
    if not d.exists():
        raise FileNotFoundError(f"训练数据目录不存在: {directory}")

    files = sorted(
        [
            p for p in d.iterdir()
            if p.suffix.lower() in ('.xlsx', '.xls')
               and not p.name.startswith('~$')  # 忽略 Excel 锁文件
        ],
        key=lambda x: x.name.lower()
    )
    if not files:
        raise FileNotFoundError(f"训练数据目录内未找到Excel文件: {directory}")

    all_dfs = []
    for idx, f in enumerate(files, 1):
        try:
            # 让 pandas 自动判断更稳
            df = pd.read_excel(f)
        except Exception as e:
            print(f"\n❌ Failed to read file: {f}")
            print(f"   suffix={f.suffix}, size={f.stat().st_size} bytes")
            raise

        df['source_file'] = f.name
        df['file_group'] = idx
        all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True)


# =========================
# 2. 选特征/目标（预设）
# =========================
def select_features_target(df: pd.DataFrame):
    if not PRESET_FEATURE_COLS or not PRESET_TARGET_COL:
        raise ValueError("请在代码顶部预设 PRESET_FEATURE_COLS 和 PRESET_TARGET_COL")
    X = df[PRESET_FEATURE_COLS].copy()
    y = df[PRESET_TARGET_COL].copy().astype(str)
    return X, y, PRESET_FEATURE_COLS, PRESET_TARGET_COL


# =========================
# 3. 特征编码（训练-预测映射一致）
# =========================
def _extend_le_with_unk_inplace(le: LabelEncoder):
    classes = list(le.classes_)
    if '<UNK>' not in classes:
        classes.append('<UNK>')
        le.classes_ = np.array(classes)
    return le


def encode_features(X: pd.DataFrame):
    encoders = {}
    X_enc = X.copy()
    for col in X_enc.columns:
        if X_enc[col].dtype == 'object':
            le = LabelEncoder()
            X_enc[col] = le.fit_transform(X_enc[col].astype(str).str.strip())
            _extend_le_with_unk_inplace(le)
            encoders[col] = le
            print(f"已将特征列 '{col}' 转换为数值编码（并追加 <UNK>）")
    X_enc = X_enc.replace([np.inf, -np.inf], np.nan)
    return X_enc, encoders


# =========================
# 4. CV 生成
# =========================
def make_stratified_cv(y, max_splits=10, random_state=42):
    cnt = Counter(y)
    min_cls = min(cnt.values())
    n_splits = max(2, min(max_splits, min_cls))
    if n_splits < max_splits:
        print(f"ℹ️ 类别样本较少，将 n_splits 从 {max_splits} 下调为 {n_splits}")
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


# =========================
# 5. 9个模型（带类权重）
# =========================
def build_models():
    imp = SimpleImputer(strategy='median')
    cw = 'balanced' if USE_CLASS_BALANCING else None

    models = {
        'Logistic Regression': make_pipeline(
            imp, StandardScaler(),
            LogisticRegression(max_iter=2000, solver='lbfgs', class_weight=cw)
        ),
        'Decision Tree': make_pipeline(
            imp, DecisionTreeClassifier(random_state=42, class_weight=cw)
        ),
        'Random Forest': make_pipeline(
            imp,
            RandomForestClassifier(
                n_estimators=400,
                max_depth=6,
                min_samples_leaf=10,
                max_features='sqrt',
                class_weight=('balanced_subsample' if USE_CLASS_BALANCING else None),
                random_state=42, n_jobs=-1
            )
        ),
        'Gradient Boosting': make_pipeline(
            imp,
            HistGradientBoostingClassifier(
                max_depth=4, learning_rate=0.05, l2_regularization=5.0, random_state=42
            )
        ),
        'SVM': make_pipeline(
            imp, StandardScaler(),
            SVC(probability=True, kernel='rbf', cache_size=1000,
                decision_function_shape='ovr', class_weight=cw, random_state=42)
        ),
        'KNN': make_pipeline(imp, StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        'Naive Bayes': make_pipeline(imp, GaussianNB()),
        'LDA': make_pipeline(imp, StandardScaler(), LinearDiscriminantAnalysis()),
        'MLP': make_pipeline(imp, StandardScaler(), MLPClassifier(max_iter=1000, random_state=42))
    }

    if MODEL_WHITELIST is not None:
        models = {k: v for k, v in models.items() if k in MODEL_WHITELIST}
        print(f"仅评估模型：{list(models.keys())}")

    return models


def export_feature_importance_if_any(pipeline, feature_cols, out_dir: Path, model_name: str):
    try:
        est = pipeline[-1]
    except Exception:
        return

    safe = _slugify(model_name)

    if hasattr(est, "feature_importances_"):
        imp = np.asarray(est.feature_importances_, dtype=float)
        df = pd.DataFrame({'feature': feature_cols, 'importance': imp}).sort_values('importance', ascending=False)
        _save_df_indexed(df, out_dir / f"feature_importance_{safe}.csv")
    elif hasattr(est, "coef_"):
        coef = np.asarray(est.coef_, dtype=float)
        if coef.ndim == 1:
            coef = coef[None, :]
        cols = {f"class_{i}_coef": coef[i, :] for i in range(coef.shape[0])}
        df = pd.DataFrame({'feature': feature_cols, **cols})
        df["coef_l2norm"] = np.linalg.norm(coef, axis=0)
        df = df.sort_values('coef_l2norm', ascending=False)
        _save_df_indexed(df, out_dir / f"feature_coeff_{safe}.csv")


def fit_params_for(model_name: str, y_subset):
    if not USE_CLASS_BALANCING:
        return {}
    if model_name == 'Gradient Boosting':
        sw = compute_sample_weight(class_weight='balanced', y=y_subset)
        return {'histgradientboostingclassifier__sample_weight': sw}
    return {}


# =========================
# 6. 训练评估（KFold 交叉验证）
# =========================
def train_evaluate_models(X, y, cv):
    models = build_models()
    results = []
    err_rows = []
    y_np = np.asarray(y)

    for name, base_model in models.items():
        fold_scores = []
        ok_folds = 0
        total_folds = 0

        for fold_idx, (tr_idx, te_idx) in enumerate(cv.split(X, y_np), start=1):
            total_folds += 1
            est = clone(base_model)
            X_tr, y_tr = X.iloc[tr_idx], y_np[tr_idx]
            X_te, y_te = X.iloc[te_idx], y_np[te_idx]
            try:
                est.fit(X_tr, y_tr, **fit_params_for(name, y_tr))
                acc = est.score(X_te, y_te)
                fold_scores.append(acc)
                ok_folds += 1
            except Exception as e:
                err_rows.append({
                    "stage": "cv_score",
                    "model": name,
                    "fold": fold_idx,
                    "error": repr(e),
                    "traceback": traceback.format_exc(),
                })
                fold_scores.append(np.nan)

        scores = np.array(fold_scores, dtype=float)
        valid_scores = scores[~np.isnan(scores)]
        cv_mean = float(np.mean(valid_scores)) if valid_scores.size else np.nan
        cv_std = float(np.std(valid_scores)) if valid_scores.size else np.nan

        final_model = clone(base_model).fit(X, y_np, **fit_params_for(name, y_np))
        train_acc = accuracy_score(y_np, final_model.predict(X))

        print(
            f"{name}: CV Accuracy = {cv_mean:.4f} ± {cv_std:.4f} | Train Acc = {train_acc:.4f} | 成功折数 {ok_folds}/{total_folds}")

        results.append({
            'name': name,
            'model': final_model,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'train_acc': train_acc,
            'params': getattr(final_model, 'get_params', lambda: {})()
        })

    if err_rows:
        err_df = pd.DataFrame(err_rows)
        err_path = Path(OUTPUT_DIR) / "debug_cv_fit_errors.csv"
        err_df.to_csv(err_path, index=False, encoding='utf-8-sig')
        print(f"⚠️ 某些折在拟合时失败，已记录到：{err_path}")

    if all(pd.isna(r['cv_mean']) for r in results):
        raise RuntimeError("All models failed during CV.")

    def _score_key(r):
        return r['cv_mean'] if pd.notna(r['cv_mean']) else -1.0

    best = max(results, key=_score_key)
    print(f"\n🏆 最佳模型：{best['name']} (CV={best['cv_mean']:.4f}±{best['cv_std']:.4f})")
    return results, best


# =========================
# 7A. sklearn 学习曲线
# =========================
def plot_learning_curve_sklearn(estimator, name, X, y, out_dir: Path):
    est = clone(estimator)
    n = len(y)
    train_sizes_abs = list(LC_TRAIN_SIZES_ABS) + [int(0.7 * n), n]
    train_sizes_abs = np.array(sorted(set([t for t in train_sizes_abs if t <= n])))

    cv_lc = StratifiedShuffleSplit(n_splits=LC_SSS_SPLITS, test_size=LC_SSS_TEST_SIZE, random_state=42)
    sizes, train_scores, val_scores = learning_curve(
        est, X, y, cv=cv_lc,
        train_sizes=train_sizes_abs, n_jobs=-1, scoring='accuracy',
    )
    train_mean, val_mean = train_scores.mean(axis=1), val_scores.mean(axis=1)
    train_std, val_std = train_scores.std(axis=1), val_scores.std(axis=1)

    safe = _slugify(name)
    df_agg = pd.DataFrame({
        'model': name,
        'train_size': sizes,
        'train_mean': train_mean, 'train_std': train_std,
        'val_mean': val_mean, 'val_std': val_std
    })
    _save_df_indexed(df_agg, out_dir / f"learning_curve_sklearn_{safe}_agg.csv")

    rows = []
    for i, s in enumerate(sizes):
        for split in range(train_scores.shape[1]):
            rows.append({
                'model': name,
                'train_size': int(s),
                'split': int(split + 1),
                'train_score': float(train_scores[i, split]),
                'val_score': float(val_scores[i, split]),
            })
    df_raw = pd.DataFrame(rows)
    _save_df_indexed(df_raw, out_dir / f"learning_curve_sklearn_{safe}_raw.csv")

    plt.figure(figsize=(8, 6))
    plt.plot(sizes, train_mean, 'o-', label='训练集')
    plt.plot(sizes, val_mean, 'o-', label='验证集')
    plt.fill_between(sizes, train_mean - train_std, train_mean + train_std, alpha=0.15)
    plt.fill_between(sizes, val_mean - val_std, val_mean + val_std, alpha=0.15)
    plt.xlabel("训练样本数");
    plt.ylabel("准确率");
    plt.title(f"{name} 学习曲线（ShuffleSplit）")
    plt.legend(loc='lower right');
    plt.grid(True, linestyle='--', alpha=0.4)
    out_path = out_dir / f"{name}_learning_curve.png"
    plt.tight_layout();
    plt.savefig(out_path, dpi=300);
    plt.close()
    return str(out_path)


# =========================
# 7B. 自定义“类均衡抽样”学习曲线
# =========================
def plot_learning_curve_balanced(
        estimator, name, X, y, class_names, cv, out_dir: Path,
        train_fracs=None, repeats=5, use_fixed_val=False, val_size=0.2,
        random_state=42, with_replacement=True
):
    rng_master = np.random.RandomState(random_state)
    n = len(y)
    n_classes = len(class_names)
    if train_fracs is None:
        train_fracs = LC_BAL_TRAIN_FRACS

    sizes = (np.array(train_fracs) * n).astype(int)
    sizes[sizes < n_classes] = n_classes

    if use_fixed_val:
        tr_pool_idx, val_idx = _fixed_val_split(y, test_size=val_size, random_state=random_state)
        X_pool, y_pool = X.iloc[tr_pool_idx], np.asarray(y)[tr_pool_idx]
        X_val, y_val = X.iloc[val_idx], np.asarray(y)[val_idx]
    else:
        X_pool, y_pool = X, np.asarray(y)

    train_means, val_means, train_stds, val_stds = [], [], [], []
    raw_rows = []
    safe = _slugify(name)

    for n_train in sizes:
        train_scores_rep, val_scores_rep = [], []
        for rep in range(repeats):
            rng = np.random.RandomState(rng_master.randint(0, 2 ** 31 - 1))
            sel_rel = _balanced_sample_indices(y_pool, n_train, n_classes, rng, with_replacement=with_replacement)
            if sel_rel.size == 0:
                continue
            X_train = X_pool.iloc[sel_rel]
            y_train = y_pool[sel_rel]

            est = clone(estimator)
            if use_fixed_val:
                est.fit(X_train, y_train, **fit_params_for(name, y_train))
                tr_acc = est.score(X_train, y_train)
                va_acc = est.score(X_val, y_val)
            else:
                k = int(np.bincount(y_train).min())
                n_splits = max(2, min(5, k))
                est.fit(X_train, y_train, **fit_params_for(name, y_train))
                tr_acc = est.score(X_train, y_train)
                if n_splits < 2:
                    va_acc = np.nan
                else:
                    inner_cv = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                               random_state=rng_master.randint(0, 2 ** 31 - 1))
                    fold_acc = []
                    for tr_idx, te_idx in inner_cv.split(X_train, y_train):
                        est_i = clone(estimator)
                        est_i.fit(X_train.iloc[tr_idx], y_train[tr_idx], **fit_params_for(name, y_train[tr_idx]))
                        fold_acc.append(est_i.score(X_train.iloc[te_idx], y_train[te_idx]))
                    va_acc = float(np.mean(fold_acc)) if len(fold_acc) else np.nan

            train_scores_rep.append(tr_acc)
            val_scores_rep.append(va_acc)
            raw_rows.append({
                'model': name,
                'train_size': int(n_train),
                'repeat': int(rep + 1),
                'train_score': float(tr_acc),
                'val_score': (np.nan if pd.isna(va_acc) else float(va_acc)),
                'use_fixed_val': bool(use_fixed_val),
            })

        train_scores_rep = np.array(train_scores_rep, dtype=float)
        val_scores_rep = np.array(val_scores_rep, dtype=float)
        train_means.append(np.nanmean(train_scores_rep))
        val_means.append(np.nanmean(val_scores_rep))
        train_stds.append(np.nanstd(train_scores_rep))
        val_stds.append(np.nanstd(val_scores_rep))

    df_raw = pd.DataFrame(raw_rows)
    _save_df_indexed(df_raw, out_dir / f"learning_curve_balanced_{safe}_raw.csv")

    df_agg = pd.DataFrame({
        'model': name,
        'train_size': sizes,
        'train_mean': train_means, 'train_std': train_stds,
        'val_mean': val_means, 'val_std': val_stds,
        'use_fixed_val': use_fixed_val
    })
    _save_df_indexed(df_agg, out_dir / f"learning_curve_balanced_{safe}_agg.csv")

    plt.figure(figsize=(8, 6))
    plt.plot(sizes, train_means, 'o-', label='训练集')
    plt.plot(sizes, val_means, 'o-', label='验证集')
    plt.fill_between(sizes, np.array(train_means) - np.array(train_stds),
                     np.array(train_means) + np.array(train_stds), alpha=0.15)
    plt.fill_between(sizes, np.array(val_means) - np.array(val_stds),
                     np.array(val_means) + np.array(val_stds), alpha=0.15)
    sub = "（固定验证集）" if use_fixed_val else "（交替抽样）"
    plt.title(f"{name} 学习曲线 {sub}")
    plt.xlabel("训练样本数");
    plt.ylabel("准确率");
    plt.grid(True, linestyle='--', alpha=0.35)
    plt.legend(loc='lower right')

    out_path = out_dir / f"{name}_learning_curve.png"
    plt.tight_layout();
    plt.savefig(out_path, dpi=300);
    plt.close()
    return str(out_path)


# =========================
# 8. 混淆矩阵（横实纵预，列归一化）
# =========================
def plot_confusion_matrix_rates(y_true, y_pred, class_order, title, out_path: Path, csv_base: Path = None):
    cm_counts = confusion_matrix(y_true, y_pred, labels=class_order)
    cm_pred_rows_true_cols = cm_counts.T
    col_sums = cm_pred_rows_true_cols.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1
    cm_rates = cm_pred_rows_true_cols / col_sums

    if csv_base is not None and RUN_EXPORT_OOF:
        df_counts = pd.DataFrame(cm_pred_rows_true_cols, columns=[f"true_{c}" for c in class_order])
        df_counts.insert(0, "pred", class_order)
        _save_df_indexed(df_counts, Path(str(csv_base) + "_counts.csv"))

        df_rates = pd.DataFrame(cm_rates, columns=[f"true_{c}" for c in class_order])
        df_rates.insert(0, "pred", class_order)
        _save_df_indexed(df_rates, Path(str(csv_base) + "_rates.csv"))

    plt.figure(figsize=(8.5, 7.5))
    sns.heatmap(cm_rates, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_order, yticklabels=class_order,
                cbar=True, vmin=0, vmax=1, linewidths=.5, linecolor='white')
    plt.xlabel("True Label");
    plt.ylabel("Predicted Label");
    plt.title(title)
    plt.tight_layout();
    plt.savefig(out_path, dpi=300);
    plt.close()
    return str(out_path)


# =========================
# 9. PDF 报告
# =========================
def generate_pdf_report(results, best, images, out_dir: Path, timestamp: str, n_folds: int):
    pdf_path = out_dir / f"model_report_{timestamp}.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
    elements = []

    elements.append(Paragraph(f"机器学习模型评估报告 - {timestamp}", styles['Title']))
    elements.append(Spacer(1, 20))

    data = [['模型名称', 'CV均值', 'CV标准差', '训练准确率']]
    for r in sorted(results, key=lambda x: (x['cv_mean'] if pd.notna(x['cv_mean']) else -1), reverse=True):
        data.append([r['name'], f"{r['cv_mean']:.4f}", f"{r['cv_std']:.4f}", f"{r['train_acc']:.4f}"])
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 16))

    if images.get('cm_train'):
        elements.append(Paragraph("训练集混淆矩阵（列归一化：对角=正确率）", styles['Heading2']))
        elements.append(Image(images['cm_train'], width=500, height=380))
        elements.append(Spacer(1, 12))
    if images.get('cm_cv'):
        elements.append(
            Paragraph(f"{n_folds}折交叉验证 OOF 混淆矩阵（列归一化：对角=正确率，推荐参考）", styles['Heading2']))
        elements.append(Image(images['cm_cv'], width=500, height=380))
        elements.append(Spacer(1, 12))

    if images.get('learning_curves'):
        elements.append(Paragraph("学习曲线", styles['Heading2']))
        for name, path in images['learning_curves']:
            elements.append(Paragraph(name, styles['Heading3']))
            elements.append(Image(path, width=500, height=320))
            elements.append(Spacer(1, 8))

    doc.build(elements)
    print(f"PDF报告已生成：{pdf_path}")


# =========================
# 10. 预测（强制归入 11 类之一）
# =========================
def predict_force_label(best_model, feature_encoders, feature_cols, predict_path: Path, out_dir: Path,
                        label_encoder_y: LabelEncoder):
    """
    逻辑不变：依旧输出 Excel。
    新增：返回预测结果 DataFrame 列表（用于你后续绘图）。
    """
    if not predict_path:
        print("未提供预测路径，跳过预测步骤。")
        return []

    p = Path(predict_path)
    if not p.exists():
        print(f"预测路径不存在，跳过预测：{predict_path}")
        return []

    pred_dfs = []

    def _predict_one_file(xlsx_path: Path):
        dfp = pd.read_excel(xlsx_path)
        if not set(feature_cols).issubset(dfp.columns):
            missing = list(set(feature_cols) - set(dfp.columns))
            raise ValueError(f"预测文件缺失必要特征列：{missing}")

        Xp = dfp[feature_cols].copy()
        for col, le in feature_encoders.items():
            vals = Xp[col].astype(str).str.strip().values
            known = set(le.classes_)
            vals = np.array([v if v in known else '<UNK>' for v in vals], dtype=str)
            Xp[col] = le.transform(vals)

        pred_idx = best_model.predict(Xp)
        pred_labels = label_encoder_y.inverse_transform(pred_idx)

        out = dfp.copy()
        out['type_predicted'] = pred_labels
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"{xlsx_path.stem}_prediction_{ts}.xlsx"
        out.to_excel(out_path, index=False)
        print(f"预测结果已保存：{out_path}")

        pred_dfs.append(out)

    if p.is_dir():
        files = sorted([f for f in p.iterdir()
                        if f.suffix.lower() in ('.xlsx', '.xls') and not f.name.startswith('~$')])
        if not files:
            print(f"目录下没有Excel文件：{p}")
            return []
        for f in files:
            _predict_one_file(f)
    else:
        _predict_one_file(p)

    return pred_dfs


# =========================
# 11. 安全 OOF 预测（返回折号）
# =========================
def oof_predict_with_optional_sample_weight(estimator, X, y, cv, model_name):
    y_oof = np.empty_like(y)
    fold_ids = np.empty_like(y, dtype=int)
    for fold_no, (tr_idx, te_idx) in enumerate(cv.split(X, y), start=1):
        est = clone(estimator)
        fit_kwargs = {}
        if model_name == 'Gradient Boosting' and USE_CLASS_BALANCING:
            sw = compute_sample_weight(class_weight='balanced', y=y[tr_idx])
            fit_kwargs = {'histgradientboostingclassifier__sample_weight': sw}
        est.fit(X.iloc[tr_idx], y[tr_idx], **fit_kwargs)
        y_oof[te_idx] = est.predict(X.iloc[te_idx])
        fold_ids[te_idx] = fold_no
    return y_oof, fold_ids


# =========================
# 新增绘图 1：预测集散点图（SD vs mean；lg(T) vs mean）
# =========================
def _build_points_from_prediction_df(pred_df: pd.DataFrame,
                                     label_col: str = "type_predicted"):
    need_any = ["DT", "M1", "S2", label_col]
    missing = [c for c in need_any if c not in pred_df.columns]
    if missing:
        raise ValueError(f"预测表缺少必要列：{missing}")

    rows = []

    def _is_missing(v):
        return v is None or (isinstance(v, float) and np.isnan(v)) or (pd.isna(v))

    for i, r in pred_df.iterrows():
        dt = r["DT"]
        m1 = r["M1"]
        s2 = r["S2"]
        lab = r[label_col]

        # 是否具备第二个峰的信息
        has_peak2 = ("M2" in pred_df.columns) and ("S1" in pred_df.columns) and (not _is_missing(r.get("M2"))) and (
            not _is_missing(r.get("S1")))

        if has_peak2:
            m2 = r["M2"]
            s1 = r["S1"]
            # 点1：mean=M1, SD=S1
            rows.append({
                "row_index": i,
                "peak_id": 1,
                "DT": float(dt),
                "mean": float(m1),
                "SD": float(s1),
                "lgT": float(np.log10(dt)) if float(dt) > 0 else np.nan,
                label_col: lab,
            })
            # 点2：mean=M2, SD=S2
            rows.append({
                "row_index": i,
                "peak_id": 2,
                "DT": float(dt),
                "mean": float(m2),
                "SD": float(s2),
                "lgT": float(np.log10(dt)) if float(dt) > 0 else np.nan,
                label_col: lab,
            })
        else:
            # 1 个峰：mean=M1, SD=S2
            rows.append({
                "row_index": i,
                "peak_id": 1,
                "DT": float(dt),
                "mean": float(m1),
                "SD": float(s2),
                "lgT": float(np.log10(dt)) if float(dt) > 0 else np.nan,
                label_col: lab,
            })

    pts = pd.DataFrame(rows)
    # 清理无穷/非法
    pts = pts.replace([np.inf, -np.inf], np.nan).dropna(subset=["mean", "SD", "lgT"])
    return pts


def plot_prediction_scatter_plots(pred_dfs: list[pd.DataFrame], out_dir: Path, timestamp: str):
    if not pred_dfs:
        print("⚠️ 没有预测结果 DataFrame，跳过预测散点图。")
        return

    # 合并多个预测文件（如果是目录预测）
    dfp_all = pd.concat(pred_dfs, ignore_index=True)

    pts = _build_points_from_prediction_df(dfp_all, label_col="type_predicted")
    pts = pts.dropna(subset=["mean"])  # mean 为空就没法画

    # 1) SD vs mean
    fig1 = out_dir / f"scatter_SD_vs_mean_{timestamp}.png"
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=pts, x="mean", y="SD", hue="type_predicted", s=40)
    plt.title("Prediction set: SD vs mean")
    plt.xlabel("mean")
    plt.ylabel("SD")
    plt.ylim(0, 4)
    plt.tight_layout()
    plt.savefig(fig1, dpi=300)
    plt.close()
    print(f"✅ 已保存散点图：{fig1}")

    # 2) lg(T) vs mean
    fig2 = out_dir / f"scatter_lgT_vs_mean_{timestamp}.png"
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=pts, x="mean", y="lgT", hue="type_predicted", s=40)
    plt.title("Prediction set: lg(T) vs mean")
    plt.xlabel("mean")
    plt.ylabel("lg(T)")
    plt.tight_layout()
    plt.savefig(fig2, dpi=300)
    plt.close()
    print(f"✅ 已保存散点图：{fig2}")


# =========================
# 2：模型评估表格图片
# =========================
def plot_model_evaluation_table_image(results: list[dict], out_dir: Path, timestamp: str):
    if not results:
        print("⚠️ results 为空，跳过模型评估表格图。")
        return

    df = pd.DataFrame([{
        "Model": r["name"],
        "CV mean": r["cv_mean"],
        "CV std": r["cv_std"],
        "Train acc": r["train_acc"]
    } for r in results]).sort_values("CV mean", ascending=False)

    # 格式化显示
    df_show = df.copy()
    for c in ["CV mean", "CV std", "Train acc"]:
        df_show[c] = df_show[c].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")

    out_path = out_dir / f"model_evaluation_table_{timestamp}.png"
    fig, ax = plt.subplots(figsize=(10, max(2.5, 0.45 * (len(df_show) + 1))))
    ax.axis("off")

    table = ax.table(
        cellText=df_show.values,
        colLabels=df_show.columns.tolist(),
        cellLoc="center",
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.35)

    plt.title("Model evaluation summary", pad=16)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ 已保存模型评估表格图：{out_path}")


# =========================
#  3：平行坐标图（训练集 true_label）
# =========================
def plot_parallel_coordinates_from_training(X_raw: pd.DataFrame, y_raw: pd.Series,
                                            feature_cols: list[str], out_dir: Path, timestamp: str):
    try:
        from pandas.plotting import parallel_coordinates
    except Exception as e:
        print(f"⚠️ 无法导入 pandas.plotting.parallel_coordinates：{e}")
        return

    dfp = X_raw[feature_cols].copy()
    dfp["label"] = y_raw.astype(str).values

    df_plot = dfp.copy()
    for c in feature_cols:
        v = pd.to_numeric(df_plot[c], errors="coerce")
        vmin, vmax = np.nanmin(v.values), np.nanmax(v.values)
        if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
            df_plot[c] = (v - vmin) / (vmax - vmin)
        else:
            df_plot[c] = v

    out_path = out_dir / f"parallel_coordinates_train_{timestamp}.png"
    plt.figure(figsize=(11, 6))
    parallel_coordinates(df_plot.dropna(subset=feature_cols), class_column="label", cols=feature_cols, alpha=0.25)
    plt.title("Training set: Parallel coordinates (normalized for visualization)")
    plt.xlabel("Features")
    plt.ylabel("Normalized value")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✅ 已保存平行坐标图：{out_path}")


# =========================
# 主流程
# =========================
def main():
    np.random.seed(42)
    out_dir = Path(OUTPUT_DIR)
    _ensure_dir(out_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_path = out_dir / "debug_log.txt"
    if log_path.exists():
        log_path.unlink()

    # 1) 加载训练数据
    print("Loading training data from Excel folder...")
    df = load_excel_folder(DATA_DIR)
    debug_dataset_snapshot(df, out_dir, stage="loaded_all", feature_cols=None, target_col=None)

    if 'IsNoise' in df.columns:
        before = len(df)
        df = df[df['IsNoise'] == False].reset_index(drop=True)
        print(f"根据 IsNoise==False 过滤后：{before} -> {len(df)}")
    debug_dataset_snapshot(df, out_dir, stage="after_IsNoise_filter", feature_cols=None, target_col=None)
    print(f"训练数据样本数：{len(df)}")

    # 2) 选列
    X_raw, y_raw, feature_cols, target_col = select_features_target(df)
    debug_dataset_snapshot(df, out_dir, stage="after_select_cols", feature_cols=feature_cols, target_col=target_col)

    # 3) 编码
    X, feature_encoders = encode_features(X_raw)
    le_y = LabelEncoder()
    y = le_y.fit_transform(y_raw.astype(str))
    class_names = list(le_y.classes_)

    _save_df(pd.DataFrame({'class_index': range(len(class_names)), 'class_name': class_names}),
             out_dir / "debug_class_index_name_map.csv")
    vc_enc = pd.Series(y).value_counts().sort_index()
    _save_df(pd.DataFrame({'class_index': vc_enc.index, 'count': vc_enc.values}),
             out_dir / "debug_after_label_encode_target_counts.csv")

    # ======= 平行坐标数据导出（训练集本身，不含预测）=======
    if EXPORT_PARALLEL_DATA:
        y_true_str = np.asarray(y_raw.astype(str).values, dtype=str)
        export_parallel_coordinates_data_from_training_itself(
            X_raw=X_raw,
            X_encoded=X,
            y_true_str=y_true_str,
            feature_cols=feature_cols,
            out_dir=out_dir,
            timestamp=timestamp,
            source_df=df
        )

    # 4) KFold CV（用于模型选择）
    cv_kfold = make_stratified_cv(y, max_splits=MAX_CV_SPLITS, random_state=42)
    debug_cv_split_distribution(np.asarray(X), np.asarray(y), cv_kfold, class_names, out_dir)

    results, best = None, None
    if RUN_MODEL_SELECTION:
        results, best = train_evaluate_models(X, y, cv_kfold)

        # 评估表（Excel）
        eval_df = pd.DataFrame([{
            '模型名称': r['name'],
            'CV均值': r['cv_mean'],
            'CV标准差': r['cv_std'],
            '训练准确率': r['train_acc'],
            '参数': str(r['params'])
        } for r in results]).sort_values('CV均值', ascending=False)
        eval_path = out_dir / f"model_evaluation_{timestamp}.xlsx"
        eval_df.to_excel(eval_path, index=False)
        print(f"评估结果已保存：{eval_path}")

        if RUN_FEATURE_IMPORT:
            export_feature_importance_if_any(best['model'], feature_cols, out_dir, best['name'])
            for r in results:
                export_feature_importance_if_any(r['model'], feature_cols, out_dir, r['name'])
    else:
        print("⚠️ 未运行模型选择（RUN_MODEL_SELECTION=False），后续依赖最佳模型的模块将被跳过。")

    # 6) 学习曲线（可选）
    learning_curves = []
    if RUN_LEARNING_CURVES and RUN_MODEL_SELECTION:
        def _plot_one(r):
            if LC_IMPL == "sklearn":
                p = plot_learning_curve_sklearn(r['model'], r['name'], X, y, out_dir)
            else:
                p = plot_learning_curve_balanced(
                    estimator=r['model'], name=r['name'],
                    X=X, y=y, class_names=class_names, cv=cv_kfold, out_dir=out_dir,
                    train_fracs=LC_BAL_TRAIN_FRACS, repeats=LC_BAL_REPEATS,
                    use_fixed_val=False, val_size=LC_SSS_TEST_SIZE,
                    random_state=42, with_replacement=True
                )
            learning_curves.append((r['name'], p))

        if PLOT_LEARNING_CURVES_FOR == "best_only":
            _plot_one(best)
        else:
            for r in results:
                _plot_one(r)
        print("学习曲线生成：", [Path(p).name for _, p in learning_curves])
    elif RUN_LEARNING_CURVES and not RUN_MODEL_SELECTION:
        print("⚠️ 需要先进行模型选择以获得模型实例，已跳过学习曲线。")

    # 7) 混淆矩阵（Train & CV OOF）
    images = {}
    if RUN_CONFUSION_MATS and RUN_MODEL_SELECTION:
        best_model = best['model']
        y_pred_train = best_model.predict(X)
        cm_train_path = out_dir / "confusion_matrix_train.png"
        plot_confusion_matrix_rates(
            le_y.inverse_transform(y),
            le_y.inverse_transform(y_pred_train),
            class_order=class_names,
            title=f"{best['name']} Confusion Matrix (Train Set, Column Normalized",
            out_path=cm_train_path,
            csv_base=out_dir / "confusion_matrix_train"
        )
        images['cm_train'] = str(cm_train_path)

        y_pred_oof, fold_ids = oof_predict_with_optional_sample_weight(best_model, X, y, cv_kfold, best['name'])
        cm_cv_path = out_dir / "confusion_matrix_cv.png"
        plot_confusion_matrix_rates(
            le_y.inverse_transform(y),
            le_y.inverse_transform(y_pred_oof),
            class_order=class_names,
            title=f"{best['name']} {cv_kfold.get_n_splits()}-Fold CV OOF Confusion Matrix (Column Normalized)",
            out_path=cm_cv_path,
            csv_base=out_dir / "confusion_matrix_cv"
        )
        images['cm_cv'] = str(cm_cv_path)

        if RUN_EXPORT_OOF:
            oof_df = pd.DataFrame({
                'index': np.arange(len(y)),
                'fold': fold_ids,
                'y_true': le_y.inverse_transform(y),
                'y_pred': le_y.inverse_transform(y_pred_oof),
            })
            oof_csv = out_dir / f"oof_predictions_{timestamp}.csv"
            _save_df_indexed(oof_df, oof_csv)
            print(f"OOF逐样本已导出：{oof_csv}")
    elif RUN_CONFUSION_MATS and not RUN_MODEL_SELECTION:
        print("⚠️ 需要最佳模型才能绘制混淆矩阵，已跳过。")

    # 9) PDF 报告（可选）
    if RUN_PDF_REPORT and RUN_MODEL_SELECTION:
        generate_pdf_report(results, best, images, out_dir, timestamp, n_folds=cv_kfold.get_n_splits())
    elif RUN_PDF_REPORT and not RUN_MODEL_SELECTION:
        print("⚠️ 需要评估结果与混淆矩阵才能生成PDF，已跳过。")

    # 10) 预测（可选）
    pred_dfs = []
    if RUN_PREDICTION and RUN_MODEL_SELECTION:
        pred_dfs = predict_force_label(
            best_model=best['model'],
            feature_encoders=feature_encoders,
            feature_cols=feature_cols,
            predict_path=PREDICT_PATH,
            out_dir=out_dir,
            label_encoder_y=le_y
        )
    elif RUN_PREDICTION and not RUN_MODEL_SELECTION:
        print("⚠️ 需要最佳模型才能进行预测，已跳过。")

    # =========================
    # 新增绘图输出（你提出的 3 项）
    # =========================
    if RUN_EXTRA_PLOTS:
        try:
            # (1) 预测集散点图（使用 type_predicted 分颜色）
            if pred_dfs:
                plot_prediction_scatter_plots(pred_dfs, out_dir=out_dir, timestamp=timestamp)
            else:
                print("⚠️ 未生成预测结果 DataFrame（可能 RUN_PREDICTION=False），跳过预测散点图。")

            # (2) 模型评估表格图片
            if results:
                plot_model_evaluation_table_image(results, out_dir=out_dir, timestamp=timestamp)
            else:
                print("⚠️ results 为空，跳过模型评估表格图。")

            # (3) 平行坐标图（训练集 true_label）
            plot_parallel_coordinates_from_training(
                X_raw=X_raw,
                y_raw=y_raw,
                feature_cols=feature_cols,
                out_dir=out_dir,
                timestamp=timestamp
            )

        except Exception as e:
            print(f"⚠️ 额外绘图模块失败：{e}")
            traceback.print_exc()

    print("\n🎉 结束。输出目录：", out_dir.resolve())


if __name__ == "__main__":
    main()