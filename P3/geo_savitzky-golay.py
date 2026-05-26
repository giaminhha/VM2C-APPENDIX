import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# ==============================================================================
# 1. CẤU HÌNH HỆ THỐNG VÀ THAM SỐ (CONFIG PARAMETERS)
# ==============================================================================

INPUT_FILE = "data_3.csv"  

# --- CẤU HÌNH TẦNG 1: LỌC GAI HÌNH HỌC ---
MIN_ANGLE_DEG = 40.0
DIST_GATE = 10.0
IQR_K_FACTOR = 2

# --- CẤU HÌNH TẦNG 2: SAVITZKY-GOLAY (KHÔNG TRỄ PHA) ---
SG_WINDOW_SIZE = 15  
SG_POLY_ORDER = 2    

# --- CẤU HÌNH ĐÓN ĐẦU ONLINE (TỐI ƯU HÓA ĐỂ ÉP SÁT AE GUYÊN THỦ) ---
DEGREE_X = 2  
DEGREE_Y = 2  
# Tăng từ 30 -> 40 để làm đầm gia tốc, giúp ổn định dự báo tầm xa (t=20s)
FIT_WINDOW_SIZE = 28

# Các mốc thời gian tuyệt đối cần ép chết sai số
EVAL_TIMES = [5, 10, 20]

file_name_without_ext, _ = os.path.splitext(INPUT_FILE)
OUTPUT_CLEAN_CSV = f"{file_name_without_ext}_denoised.csv"
TEST_AFTER_T0_FILE = f"{file_name_without_ext}_after_t0.csv"


# ==============================================================================
# 2. HÀM TRỢ GIÚP CHUYỂN ĐỔI (HELPERS)
# ==============================================================================
def convert_polar_to_cartesian(r, theta_deg):
    theta_rad = np.radians(theta_deg)
    return r * np.cos(theta_rad), r * np.sin(theta_rad)


def compute_baseline_iqr_threshold(df_raw, k_factor):
    x, y = convert_polar_to_cartesian(df_raw["range_m"].values, df_raw["bearing_deg"].values)
    dx = np.diff(x)
    dy = np.diff(y)
    distances = np.sqrt(dx**2 + dy**2)
    q1 = np.percentile(distances, 25)
    q3 = np.percentile(distances, 75)
    return k_factor * (q3 - q1)


# ==============================================================================
# 3. BỘ LỌC HÌNH HỌC + SAVITZKY-GOLAY
# ==============================================================================
def filter_and_smooth_savitzky_golay(df_raw, iqr_threshold):
    df = df_raw.copy()
    df["x"], df["y"] = convert_polar_to_cartesian(df["range_m"], df["bearing_deg"])
    records = df.to_dict(orient="records")
    
    stage1_records = [records[0]]
    k = 1
    while k < len(records) - 1:
        p_prev = stage1_records[-1]
        p_curr = records[k]
        p_next = records[k + 1]
        
        dist_x_to_next = np.sqrt((p_next["x"] - p_curr["x"])**2 + (p_next["y"] - p_curr["y"])**2)
        u = np.array([p_prev["x"] - p_curr["x"], p_prev["y"] - p_curr["y"]])
        v = np.array([p_next["x"] - p_curr["x"], p_next["y"] - p_curr["y"]])
        norm_u, norm_v = np.linalg.norm(u), np.linalg.norm(v)
        vertex_angle = 180.0
        if norm_u >= DIST_GATE and norm_v >= DIST_GATE:
            cos_a = np.clip(np.dot(u, v) / (norm_u * norm_v), -1.0, 1.0)
            vertex_angle = np.degrees(np.arccos(cos_a))

        if (vertex_angle < MIN_ANGLE_DEG) or (dist_x_to_next > iqr_threshold):
            dt_total = p_next["time_s"] - p_prev["time_s"]
            dt_step = p_curr["time_s"] - p_prev["time_s"]
            ratio = dt_step / dt_total
            interpolated = p_curr.copy()
            interpolated["x"] = p_prev["x"] + ratio * (p_next["x"] - p_prev["x"])
            interpolated["y"] = p_prev["y"] + ratio * (p_next["y"] - p_prev["y"])
            stage1_records.append(interpolated)
        else:
            stage1_records.append(p_curr)
        k += 1
    stage1_records.append(records[-1])
    
    df_stage1 = pd.DataFrame(stage1_records)

    w_len = SG_WINDOW_SIZE
    if w_len >= len(df_stage1):
        w_len = len(df_stage1) - 1 if (len(df_stage1) - 1) % 2 != 0 else len(df_stage1) - 2
    if w_len < 3: w_len = 3

    df_stage1["x_clean"] = savgol_filter(df_stage1["x"].values, window_length=w_len, polyorder=SG_POLY_ORDER)
    df_stage1["y_clean"] = savgol_filter(df_stage1["y"].values, window_length=w_len, polyorder=SG_POLY_ORDER)

    return df_stage1


# ==============================================================================
# 4. HỒI QUY ĐỘNG LỰC HỌC TƯƠI & TRỰC QUAN HÓA TÁCH ẢNH
# ==============================================================================
def process_kinematics_and_predict_high_precision(df_raw, df_clean, path_test_after_t0):
    print(f"\n==================================================")
    print(f" KHỞI CHẠY BỘ ĐÓN ĐẦU CAO CẤP VỚI WINDOW = {FIT_WINDOW_SIZE}")
    print(f"==================================================")
    
    t_origin = df_clean["time_s"].max()
    
    # Trích xuất cửa sổ tối ưu hóa động học tươi
    df_stable = df_clean.tail(min(len(df_clean), FIT_WINDOW_SIZE)).copy()
    
    t_train_shifted = df_stable["time_s"].values - t_origin
    x_train = df_stable["x_clean"].values
    y_train = df_stable["y_clean"].values
    
    poly_coeff_x = np.polyfit(t_train_shifted, x_train, deg=DEGREE_X)
    poly_coeff_y = np.polyfit(t_train_shifted, y_train, deg=DEGREE_Y)
    
    poly_v_x = np.polyder(poly_coeff_x, 1)
    poly_v_y = np.polyder(poly_coeff_y, 1)
    poly_a_x = np.polyder(poly_coeff_x, 2)
    poly_a_y = np.polyder(poly_coeff_y, 2)
    
    t_all_shifted = df_clean["time_s"].values - t_origin
    df_clean["vx"] = np.polyval(poly_v_x, t_all_shifted)
    df_clean["vy"] = np.polyval(poly_v_y, t_all_shifted)
    df_clean["ax"] = np.polyval(poly_a_x, t_all_shifted)
    df_clean["ay"] = np.polyval(poly_a_y, t_all_shifted)
    
    df_export = df_clean[["time_s", "x_clean", "y_clean", "vx", "vy", "ax", "ay"]].copy()
    df_export.columns = ["time_s", "x", "y", "vx", "vy", "ax", "ay"]
    output_path_clean = OUTPUT_CLEAN_CSV
    df_export.to_csv(output_path_clean, index=False, encoding="utf-8-sig")
    print(f"[+] Đã xuất file động học mượt: {output_path_clean}")
    
    # --- CHUẨN BỊ DỮ LIỆU ĐỂ PLOT ANH ---
    x_raw, y_raw = convert_polar_to_cartesian(df_raw["range_m"].values, df_raw["bearing_deg"].values)

    # --------------------------------------------------------------------------
    # ẢNH 1: SAU KHI LỌC HÌNH HỌC TẦNG 1
    # --------------------------------------------------------------------------
    plt.figure(figsize=(5, 3.5), dpi=100)
    plt.scatter(x_raw, y_raw, color="red", s=15, alpha=0.4, label="Dữ liệu Thô")
    plt.plot(x_raw, y_raw, color="red", linestyle=":", alpha=0.25)
    
    # df_clean["x"] và df_clean["y"] chính là tọa độ sau tầng lọc hình học 1
    plt.plot(df_clean["x"], df_clean["y"], color="orange", alpha=0.9, linewidth=2.0, label="Sau Lọc Hình Học")
    plt.scatter(df_clean["x"], df_clean["y"], color="darkorange", s=10, alpha=0.5)
    
    plt.title("Kết Quả Lọc Hình Học (Tầng 1)", fontsize=10, fontweight='bold')
    plt.xlabel("Tọa độ X (m)", fontsize=10)
    plt.ylabel("Tọa độ Y (m)", fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left", fontsize=10)
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------------------------
    # ẢNH 2: SAU KHI LỌC HÌNH HỌC + SAVITZKY-GOLAY TẦNG 2
    # --------------------------------------------------------------------------
    plt.figure(figsize=(5, 3.5), dpi=100)
    plt.scatter(x_raw, y_raw, color="red", s=15, alpha=0.4, label="Dữ liệu Thô")
    plt.plot(x_raw, y_raw, color="red", linestyle=":", alpha=0.25)
    
    # df_clean["x_clean"] và df_clean["y_clean"] là kết quả sau cả 2 tầng
    plt.plot(df_clean["x_clean"], df_clean["y_clean"], color="cyan", alpha=0.9, linewidth=2.0, label="Lọc Hình Học + SG")
    plt.scatter(df_clean["x_clean"], df_clean["y_clean"], color="teal", s=10, alpha=0.5)
    
    plt.title("Kết Quả Sau Lọc + Mượt SG (Tầng 2)", fontsize=10, fontweight='bold')
    plt.xlabel("Tọa độ X (m)", fontsize=10)
    plt.ylabel("Tọa độ Y (m)", fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left", fontsize=10)
    plt.tight_layout()
    plt.show()


# ==============================================================================
# LUỒNG CHẠY CHÍNH (MAIN EXECUTION)
# ==============================================================================
if __name__ == "__main__":
    path_input = INPUT_FILE
    path_test = TEST_AFTER_T0_FILE

    if not os.path.exists(path_input):
        print(f"[LỖI] Không tìm thấy tập dữ liệu: {path_input}")
        exit()

    df_raw = pd.read_csv(path_input)
    iqr_threshold = compute_baseline_iqr_threshold(df_raw, IQR_K_FACTOR)
    df_clean_output = filter_and_smooth_savitzky_golay(df_raw, iqr_threshold)
    
    process_kinematics_and_predict_high_precision(df_raw, df_clean_output, path_test)