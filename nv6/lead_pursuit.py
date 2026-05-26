import os
import glob
import numpy as np
import pandas as pd

# ==========================================
# CẤU HÌNH HỆ THỐNG
# ==========================================
DATA_DIR = ""
FILES = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))[::]

# --- Tham số Mô phỏng ---
VB_MAX_SPEED = 100.0          # Vận tốc tối đa của tên lửa/người đuổi (B)
SMOOTH_K = 5                  # Số lượng điểm dữ liệu dùng để hồi quy làm mịn vận tốc
MAX_PRED_ITER = 12            # Số vòng lặp tối đa để tính điểm đón đầu
CATCH_RADIUS = 20.0           # Bán kính đánh chặn thành công
TIME_LIMIT = 60.0             # Thời gian rượt đuổi tối đa cho phép
CLOSE_DIST_THRESHOLD = 40.0   # (Giữ lại cho tương thích cấu hình)
EMA_ALPHA = 0.5               # Hệ số làm mịn lũy thừa (Exponential Moving Average)

# ==========================================
# CÁC HÀM TOÁN HỌC & HỖ TRỢ
# ==========================================

def estimate_velocity(pos: np.ndarray, time_arr: np.ndarray, i: int, k: int) -> np.ndarray:
    """
    Ước lượng vector vận tốc dựa trên k điểm dữ liệu gần nhất bằng phương pháp hồi quy tuyến tính.
    """
    j = max(0, i - k + 1)
    tt = time_arr[j:i + 1]
    px = pos[j:i + 1, 0]
    py = pos[j:i + 1, 1]

    if len(tt) < 2:
        return np.zeros(2)

    vx = np.polyfit(tt, px, 1)[0]
    vy = np.polyfit(tt, py, 1)[0]
    return np.array([vx, vy])


def predict_intercept_point(pos_A: np.ndarray, pos_B: np.ndarray, vel_A: np.ndarray, vB_speed: float, max_iter: int) -> np.ndarray:
    """
    Dự đoán điểm đánh chặn (intercept point) bằng phương pháp lặp để làm dữ liệu tham khảo.
    """
    if vB_speed <= 1e-12:
        return pos_A.copy()

    pred = pos_A.copy()
    for _ in range(max_iter):
        tau = np.linalg.norm(pred - pos_B) / vB_speed
        pred = pos_A + vel_A * tau
    return pred


def compute_lead_pursuit_velocity(pos_A: np.ndarray, pos_B: np.ndarray, vel_A: np.ndarray, dt: float) -> np.ndarray:
    """
    Tính toán vector vận tốc cho thuật toán Đón đầu Thuần túy (Pure Lead Pursuit).
    Sử dụng phương trình bậc 2 để tìm chính xác điểm cắt góc trong tương lai.
    """
    R_vec_real = pos_A - pos_B
    dist_ab = np.linalg.norm(R_vec_real)

    if dist_ab <= 1e-12 or dt <= 0:
        return np.zeros(2)

    # 1. BẮT DÍNH MỤC TIÊU Ở VỊ TRÍ TƯƠNG LAI (Discrete Next-Frame Catch)
    # Ngăn chặn hiện tượng bay vọt lố qua mục tiêu do thời gian rời rạc
    A_next = pos_A + vel_A * dt
    R_vec_next = A_next - pos_B
    req_speed = np.linalg.norm(R_vec_next) / dt

    if req_speed <= VB_MAX_SPEED:
        return R_vec_next / dt 

    # 2. THUẬT TOÁN LEAD PURSUIT (TOÀN TRÌNH)
    # Giải phương trình bậc 2 để tìm thời gian va chạm (tau)
    vA_mag_sq = np.dot(vel_A, vel_A)
    vB_mag_sq = VB_MAX_SPEED ** 2
    
    a = vB_mag_sq - vA_mag_sq
    b = -2.0 * np.dot(R_vec_real, vel_A)
    c = -(dist_ab ** 2)

    tau = -1.0 
    
    if abs(a) < 1e-6:
        # Xử lý trường hợp đặc biệt khi vận tốc 2 bên bằng nhau (phương trình suy biến thành bậc 1)
        if b > 0: 
            tau = -c / b
    else:
        delta = b**2 - 4*a*c
        if delta >= 0:
            t1 = (-b + np.sqrt(delta)) / (2*a)
            t2 = (-b - np.sqrt(delta)) / (2*a)
            # Lọc lấy các nghiệm thời gian trong tương lai (t > 0)
            roots = [t for t in (t1, t2) if t > 0]
            if roots: 
                tau = min(roots)

    # 3. RA LỆNH VẬN TỐC
    if tau > 0:
        # Tìm thấy điểm cắt góc tương lai -> Lao tới với tốc độ tối đa
        predicted_pos_A = pos_A + vel_A * tau
        R_vec_pred = predicted_pos_A - pos_B
        R_mag_pred = np.linalg.norm(R_vec_pred)
        desired_v = (R_vec_pred / R_mag_pred) * VB_MAX_SPEED
    else:
        # Nếu mục tiêu có khả năng chạy thoát (không có điểm cắt), chĩa thẳng mũi đuổi theo
        desired_v = (R_vec_real / dist_ab) * VB_MAX_SPEED

    return desired_v


# ==========================================
# LOGIC MÔ PHỎNG CHÍNH
# ==========================================

def simulate_one(file_path: str) -> dict:
    """Thực thi mô phỏng một vòng lặp truy đuổi dựa trên tệp CSV."""
    df = pd.read_csv(file_path)

    t = df["time_s"].to_numpy(dtype=float)
    r = df["range_m"].to_numpy(dtype=float)
    theta = np.deg2rad(df["bearing_deg"].to_numpy(dtype=float))

    A_raw = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    n_steps = len(t)
    
    if n_steps == 0:
        raise ValueError(f"Tệp dữ liệu trống: {file_path}")

    B = np.zeros((n_steps, 2))
    P = np.zeros((n_steps, 2))
    V = np.zeros((n_steps, 2))
    VB_speed_log = np.zeros(n_steps)
    dist_log = np.full(n_steps, np.nan)

    A_ema = np.zeros((n_steps, 2))
    A_ema[0] = A_raw[0] 

    success = False
    stop_idx = n_steps - 1

    for i in range(n_steps - 1):
        elapsed = t[i] - t[0] + 1.0

        if i > 0:
            A_ema[i] = EMA_ALPHA * A_raw[i] + (1.0 - EMA_ALPHA) * A_ema[i - 1]

        V[i] = estimate_velocity(A_ema, t, i, SMOOTH_K)
        P[i] = predict_intercept_point(A_raw[i], B[i], V[i], VB_MAX_SPEED, MAX_PRED_ITER)
        dist_log[i] = np.linalg.norm(A_raw[i] - B[i])

        if dist_log[i] < CATCH_RADIUS:
            success = True
            stop_idx = i
            break

        if elapsed > TIME_LIMIT:
            stop_idx = i
            break

        dt = t[i + 1] - t[i]

        desired_v = compute_lead_pursuit_velocity(A_raw[i], B[i], V[i], dt)

        B[i + 1] = B[i] + desired_v * dt
        VB_speed_log[i] = np.linalg.norm(desired_v)

    dist_log[stop_idx] = np.linalg.norm(A_raw[stop_idx] - B[stop_idx])

    if not success and dist_log[stop_idx] < CATCH_RADIUS:
        success = True

    if stop_idx > 0 and np.all(A_ema[stop_idx] == 0):
        A_ema[stop_idx] = EMA_ALPHA * A_raw[stop_idx] + (1.0 - EMA_ALPHA) * A_ema[stop_idx - 1]

    V[stop_idx] = estimate_velocity(A_ema, t, stop_idx, SMOOTH_K)
    P[stop_idx] = predict_intercept_point(A_raw[stop_idx], B[stop_idx], V[stop_idx], VB_MAX_SPEED, MAX_PRED_ITER)
    
    if stop_idx > 0:
        VB_speed_log[stop_idx] = VB_speed_log[stop_idx - 1]

    final_time_s = t[stop_idx] - t[0] + 1.0

    summary = {
        "file": os.path.basename(file_path),
        "status": "success" if success else "fail",
        "final_time_s": final_time_s,
        "final_dist_m": dist_log[stop_idx],
        "min_dist_m": np.nanmin(dist_log[:stop_idx + 1]),
        "pred_dist_last_m": np.linalg.norm(P[stop_idx] - B[stop_idx]),
        "|vA|_last": np.linalg.norm(V[stop_idx]),
        "|vB|_last": VB_speed_log[stop_idx],
    }

    return {
        "t": t,
        "stop_idx": stop_idx,
        "success": success,
        "file_path": file_path,
        "summary": summary,
    }


def analyze_results(sim_results: list) -> None:
    """Thống kê và phân loại các tệp thất bại theo từng phân khúc."""
    success_times = []
    failed_files = []
    
    fail_counts = {
        "0-249": 0,
        "250-499": 0,
        "500-749": 0,
        "750-999": 0
    }

    for sim in sim_results:
        if sim["success"]:
            t = sim["t"]
            stop_idx = sim["stop_idx"]
            chase_time = t[stop_idx] - t[0] + 1.0
            success_times.append(chase_time)
        else:
            file_path = sim["file_path"]
            failed_files.append(file_path)
            base_name = os.path.basename(file_path)
            
            try:
                file_idx = int(base_name.split('_')[2])
                if 0 <= file_idx <= 249: fail_counts["0-249"] += 1
                elif 250 <= file_idx <= 499: fail_counts["250-499"] += 1
                elif 500 <= file_idx <= 749: fail_counts["500-749"] += 1
                elif 750 <= file_idx <= 999: fail_counts["750-999"] += 1
            except (IndexError, ValueError):
                pass

    avg_time = np.mean(success_times) if success_times else float("nan")

    print("===== TỔNG KẾT PHÂN TÍCH =====")
    print(f"Thành công:  {len(success_times)}")
    print(f"Thất bại:    {len(failed_files)}")
    print(f"Tg TB (s):   {avg_time:.2f} s")
    
    print("\n=== SỐ LƯỢNG THẤT BẠI THEO PHÂN KHÚC ===")
    for range_name, count in fail_counts.items():
        print(f"Khoảng {range_name:<7}: {count} tệp")

    print("\nDANH SÁCH TỆP THẤT BẠI = [")
    for f in failed_files:
        print(f'    "{f}",')
    print("]")


# ==========================================
# KHỞI CHẠY CHƯƠNG TRÌNH
# ==========================================
if __name__ == "__main__":
    if not FILES:
        print(f"Không tìm thấy tệp CSV nào trong {DATA_DIR}. Vui lòng kiểm tra lại cấu hình.")
    else:
        sim_results = [simulate_one(f) for f in FILES]
        summary_df = pd.DataFrame([s["summary"] for s in sim_results])
        
        print("Mẫu dữ liệu thống kê (10 tệp đầu tiên):")
        print(summary_df.head(10).to_string(index=False, float_format=lambda x: f"{x:8.2f}"))
        print("\n" + "="*40 + "\n")
        
        analyze_results(sim_results)