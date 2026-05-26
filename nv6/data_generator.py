import os
import glob
import numpy as np
import pandas as pd

# ==========================================
# CẤU HÌNH HỆ THỐNG & THAM SỐ MÔ PHỎNG
# ==========================================
OUT_DIR = ""
NUM_FILES = 1000
N_STEPS = 60
DT = 1.0

# --- Tham số Vận tốc (m/s) ---
PURSUER_VMAX = 100.0  # Vận tốc tối đa của hệ thống bám sát (dùng để đánh giá khả năng đánh chặn)
TARGET_VMAX = 85.0    # Vận tốc tối đa cho phép của mục tiêu
TARGET_VMIN = 20.0    # Vận tốc tối thiểu cho phép của mục tiêu

# --- Giới hạn Không gian & Thuật toán ---
MAX_ALLOWED_RANGE = 5000.0  # Khoảng cách tối đa trước khi mục tiêu bị coi là ra khỏi vùng kiểm soát
MAX_RETRIES = 500           # Số lần thử lại tối đa để tạo ra một quỹ đạo hợp lệ

# --- Thông số Cơ động (Ngoặt hướng) ---
CATCH_MARGIN_FACTOR = 0.90  # Hệ số dung sai an toàn khi tính toán khả năng đánh chặn
MIN_TURN_GAP = 5            # Khoảng thời gian tối thiểu (số bước) giữa hai lần ngoặt
TURN_DURATION_MIN = 3       # Thời gian thực hiện một pha ngoặt tối thiểu (số bước)
TURN_DURATION_MAX = 5       # Thời gian thực hiện một pha ngoặt tối đa (số bước)

# --- Cấu hình Nhiễu (Noise) ---
NOISE_BASE_ACCEL = 0.8        # Nhiễu gia tốc cơ bản trong trạng thái bay thẳng
NOISE_EXTRA_ACCEL = 0.3       # Nhiễu gia tốc bổ sung khi mục tiêu đang thực hiện ngoặt hướng
NOISE_STEP_VELOCITY = 0.5     # Nhiễu vận tốc ngẫu nhiên tại mỗi bước thời gian
NOISE_MEASURE_RANGE = 10.0    # Độ lệch chuẩn nhiễu đo lường khoảng cách (m)
NOISE_MEASURE_BEARING = 0.15  # Độ lệch chuẩn nhiễu đo lường góc phương vị (độ)

rng = np.random.default_rng()

# ==========================================
# CÁC HÀM XỬ LÝ TỆP TIN
# ==========================================
def clear_csv_files(folder_path: str) -> None:
    """
    Xóa toàn bộ các tệp dữ liệu .csv cũ trong thư mục đích để chuẩn bị cho lượt tạo mới.
    
    Args:
        folder_path (str): Đường dẫn tới thư mục cần dọn dẹp.
    """
    if not os.path.exists(folder_path):
        return
    for f in glob.glob(os.path.join(folder_path, "*.csv")):
        if os.path.isfile(f):
            os.remove(f)

# ==========================================
# CÁC HÀM HỖ TRỢ TOÁN HỌC & VẬT LÝ
# ==========================================
def clip_speed(v: np.ndarray, vmin: float = TARGET_VMIN, vmax: float = TARGET_VMAX) -> np.ndarray:
    """
    Chuẩn hóa vector vận tốc, đảm bảo độ lớn luôn nằm trong giới hạn [vmin, vmax].
    
    Args:
        v (np.ndarray): Vector vận tốc ban đầu (2D).
        vmin (float): Vận tốc tối thiểu.
        vmax (float): Vận tốc tối đa.
        
    Returns:
        np.ndarray: Vector vận tốc đã được điều chỉnh độ lớn.
    """
    speed = np.linalg.norm(v)
    
    # Xử lý trường hợp ngoại lệ khi mục tiêu đứng yên hoàn toàn
    if speed < 1e-12:
        return np.array([vmin, 0.0])
    
    # Chuẩn hóa nếu vượt ngưỡng
    if speed > vmax:
        return v * (vmax / speed)
    if speed < vmin:
        return v * (vmin / speed)
    return v

def rand_vec(speed_min: float, speed_max: float) -> np.ndarray:
    """
    Tạo ngẫu nhiên một vector 2D với hướng bất kỳ và độ lớn nằm trong khoảng cho trước.
    """
    speed = rng.uniform(speed_min, speed_max)
    ang = rng.uniform(-np.pi, np.pi)
    return speed * np.array([np.cos(ang), np.sin(ang)])

def significant_new_velocity(v_old: np.ndarray, vmax: float = TARGET_VMAX) -> np.ndarray:
    """
    Sinh ra một vector vận tốc mới, đảm bảo có sự thay đổi rõ rệt về hướng và độ lớn
    so với vector vận tốc cũ để mô phỏng các pha cơ động gắt của mục tiêu.
    """
    s_old = np.linalg.norm(v_old)
    a_old = np.arctan2(v_old[1], v_old[0]) if s_old > 1e-12 else rng.uniform(-np.pi, np.pi)

    # Thử tối đa 100 lần để tìm ra một góc ngoặt và tốc độ đủ lớn (thay đổi vector > 20m/s)
    for _ in range(100):
        s_new = rng.uniform(50.0, vmax)
        # Ép mục tiêu phải ngoặt một góc từ 67 đến 167 độ
        da = rng.uniform(np.deg2rad(67), np.deg2rad(167)) * rng.choice([-1, 1])
        a_new = a_old + da
        v_new = s_new * np.array([np.cos(a_new), np.sin(a_new)])
        
        if np.linalg.norm(v_new - v_old) > 20.0:
            return clip_speed(v_new, TARGET_VMIN, vmax)
            
    # Phương án dự phòng nếu không tìm được vector thỏa mãn sau 100 lần lặp
    return clip_speed(rand_vec(TARGET_VMIN, 67), TARGET_VMIN, vmax)

# ==========================================
# LOGIC TẠO DỮ LIỆU MÔ PHỎNG
# ==========================================
def generate_one_dataset(n_changes: int, file_path: str, outlier_prob: float) -> list:
    """
    Mô phỏng toàn bộ một quỹ đạo di chuyển của mục tiêu và lưu thành tệp CSV.
    
    Args:
        n_changes (int): Số lượng điểm ngoặt mong muốn trong quỹ đạo.
        file_path (str): Đường dẫn lưu tệp CSV.
        outlier_prob (float): Xác suất xuất hiện dữ liệu đo lường lỗi (outlier).
        
    Returns:
        list: Danh sách các mốc thời gian (bước) xảy ra điểm ngoặt.
    """
    # Không cho phép ngoặt ở 6 bước đầu tiên và 5 bước cuối cùng
    valid_change_range = np.arange(6, N_STEPS - 5)
    max_possible_changes = len(valid_change_range)
    n_changes = int(np.clip(n_changes, 0, max_possible_changes))
    
    for _ in range(MAX_RETRIES):
        change_points = []
        
        # Thiết lập các mốc thời gian ngoặt hướng ngẫu nhiên
        if n_changes > 0:
            possible_points = list(valid_change_range)
            for _ in range(n_changes):
                if not possible_points:
                    break
                cp = rng.choice(possible_points)
                change_points.append(cp)
                # Đảm bảo các lần ngoặt cách nhau một khoảng thời gian tối thiểu
                possible_points = [p for p in possible_points if abs(p - cp) >= MIN_TURN_GAP]
            change_points.sort()
            
        change_set = set(change_points)

        # Khởi tạo vị trí và vận tốc ban đầu
        pos = rand_vec(1000.0, min(4000.0, MAX_ALLOWED_RANGE - 500.0)) 
        vel = clip_speed(rand_vec(20.0, 80.0), TARGET_VMIN, TARGET_VMAX)
        
        rows = []
        out_of_bounds = False
        is_catchable = False 
        
        turning_steps_remaining = 0
        target_vel = vel.copy()
        
        acc = rng.normal(0.0, NOISE_BASE_ACCEL, size=2)

        for step in range(1, N_STEPS + 1):
            # Nếu chạm đến mốc thời gian bắt đầu chuyển hướng
            if step in change_set:
                target_vel = significant_new_velocity(vel, vmax=TARGET_VMAX)
                turning_steps_remaining = rng.integers(TURN_DURATION_MIN, TURN_DURATION_MAX + 1)
                # Tăng thêm nhiễu gia tốc trong quá trình cơ động
                acc = rng.normal(0.0, NOISE_BASE_ACCEL, size=2) + rng.normal(0.0, NOISE_EXTRA_ACCEL, size=2)
            elif turning_steps_remaining == 0:
                acc = rng.normal(0.0, NOISE_BASE_ACCEL, size=2)

            # Xử lý quá trình nội suy vận tốc chuyển tiếp trong lúc ngoặt
            if turning_steps_remaining > 0:
                current_speed = np.linalg.norm(vel)
                current_angle = np.arctan2(vel[1], vel[0])
                
                t_speed = np.linalg.norm(target_vel)
                t_angle = np.arctan2(target_vel[1], target_vel[0])
                
                angle_diff = (t_angle - current_angle + np.pi) % (2 * np.pi) - np.pi
                
                speed_step = (t_speed - current_speed) / turning_steps_remaining
                angle_step = angle_diff / turning_steps_remaining
                
                new_speed = current_speed + speed_step
                new_angle = current_angle + angle_step
                
                vel = new_speed * np.array([np.cos(new_angle), np.sin(new_angle)])
                vel = clip_speed(vel + acc * DT, TARGET_VMIN, TARGET_VMAX)
                
                turning_steps_remaining -= 1
            else:
                # Trạng thái bay ổn định, chỉ cộng thêm nhiễu nhỏ
                small_noise = rng.normal(0.0, NOISE_STEP_VELOCITY, size=2)
                vel = clip_speed(vel + acc * DT + small_noise, TARGET_VMIN, TARGET_VMAX)

            # Cập nhật vị trí và tính toán tọa độ cực
            pos = pos + vel * DT
            x, y = pos
            r_true = np.hypot(x, y)
            t_current = step * DT
            
            # Đánh giá xem hệ thống có khả năng bắt kịp mục tiêu không
            if r_true <= (PURSUER_VMAX * CATCH_MARGIN_FACTOR) * t_current:
                is_catchable = True

            # Hủy quỹ đạo nếu mục tiêu bay khỏi giới hạn không gian
            if r_true > MAX_ALLOWED_RANGE:
                out_of_bounds = True
                break
                
            bearing_true = np.degrees(np.arctan2(y, x))
            
            # --- Thêm nhiễu đo lường của cảm biến ---
            r_noisy = r_true + rng.normal(0.0, NOISE_MEASURE_RANGE)
            bearing_noisy = bearing_true + rng.normal(0.0, NOISE_MEASURE_BEARING)
            
            # --- Sinh dữ liệu Outlier ngẫu nhiên ---
            if rng.random() < outlier_prob:
                r_noisy += rng.choice([-1, 1]) * rng.uniform(45.0, 60.0)
                bearing_noisy += rng.choice([-1, 1]) * rng.uniform(1.0, 1.5)
                
            # Đảm bảo góc phương vị luôn nằm trong khoảng [-180, 180] và khoảng cách >= 0
            bearing_noisy = (bearing_noisy + 180) % 360 - 180
            r_noisy = max(0.0, r_noisy)

            rows.append([step, r_noisy, bearing_noisy])

        # Nếu quỹ đạo hợp lệ, tiến hành lưu trữ
        if not out_of_bounds and is_catchable:
            pd.DataFrame(rows, columns=["time_s", "range_m", "bearing_deg"]).to_csv(file_path, index=False)
            return change_points
            
    # Lưu kết quả tốt nhất có thể nếu đã dùng hết số lần thử lại
    print(f"Cảnh báo: Không thể tạo quỹ đạo thỏa mãn điều kiện cho {os.path.basename(file_path)}. Tiến hành lưu lần thử tốt nhất.")
    pd.DataFrame(rows, columns=["time_s", "range_m", "bearing_deg"]).to_csv(file_path, index=False)
    return change_points

# ==========================================
# KHỞI CHẠY CHƯƠNG TRÌNH
# ==========================================
def get_simulation_params(index: int) -> tuple[int, int, float]:
    """
    Trả về số điểm ngoặt (tối thiểu, tối đa) và xác suất sinh dữ liệu lỗi (outlier) 
    dựa trên chỉ số của tệp để mô phỏng các mức độ khó khác nhau.
    """
    if index < 250:
        return 0, 5, 0.0
    elif index < 500:
        return 0, 5, 0.05
    elif index < 750:
        return 6, 12, 0.0
    else:
        return 6, 12, 0.1

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    clear_csv_files(OUT_DIR)
    
    meta = []
    
    for i in range(NUM_FILES):
        min_c, max_c, outlier_prob = get_simulation_params(i)
        n_changes = int(rng.integers(min_c, max_c))
        
        file_path = os.path.join(OUT_DIR, f"data_A_{i:03d}_N{n_changes}.csv")
        
        cps = generate_one_dataset(n_changes, file_path, outlier_prob)
        meta.append((os.path.basename(file_path), len(cps), cps))

    for name, n_changes, cps in meta:
        print(f"{name}: Số ngoặt yêu cầu={n_changes}, Ngoặt thực tế={cps}")

if __name__ == "__main__":
    main()