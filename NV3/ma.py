import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Nạp và xử lý dữ liệu
df = pd.read_csv('data_2.csv').sort_values('time_s').reset_index(drop=True)
df['theta_rad'] = np.radians(df['bearing_deg'])
df['x'] = df['range_m'] * np.cos(df['theta_rad'])
df['y'] = df['range_m'] * np.sin(df['theta_rad'])

# Hàm lọc nhiễu gai (Bắt buộc với MA vì rất nhạy cảm với outlier)
def remove_spikes(df, max_speed=150):
    df_clean = df.copy()
    speed_f = (np.sqrt(df_clean['x'].diff(-1).abs()**2 + df_clean['y'].diff(-1).abs()**2) / df_clean['time_s'].diff(-1).abs()).fillna(0)
    speed_b = (np.sqrt(df_clean['x'].diff(1).abs()**2 + df_clean['y'].diff(1).abs()**2) / df_clean['time_s'].diff(1).abs()).fillna(0)
    is_outlier = (speed_b > max_speed) & (speed_f > max_speed)
    return df_clean[~is_outlier].reset_index(drop=True)

df_clean = remove_spikes(df)

# Tính MA cơ bản
window_ma = 7
df_clean['x_ma'] = df_clean['x'].rolling(window=window_ma, min_periods=1).mean()
df_clean['y_ma'] = df_clean['y'].rolling(window=window_ma, min_periods=1).mean()

# PLOT: Dữ liệu sạch + MA
plt.figure(figsize=(5, 3.5))
plt.plot(df_clean['x'], df_clean['y'], 'k.', alpha=0.3, markersize=6, label='Dữ liệu sạch')
plt.plot(df_clean['x_ma'], df_clean['y_ma'], 'b--', linewidth=2, label=f'MA (window={window_ma})')

plt.title('Lọc Trung bình trượt - MA cho data_2')
plt.xlabel('Tọa độ X (m)')
plt.ylabel('Tọa độ Y (m)')
plt.legend(fontsize=8)
plt.grid(True, linestyle=':', alpha=0.7)

plt.tight_layout()
plt.savefig('plot_ma_data_2.png', dpi=100)
plt.close()