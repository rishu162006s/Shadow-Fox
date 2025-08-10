

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_PATH = '/mnt/data/delhiaqi.csv'
OUTDIR = Path('/mnt/data/aqi_analysis_outputs')
PLOTS_DIR = OUTDIR / 'plots'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('.', '_')

dt_col = None
for c in df.columns:
    if c in ['datetime','date','timestamp','time'] or 'date' in c or 'time' in c:
        dt_col = c; break
if dt_col is None:
    raise ValueError('No datetime column found in CSV.')
df[dt_col] = pd.to_datetime(df[dt_col], errors='coerce')
df = df.dropna(subset=[dt_col]).sort_values(dt_col).reset_index(drop=True)
df.rename(columns={dt_col:'datetime'}, inplace=True)

cpcb_breakpoints = {'pm2_5': [(0, 30, 0, 50), (31, 60, 51, 100), (61, 90, 101, 200), (91, 120, 201, 300), (121, 250, 301, 400), (251, 9999, 401, 500)], 'pm10': [(0, 50, 0, 50), (51, 100, 51, 100), (101, 250, 101, 200), (251, 350, 201, 300), (351, 430, 301, 400), (431, 9999, 401, 500)], 'no2': [(0, 40, 0, 50), (41, 80, 51, 100), (81, 180, 101, 200), (181, 280, 201, 300), (281, 400, 301, 400), (401, 9999, 401, 500)], 'so2': [(0, 40, 0, 50), (41, 80, 51, 100), (81, 380, 101, 200), (381, 800, 201, 300), (801, 1600, 301, 400), (1601, 9999, 401, 500)], 'co': [(0, 1.0, 0, 50), (1.1, 2.0, 51, 100), (2.1, 10, 101, 200), (10.1, 17, 201, 300), (17.1, 34, 301, 400), (34.1, 9999, 401, 500)], 'o3': [(0, 50, 0, 50), (51, 100, 51, 100), (101, 168, 101, 200), (169, 208, 201, 300), (209, 748, 301, 400), (749, 9999, 401, 500)], 'nh3': [(0, 200, 0, 50), (201, 400, 51, 100), (401, 800, 101, 200), (801, 1200, 201, 300), (1201, 1800, 301, 400), (1801, 9999, 401, 500)]}

pollutants = list(cpcb_breakpoints.keys())
for pol in pollutants:
    if pol not in df.columns:
        df[pol] = np.nan

def compute_sub_index(pol,val):
    if pd.isna(val): return np.nan
    for bp_lo,bp_hi,a_lo,a_hi in cpcb_breakpoints[pol]:
        if bp_lo <= val <= bp_hi:
            return ((a_hi-a_lo)/(bp_hi-bp_lo))*(val-bp_lo)+a_lo
    return np.nan

for pol in pollutants:
    df[f'{pol}_sub'] = df[pol].apply(lambda x: compute_sub_index(pol,x))

sub_cols = [f'{pol}_sub' for pol in pollutants]
df['AQI'] = df[sub_cols].max(axis=1, skipna=True)
df['dominant_pollutant'] = df[sub_cols].idxmax(axis=1).str.replace('_sub','')

df['hour'] = df['datetime'].dt.hour
df['date'] = df['datetime'].dt.date
def daypart(h):
    if 6 <= h <= 11: return 'Morning'
    if 12 <= h <= 17: return 'Afternoon'
    return 'Night'
df['daypart'] = df['hour'].apply(daypart)

hourly_aqi = df.groupby('hour')['AQI'].mean().reindex(range(24))
daypart_avg = df.groupby('daypart')['AQI'].mean().reindex(['Morning','Afternoon','Night'])
corrs = df[pollutants + ['AQI']].corr()['AQI'].drop('AQI').sort_values(ascending=False)

# Save outputs and plots (same as notebook)
df.to_csv(OUTDIR/'delhiaqi_with_aqi_full.csv', index=False)
hourly_aqi.to_csv(PLOTS_DIR/'hourly_aqi.csv')
daypart_avg.to_csv(PLOTS_DIR/'daypart_avg.csv')
corrs.to_csv(PLOTS_DIR/'pollutant_corr_with_aqi.csv')

plt.figure(figsize=(10,4))
plt.plot(hourly_aqi.index, hourly_aqi.values, marker='o')
plt.title('Average AQI by Hour of Day')
plt.xlabel('Hour of day')
plt.ylabel('Average AQI')
plt.grid(True)
plt.xticks(range(0,24))
plt.tight_layout()
plt.savefig(PLOTS_DIR/'hourly_aqi.png')
plt.close()

plt.figure(figsize=(6,4))
plt.bar(daypart_avg.index, daypart_avg.values)
plt.title('Average AQI by Daypart')
plt.xlabel('Daypart')
plt.ylabel('Average AQI')
plt.tight_layout()
plt.savefig(PLOTS_DIR/'daypart_avg.png')
plt.close()

plt.figure(figsize=(8,4))
plt.bar(corrs.index, corrs.values)
plt.title('Correlation of pollutants with AQI (overall)')
plt.xlabel('Pollutant')
plt.ylabel('Correlation with AQI')
plt.tight_layout()
plt.savefig(PLOTS_DIR/'pollutant_corr_with_aqi.png')
plt.close()

print('Analysis complete. Outputs saved in', OUTDIR)
