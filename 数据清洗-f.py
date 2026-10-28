import pandas as pd
import numpy as np
import glob
import os
import warnings

warnings.filterwarnings('ignore')

# 统一的基础路径
base_path = r"D:\mju\2026-1\学习\竞赛\大数据挑战\2026年MathorCup大数据挑战赛-赛道B初赛\附件\data"


# ==========================================
# 1. 处理 FLUX（湍流）数据
# ==========================================
def clean_flux_data(path):
    file_list = glob.glob(os.path.join(path, "*[Ff][Ll][Uu][Xx]*.csv"))
    print(f"找到了 {len(file_list)} 个 FLUX 文件")

    df_list = []
    for f in file_list:
        df = pd.read_csv(f)
        time_cols = [col for col in df.columns if 'time' in str(col).lower()]
        if time_cols:
            df = df.rename(columns={time_cols[0]: 'Timestamp'})
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df = df.set_index('Timestamp')
            df_list.append(df)

    df_flux = pd.concat(df_list).sort_index()

    # 1. 强制转数字防报错
    for col in df_flux.columns:
        df_flux[col] = pd.to_numeric(df_flux[col], errors='coerce')

    # 2. 替换哨兵值
    df_flux = df_flux.replace([-9999, 9999, -99999, 99999, -999.9], np.nan)

    # 3. 利用 QC 列精准剔除废数据 (如果 QC 值 >= 8 代表是插值或缺失)
    if 'QC_LE' in df_flux.columns and 'LE(W/m2)' in df_flux.columns:
        df_flux.loc[df_flux['QC_LE'] >= 8, 'LE(W/m2)'] = np.nan
    if 'QC_H' in df_flux.columns and 'H(W/m2)' in df_flux.columns:
        df_flux.loc[df_flux['QC_H'] >= 8, 'H(W/m2)'] = np.nan
    if 'QC_USTAR' in df_flux.columns and 'USTAR(m/s)' in df_flux.columns:
        df_flux.loc[df_flux['QC_USTAR'] >= 8, 'USTAR(m/s)'] = np.nan

    # 4. 单位统一 (把 kPa 转为 hPa)
    if 'PA(kPa)' in df_flux.columns:
        df_flux['PA(kPa)'] = df_flux['PA(kPa)'] * 10

    # 5. 物理边界过滤
    if 'LE(W/m2)' in df_flux.columns:
        df_flux.loc[(df_flux['LE(W/m2)'] < -200) | (df_flux['LE(W/m2)'] > 1000), 'LE(W/m2)'] = np.nan
    if 'H(W/m2)' in df_flux.columns:
        df_flux.loc[(df_flux['H(W/m2)'] < -200) | (df_flux['H(W/m2)'] > 1000), 'H(W/m2)'] = np.nan

    # 6. FETCH 与 USTAR 过滤（夜间静稳和足迹污染）
    if 'FETCH_90(m)' in df_flux.columns:
        df_flux.loc[df_flux['FETCH_90(m)'] < 50, ['LE(W/m2)', 'H(W/m2)']] = np.nan
    if 'USTAR(m/s)' in df_flux.columns:
        df_flux.loc[df_flux['USTAR(m/s)'] < 0.1, ['LE(W/m2)', 'H(W/m2)']] = np.nan

    # 7. 删除 QC 列
    qc_cols = [col for col in df_flux.columns if 'QC' in str(col).upper()]
    df_flux = df_flux.drop(columns=qc_cols)

    # 时间截断
    df_flux = df_flux.loc[:'2026-08-27 08:00']

    # 8. 小规模缺失插值（只补 <= 3 小时的）
    cols_to_interpolate = ['Ux(m/s)', 'Uy(m/s)', 'Uz(m/s)', 'T_SONIC(deg C)',
                           'VPD(hPa)', 'H2O_density(g/m3)', 'LE(W/m2)', 'H(W/m2)', 'USTAR(m/s)', 'TKE(m2/s2)']
    cols_to_interpolate = [c for c in cols_to_interpolate if c in df_flux.columns]
    for col in cols_to_interpolate:
        df_flux[col] = df_flux[col].interpolate(method='linear', limit=3, limit_direction='both')

    print(f"✅ FLUX 清洗完毕，有效 LE 行数: {df_flux['LE(W/m2)'].notna().sum()}")
    return df_flux


# ==========================================
# 2. 处理 MET/RADM/SOIL 数据
# ==========================================
def clean_other_data(data_type, path):
    print(f"\n========== 正在处理 {data_type} 数据 ==========")
    file_list = glob.glob(os.path.join(path, f"*{data_type}*.csv"))

    df_list = []
    for f in file_list:
        df = pd.read_csv(f)
        time_cols = [col for col in df.columns if 'time' in str(col).lower()]
        if time_cols:
            df = df.rename(columns={time_cols[0]: 'Timestamp'})
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df = df.set_index('Timestamp')
            df_list.append(df)

    if not df_list:
        print(f"⚠️ 没找到 {data_type} 文件！")
        return pd.DataFrame()

    df = pd.concat(df_list).sort_index()

    # 1. 强转数字 + 替换哨兵值
    for col in df.columns:
        if 'QC' not in col.upper():
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.replace([9999.9, -9999, 9999, 99999], np.nan)

    # 2. 删掉 QC 列
    df = df.drop(columns=[c for c in df.columns if 'QC' in c.upper()])

    # 3. 物理合理性检验 (来自高阶思路提示)
    # 长波辐射在夜间可能出现负值，这是传感器漂移，予以修正
    if data_type == "RADM":
        if 'Rld(W/m2)' in df.columns:
            df.loc[df['Rld(W/m2)'] < 0, 'Rld(W/m2)'] = np.nan
        if 'Rlu(W/m2)' in df.columns:
            df.loc[df['Rlu(W/m2)'] < 0, 'Rlu(W/m2)'] = np.nan

    # 4. 截断时间
    df = df.loc[:'2026-08-27 08:00']

    # 5. 小规模插值
    df = df.interpolate(method='linear', limit=3, limit_direction='both')

    print(f"✅ {data_type} 处理完成！形状: {df.shape}")
    return df


# ==========================================
# 3. 执行与合并
# ==========================================
if __name__ == "__main__":
    # 清洗四类数据
    df_flux_final = clean_flux_data(base_path)
    df_met = clean_other_data("MET", base_path)
    df_radm = clean_other_data("RADM", base_path)
    df_soil = clean_other_data("SOIL", base_path)

    print("\n========== 正在合并所有数据 (带自动重命名) ==========")

    # 1. 给除了 Timestamp 之外的所有列加上前缀，彻底杜绝重名
    df_met_renamed = df_met.add_prefix('MET_')
    df_flux_renamed = df_flux_final.add_prefix('FLUX_')
    df_radm_renamed = df_radm.add_prefix('RADM_')
    df_soil_renamed = df_soil.add_prefix('SOIL_')

    # 2. 现在这四张表没有任何同名列了，放心 join
    df_master = df_met_renamed.join([df_flux_renamed, df_radm_renamed, df_soil_renamed], how='inner')

    print(f"🎉 终极数据库合并完成！最终形状: {df_master.shape}")

    # 保存
    df_master.to_csv("clead.csv", encoding='utf-8-sig')
    print("✅ 最终数据已保存为 clead.csv")