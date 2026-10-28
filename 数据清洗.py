import pandas as pd
import glob
import os

folder_path = r"D:\mju\2026-1\学习\竞赛\大数据挑战\2026年MathorCup大数据挑战赛-赛道B初赛\附件\data"



print(f"正在检查路径: {folder_path}")
print(f"该路径是否存在: {os.path.exists(folder_path)}")

if os.path.exists(folder_path):
    # 列出文件夹里所有的文件
    all_files = os.listdir(folder_path)
    print(f"\n文件夹里一共有 {len(all_files)} 个文件。前 10 个文件是：")
    for f in all_files[:10]:
        print(f"  - {f}")

    # 尝试匹配一下 MET 的文件
    print("\n尝试匹配 MET_Gyirong_*.csv ...")
    test_pattern = os.path.join(folder_path, "MET_Gyirong_*.csv")
    matched = glob.glob(test_pattern)
    print(f"匹配到了 {len(matched)} 个文件。")

    # 尝试匹配所有 csv 看看
    test_all_csv = glob.glob(os.path.join(folder_path, "*.csv"))
    print(f"文件夹里共有 {len(test_all_csv)} 个 .csv 文件。")

def load_csv_files(data_type):
    file_pattern = os.path.join(folder_path, f"{data_type}_Gyirong_*.csv")
    file_list = glob.glob(file_pattern)
    df_list = []

    for f in file_list:
        try:
            df = pd.read_csv(f)

            # ================= 核心修改区 =================
            # 寻找包含 'time' 或 'date' 的列（忽略大小写）
            time_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]

            if time_cols:
                # 如果找到了，统一把找到的第一列重命名为 Timestamp
                df = df.rename(columns={time_cols[0]: 'Timestamp'})
                df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            else:
                print(f"⚠️ 警告: 文件 {os.path.basename(f)} 中找不到时间列，已跳过！")
                continue
            # ==============================================

            df_list.append(df)
        except Exception as e:
            print(f"读取 {f} 失败: {e}")

    if df_list:
        return pd.concat(df_list, ignore_index=True)
    return pd.DataFrame()


# 重新执行读取
print("正在读取并合并 MET...")
df_met = load_csv_files("MET")

print("正在读取并合并 FLUX...")
df_flux = load_csv_files("FLUX")

print("正在读取并合并 RADM...")
df_radm = load_csv_files("RADM")

print("正在读取并合并 SOIL...")
df_soil = load_csv_files("SOIL")

# 横向合并
print("\n正在进行四大类型数据的横向合并...")
# 注意：如果某一种类型一个文件都没读到（空表），我们就跳过它
dfs_to_merge = [df for df in [df_met, df_flux, df_radm, df_soil] if not df.empty]

if dfs_to_merge:
    df_all = dfs_to_merge[0]
    for next_df in dfs_to_merge[1:]:
        df_all = df_all.merge(next_df, on='Timestamp', how='outer')

    df_all = df_all.sort_values('Timestamp').set_index('Timestamp')
    print(f"\n合并完成！数据表形状: {df_all.shape} (行数, 列数)")
else:
    print("\n❌ 所有文件都没读取成功，请检查文件夹路径和文件名！")

df_all.to_csv("jion_all.csv")