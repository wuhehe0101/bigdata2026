import pandas as pd

# 读取灾害记录
df_disaster = pd.read_csv(r"D:\bigdadta\吉隆县历史灾害_干净版.csv")
df_disaster['发生日期'] = pd.to_datetime(df_disaster['发生日期'])

    # 查看年份分布
print("=== 34条灾害记录的年份分布 ===")
print(df_disaster['发生日期'].dt.year.value_counts().sort_index())

    # 读取气象数据
df_weather = pd.read_csv(r"D:\bigdadta\clead.csv")
df_weather['Timestamp'] = pd.to_datetime(df_weather['Timestamp'])
df_weather.set_index('Timestamp', inplace=True)

    # 看看气象数据的真实起止时间
print(f"\n气象数据有效起止时间: {df_weather.index.min()} 至 {df_weather.index.max()}")

    # 检查有多少灾害日落在 2021-2025 气象覆盖期内
overlap_count = sum((df_disaster['发生日期'] >= '2021-01-01') & (df_disaster['发生日期'] <= '2025-12-31'))
print(f"\n✅ 落在气象覆盖期(2021-2025)内的灾害记录数: {overlap_count} 条")