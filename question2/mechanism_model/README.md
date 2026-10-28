# 第二问：机制激活逻辑回归（当前版本）

本目录是重新建模后的主版本。上一级 analyze.py 是早期单站日尺度基线，不是本版本入口。

## 运行

当前环境无需再安装依赖：

```bash
cd /workspace/macap
MPLCONFIGDIR=/workspace/macap-setup/mpl OMP_NUM_THREADS=2 /workspace/macap-env/bin/python question2/mechanism_model/run.py
/workspace/macap-env/bin/python question2/mechanism_model/predict.py --rain24 1 --antecedent48 10 --degree72 30 --gate 0.5
```

迁移到其他环境时，在仓库根目录使用Python 3.12创建虚拟环境并安装本目录requirements.txt。路径均由脚本位置推导，不依赖Windows路径。源数据应保持仓库中的相对位置。执行run.py会重新生成outputs，耗时通常几十秒。

`predict.py`输入的gate是机制评分（0至1），不是已知灾害类型编码；也可以用`--site`指定outputs/site_features.csv中的site值。P24为评估前0—24小时降雨，antecedent48为前24—72小时降雨，degree72为前72小时正温小时和除以24。时间必须和原始MET文件一致，时区尚待源数据说明确认。预测结果为未校准相对评分，不能解释为真实发生概率。

## 实现和检验

- run.py：数据核查、冰川空间特征、空间分块机制分类、带权触发回归、时间隔离验证、空间与时间联合留出、两类区块bootstrap、阈值分析、独立案例。
- estimators.py：使用SciPy实现非负系数约束的L2逻辑回归；无约束基线使用scikit-learn。
- predict.py：加载保存模型，在新进程独立计算评分。
- 第二问_机制模型与检验.md：建模方案、公式、实测成绩及限制。

脚本内执行有效检查：过去窗口因果性（更改评估时刻之后的观测不改变特征）、72小时训练测试隔离、日期不重叠、有效日期权重、非负模型数值单调性、模型保存与加载一致性。run.py抛错即为运行失败，不静默跳过必需数据。全体输入文件及shapefile附属文件的SHA256保存在outputs/input_manifest.json，方便核对复现版本。

固定C=0.1、固定随机种子42；决策树深度2、最小叶节点100条站点日记录。没有使用测试集选择超参数。空间网格为UTM45N中20km固定网格，不是独立流域；邻接块仍可能相关。

## 输出阅读顺序

1. summary.json：所有检验成绩、样本计数、系数及稳定性。
2. validation.png：时间留出ROC、逐日评分及回归系数。
3. trigger_coefficients.csv、gate_coefficients.csv：可复核参数。
4. conditional_thresholds.csv：条件临界面示例；范围标记不能忽略。
5. training_block_bootstrap.csv：临界面不稳定性。
6. independent_cases.csv：2025-07-08、2026-08-26零时评分。
7. inventory_features.csv、site_features.csv、daily_weather.csv、site_day_panel.csv：模型输入。
8. catalog_match_audit.csv：带日期点与84个范围的距离审计，不是自动类型匹配。
9. model.joblib：机制门控与物理约束触发模型；unconstrained_model.joblib：无约束触发基线。

## 关键解释

84个范围只有降雨型、冰水混合型、冰雪融水型、冰湖溃决型，没有冰崩型。发生日期台账仅有沟谷/山坡分类。当前能监督拟合的是“降雨主导/冰雪参与”的门控，无法识别四种独立失稳机制。尤其不能把冰水混合型自动当成冰崩型。

当前非负约束模型将冰川交互系数压到零；这说明该标签组合没有支持额外激活效应，不能说明冰川在自然界中不起作用。模型退化为气象触发基线是需要报告的结果，不应手动加权制造“机制有效”的结论。物理阈值和实际运行预警尚未验证。
