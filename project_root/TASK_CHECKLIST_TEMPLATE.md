# GPU 实验任务清单 — 交接格式

> **用途**：本文件定义了开发端 → GPU 执行端的任务交接标准格式。
> 每个实验批次创建一个 `TASK_<批次名>.md` 文件，按此格式填写后提交 git，
> GPU 端 pull 后按清单顺序执行并逐项回填结果。

---

## 使用方式

1. 开发端在 `project_root/` 下创建 `TASK_<批次名>.md`
2. 按下方格式填写任务
3. `git commit` + `git push`
4. GPU 端 `git pull`，按清单执行
5. GPU 端每完成一项回填 `[DONE]` 及结果摘要，commit + push
6. 开发端 pull 后验收

---

## 任务清单格式

```markdown
# 任务清单: <批次名称>

**创建时间**: YYYY-MM-DD HH:MM
**创建人**: <姓名>
**预计完成**: YYYY-MM-DD
**状态**: [ ] 待执行 / [~] 执行中 / [x] 已完成

---

## 任务 01 — <任务简述>

- **状态**: [ ] 待执行 / [~] 执行中 / [x] 已完成 / [!] 异常
- **优先级**: P0(必须) / P1(重要) / P2(可选)
- **模型方法**: P0 / P1 / P4 / P11 / P12 / 其他________
- **场景**: S1_balanced / S2_clustered / S3_maneuver / mixed(S1+S2+S3)
- **模型种子列表**: 0,1,2,3,4
- **训练种子范围**: 10-69(train) / 70-79(val) / 80-89(test)
- **训练参数**: epochs=80, lr=1e-3, batch_size=64
- **特殊配置**: (如有，在此注明)

### 执行命令

\```bash
cd project_root
python main.py --preset <preset_name> --mode train --epochs 80 --device cuda
\```

或直接用脚本:
\```bash
cd project_root
python scripts/run_phase1_nominal_benchmark.py --methods P11 --scenes S1 --model-seeds 0,1,2,3,4
\```

### 验收标准

- [ ] 结果目录存在: `project_root/results/<run_dir>/`
- [ ] `metrics.json` 存在且包含 RMSE/MAE 等关键指标
- [ ] 关键指标在合理范围内:
  - val RMSE < ________
  - test RMSE < ________
- [ ] 无 NaN/Inf 异常值
- [ ] 模型权重文件存在 (如有)

### 执行记录 (GPU端回填)

- **实际开始**: YYYY-MM-DD HH:MM
- **实际结束**: YYYY-MM-DD HH:MM
- **结果路径**: `project_root/results/<run_dir>/`
- **关键指标**:
  - val RMSE: ________
  - test RMSE: ________
- **异常备注**: (无则填"无")
```

---

## 验收标准说明

| 检查项 | 说明 | 必须 |
|--------|------|------|
| `metrics.json` | 包含完整指标（RMSE, MAE, 分传感器指标等） | 是 |
| 模型权重 | `.pth` 或 `.pt` 文件存在 | 是 |
| 无 NaN/Inf | 所有指标为有限数值 | 是 |
| 指标数值 | 与预期范围对比（由开发端填写预期值） | 是 |
| 日志文件 | `train.log` 或等价日志 | 建议 |

---

## 异常处理

如果任务出现异常，GPU 端需在任务状态标记 `[!]` 并在执行记录中注明：

1. **OOM**: 显存不足 → 减小 batch_size 重试
2. **NaN Loss**: 数值不稳定 → 降低 lr 重试
3. **文件缺失**: 缺少数据集或配置 → 检查 git pull 是否完整
4. **其他**: 详细描述报错信息，commit 后通知开发端
