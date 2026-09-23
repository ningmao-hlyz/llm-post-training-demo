# LLM Post-Training Demo: SFT + LoRA + 评测 完整链路

> 面试腾讯 AI 全栈(后训练方向)后, 为深入理解后训练全链路所做的完整实践。
> 从业务数据构造 → LoRA SFT → base vs 微调对比 → 自动化评测, 全链路跑通。
>
> 基座模型: **Qwen2.5-0.5B-Instruct** | 训练框架: **TRL SFTTrainer + PEFT** | 单卡 ~6GB 显存即可复现

## 任务设计: 语音助手 NLU(意图识别 + 槽位抽取)

贴真实业务场景(语音助手): 用户说一句话, 模型输出结构化 JSON。

```
输入: "帮我麻烦把客厅的空调开一下好嘛"
输出: {"intent": "device.control", "slots": {"device": "ac.livingroom", "action": "on"}}
```

覆盖 8 个意图(闹钟/天气/音乐/音量/提醒/日程/电话/智能家居控制), 槽位含必选与可选,
时间口语表达统一规范化(`"早上八点" -> "08:00"`), 模型需学到语义映射而非逐字复制。

## 全链路

```
Qwen2.5-0.5B-Instruct (基座, 冻结)
        │
        ├── ① build_dataset.py   程序化构造 1200 train + 200 eval(句式 held-out)
        │
        ├── ② train_sft.py       TRL SFTTrainer + LoRA(r=16, 挂全部7类线性层)
        │                        只对 assistant 回复计算 loss
        │
        ├── ③ infer_compare.py   base vs LoRA 同题对比 → results/compare_examples.md
        │
        └── ④ evaluate.py        三层评测: 格式合法率 / 意图准确率 / 槽位 P-R-F1
                                 + 通用能力回归(防灾难性遗忘)
```

## 评测结果(200 条 held-out, 句式与训练集完全不相交)

| 模型 | 格式合法率 | 意图准确率 | 槽位 P | 槽位 R | 槽位 F1 |
|---|---|---|---|---|---|
| base (Qwen2.5-0.5B-Instruct) | TODO | TODO | TODO | TODO | TODO |
| + LoRA SFT (本文) | TODO | TODO | TODO | TODO | TODO |

<!-- RESULTS_PLACEHOLDER -->

训练 loss 曲线: `results/loss_curve.png`(训练完成后回填)

## 快速开始

```bash
# 1. 构造数据(纯 Python, 无需 GPU)
python data/build_dataset.py

# 2. SFT 训练(单卡 ~6GB 显存, 0.5B + LoRA 约 10-20 分钟)
pip install -r requirements.txt
python scripts/train_sft.py

# 3. 对比推理 + 评测
python scripts/infer_compare.py
python scripts/evaluate.py --model base
python scripts/evaluate.py --model lora --regression
```

国内环境建议 `export HF_ENDPOINT=https://hf-mirror.com` 加速模型下载。

## 关键设计决策(面试考点)

1. **数据 held-out**: 评测集句式模板与训练集完全不相交, 防止模型背模板,
   评测考的是泛化; 槽位存在性由话语内容决定(明确提及才输出)。
2. **LoRA 而非全量微调**: 0.5B 全量微调需 20GB+ 显存(梯度+Adam 状态),
   LoRA(r=16, 7 类线性层)可训练参数仅 ~8.8M(1.8%), 训练完合并回 W 推理零开销。
3. **评测先行**: 格式合法率(SFT 最立竿见影指标) + 意图准确率/槽位 F1(业务指标)
   + 通用回归(防灾难性遗忘), 对应工业界"训练前先定指标"。
4. **时间规范化**: 词表无歧义(裸"六点"可能指早晚, 已剔除), 口语→HH:MM 标准化,
   数据审计脚本验证槽位值 100% 可从话语溯源。

## 为什么是这套技术栈

- **TRL SFTTrainer**: HuggingFace 官方后训练框架, conversational 数据自动套
  chat template 并只对 assistant 部分算 loss, 是开源 SFT 的事实标准。
- **PEFT LoRA**: 参数高效微调事实标准, 适配器可独立管理/合并/热插拔。
- **Qwen2.5-0.5B-Instruct**: 小模型验证全链路, 工业界同流程换更大模型即可。

## 后续可扩展方向

- 用 lm-evaluation-harness 跑 CMMLU/C-Eval 子集做标准化通用回归
- DPO/ORPO 偏好优化阶段(后训练完整链路: SFT → RLHF/DPO)
- 多轮对话与拒答意图(out-of-scope detection)
