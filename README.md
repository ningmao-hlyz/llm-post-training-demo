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
| base (Qwen2.5-0.5B-Instruct) | 97.0% | 0.0% | 0.375 | 0.085 | **0.139** |
| + LoRA SFT (本文) | **100%** | **98.5%** | **0.959** | **0.969** | **0.964** |

**一句话结论**: base 能模仿 JSON 的"形"(格式合法率 97%), 但完全不知道任务协议——
意图标签自编(`查询天气`/`请求`)、槽位键名自编(`location`/`时间`)、时间不规范化
(`晚上十点` 而非 `22:00`)。LoRA-SFT 用 **1.75% 可训练参数**把任务协议烧进权重:
意图准确率 0% → 98.5%, 槽位 F1 13.9% → 96.4%。典型对比见
[compare_examples.md](results/compare_examples.md)。

训练 loss 曲线(225 步, 78 秒, final loss 0.0002): `results/loss_curve.png`

通用能力回归(5 个通识问题, 人工检查): 自我认知/事实/翻译/代码均正常, 无灾难性遗忘,
见 [regression_lora.md](results/regression_lora.md)。

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

## 踩坑实录(真实调试过程)

**坑 1: chat template 的 `{% generation %}` 标记**

最初用 conversational 格式 + `assistant_only_loss=True`(只对 assistant 回复算 loss),
报错 `at least one example has no assistant tokens`。根因: assistant mask 依赖
chat template 中的 `{% generation %}` 关键字, **Qwen2.5 的模板没有**(Qwen3 有),
TRL 只对已知模型族自动 patch。解法: 改用 **prompt-completion 格式**
(`prompt=[system,user], completion=[assistant]`), TRL 对该格式默认只对
completion 算 loss, 效果等价且不依赖模板标记。

**坑 2: PEFT 原地注入导致对照组污染**

`infer_compare.py` 中先加载 base 再 `PeftModel.from_pretrained(base, path)`,
之后 `gen(base)` 的输出和 lora 完全一样——因为 **PEFT 的 from_pretrained 是原地注入**,
LoRA 层直接替换了 base 引用内部的 Linear 层, "对照组"实际走的也是激活 adapter 的前向。
等同于 A/B 实验中对照组偷吃了试验药。解法: base 推理放在
`with lora.disable_adapter():` 上下文中执行。修复后 base 暴露真实水平
(intent 0%), 与独立进程评测的结果自洽。

**坑 3: 评测驱动数据迭代**

首轮评测 slot F1 仅 0.933, 排查发现 `volume.set` 的 ground truth 存在自相矛盾:
同样句式"音量设置成40/70", direction 槽位随机 up/down——话语里没有"大/小",
方向无从判断, 模型学不会是数据的错。修复(direction 槽位仅在话语出现
"大/小"时存在)后重训, slot F1 0.933 → **0.964**。
教训: **指标异常时先审数据, 再怪模型**。

## 为什么是这套技术栈

- **TRL SFTTrainer**: HuggingFace 官方后训练框架, conversational 数据自动套
  chat template 并只对 assistant 部分算 loss, 是开源 SFT 的事实标准。
- **PEFT LoRA**: 参数高效微调事实标准, 适配器可独立管理/合并/热插拔。
- **Qwen2.5-0.5B-Instruct**: 小模型验证全链路, 工业界同流程换更大模型即可。

## 后续可扩展方向

- 用 lm-evaluation-harness 跑 CMMLU/C-Eval 子集做标准化通用回归
- DPO/ORPO 偏好优化阶段(后训练完整链路: SFT → RLHF/DPO)
- 多轮对话与拒答意图(out-of-scope detection)
