# LLM Post-Training Demo: SFT + LoRA + 评测 完整链路

> 从业务数据构造 → LoRA SFT → base vs 微调对比 → 自动化评测, 后训练全链路实践。
> 基座: **Qwen2.5-0.5B-Instruct** | 训练框架: **TRL SFTTrainer + PEFT** | 单卡 ~6GB 显存可复现
> 底层原理、参数直觉、踩坑根因的完整复盘见 [KNOWLEDGE.md](KNOWLEDGE.md)

## 任务: 语音助手 NLU(意图识别 + 槽位抽取)

用户说一句话, 模型输出结构化 JSON:

```
输入: "帮我麻烦把客厅的空调开一下好嘛"
输出: {"intent": "device.control", "slots": {"device": "ac.livingroom", "action": "on"}}
```

覆盖 8 个意图(闹钟/天气/音乐/音量/提醒/日程/电话/智能家居), 槽位含必选与可选,
时间口语统一规范化(`"早上八点" -> "08:00"`), 模型需学到语义映射而非逐字复制。

## 仓库结构

| 文件 | 作用 |
|---|---|
| `data/build_dataset.py` | 程序化构造数据集: 模板×词表, train/eval 句式池完全不相交(held-out) |
| `data/train.jsonl` | 1200 条训练数据, `{"messages": [system,user,assistant], "meta": GT}` |
| `data/eval.jsonl` | 200 条评测数据, 句式与训练集不相交 |
| `scripts/train_sft.py` | SFT 训练: TRL SFTTrainer + LoRA(r=16, 挂全部 7 类线性层), 只对 assistant 回复算 loss |
| `scripts/infer_compare.py` | base vs LoRA 同题对比推理 → `results/compare_examples.md` |
| `scripts/evaluate.py` | 三层评测(格式合法率/意图准确率/槽位 P-R-F1) + 通用能力回归 |
| `results/loss_curve.png` | 训练 loss 曲线(225 步, 0.72 → 0.0002) |
| `results/compare_examples.md` | base vs LoRA 逐条对比样例 |
| `results/eval_base.json` / `eval_lora.json` | 两个模型的量化评测报告 |
| `results/regression_lora.md` | 通用能力回归记录(防灾难性遗忘) |
| `KNOWLEDGE.md` | 知识笔记: SFT 机制 / LoRA 数学 / 坑的根因 / 评测设计 |
| `requirements.txt` | 依赖(trl / peft / transformers / datasets / matplotlib) |

## 全链路: 训练是怎么做的

```
Qwen2.5-0.5B-Instruct (基座, 冻结 98.25%)
   ① build_dataset.py   构造 1200 train + 200 eval(句式 held-out)
   ② train_sft.py       prompt-completion 格式 + LoRA SFT
                         225 步 / 78 秒(A10 24G) / final loss 0.0002
   ③ infer_compare.py   base vs LoRA 同题对比(对照组在 disable_adapter 下推理)
   ④ evaluate.py        三层评测 + 通用能力回归
```

关键决策:

1. **数据 held-out**: 评测句式与训练句式完全不相交, 防背模板, 考泛化;
   槽位存在性由话语内容决定(明确提及才输出)。
2. **LoRA 而非全量微调**: 可训练参数 8.8M(1.75%), 训练显存约为全量 1/8,
   训练完可合并回 W 推理零开销。
3. **prompt-completion 格式**: 显式标注"题目/答案"分界线,
   不依赖 chat template 的 `{% generation %}` 标记(见坑 1)。
4. **评测先行**: 三层指标 + 回归; 指标异常时先审数据再怪模型(见坑 3)。

## 训练配置逐项说明

| 参数 | 值 | 为什么 |
|---|---|---|
| `epochs` | 3 | 1200 条小数据 3 轮足够收敛, 更多有过拟合风险 |
| `learning_rate` | 2e-4 | LoRA 量级: 旁路是随机初始化的新参数, 无历史包袱, 可比全量微调(1e-5)大一个数量级 |
| `per_device_train_batch_size` | 8 | 单卡吞吐 |
| `gradient_accumulation_steps` | 2 | 攒 2 个 micro-batch 的梯度再更新, 有效 batch = 16 |
| `lora_r` | 16 | ΔW 秩上限; 任务简单, 16 绰绰有余(原论文许多任务 r≤8 即饱和) |
| `lora_alpha` | 32 (=2r) | 缩放 α/r 恒为 2.0, 解耦"秩大小"与"变化幅度" |
| `lora_dropout` | 0.05 | 旁路轻量正则 |
| `target_modules` | q/k/v/o + gate/up/down | attention 4 类 + FFN 3 类全挂, 比只挂 q/v 效果好 |
| `lr_scheduler_type` | cosine + warmup 10 步 | 前期 lr 从 0 线性爬升防初期震荡, 后期平滑衰减 |
| `completion_only_loss` | True | 只对 completion(assistant) 算 loss, SFT 的核心 |
| `bf16` | True | 半精度训练, 省显存提速(A10 支持) |

参数量账本: **可训练 = 549,888 × r = 8,798,208(1.75%)**, 其余 98.25% 冻结。

## 一条数据的底层链路(30 秒版)

```
train.jsonl 的 messages
  → 拆分: prompt=[system, user], completion=[assistant]
  → chat template 渲染: <|im_start|>user ... <|im_end|> <|im_start|>assistant ...
  → tokenize: 文本 → token id 序列
  → label mask: prompt 段 label=-100(CrossEntropyLoss 的 ignore_index),
                completion 段 label=token id
  → 前向 + cross-entropy: loss 只落在 assistant 段
```

模型每个位置都在预测"下一个 token", label mask 保证只有"答案区"计入 loss——
预训练学语言, SFT 学响应协议。完整机制(shift 预测 / B=0 初始化 / 低秩假设 /
16 bytes-per-param 显存账本)见 [KNOWLEDGE.md](KNOWLEDGE.md)。

## 训练过程实录: 3 个坑

**坑 1: `assistant_only_loss` 依赖 chat template 的 `{% generation %}` 标记**

最初用 conversational 格式 + `assistant_only_loss=True`, 报错
`at least one example has no assistant tokens`。根因: assistant 掩码的生成依赖
chat template 里的 `{% generation %}` 标记块, **Qwen2.5 的模板没有**(Qwen3 才有)。
解法: 改用 **prompt-completion 格式**, 分界线从"模板推断"变为"数据显式给定",
效果等价且不依赖模板特性。

**坑 2: PEFT 原地注入污染对照组**

`PeftModel.from_pretrained(base, path)` 会**原地**替换 base 内部的 Linear 层,
之后 `gen(base)` 走的也是激活 adapter 的前向——base 和 lora 输出完全一致,
"对照组偷吃了试验药"。解法: base 推理放进 `with lora.disable_adapter():` 上下文。
修复后 base 暴露真实水平(intent 0%), 与独立进程评测自洽。

**坑 3: 评测驱动数据迭代**

首轮 slot F1 0.933。错误分析发现 `volume.set` 的 direction 槽位自相矛盾:
"音量设置成40"(无方向语义)的 GT 随机标 up/down——话语不含方向信息, 模型注定学不会。
修复(direction 仅在话语出现"大/小"时存在)后重训, **0.933 → 0.964**。
教训: **指标异常先审数据, 再怪模型**。

## 训练结果(200 条 held-out, 句式与训练集完全不相交)

| 模型 | 格式合法率 | 意图准确率 | 槽位 P | 槽位 R | 槽位 F1 |
|---|---|---|---|---|---|
| base (Qwen2.5-0.5B-Instruct) | 97.0% | 0.0% | 0.375 | 0.085 | 0.139 |
| + LoRA SFT | **100%** | **98.5%** | **0.959** | **0.969** | **0.964** |

base 会模仿 JSON 的"形"(格式合法率 97%), 但完全不知道任务协议——意图标签自编、
槽位键名自编、时间不规范化。LoRA-SFT 用 1.75% 可训练参数把任务协议烧进权重:
意图 0% → 98.5%, 槽位 F1 13.9% → 96.4%。

- 训练: 225 步, 78 秒, final loss 0.0002 → `results/loss_curve.png`
- 逐条对比样例 → `results/compare_examples.md`
- 通用能力回归(自我认知/事实/翻译/推理/代码)正常, 无灾难性遗忘 → `results/regression_lora.md`

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

## 后续可扩展方向

- 用 lm-evaluation-harness 跑 CMMLU/C-Eval 子集做标准化通用回归
- DPO/ORPO 偏好优化阶段(后训练完整链路: SFT → RLHF/DPO)
- 多轮对话与拒答意图(out-of-scope detection)
