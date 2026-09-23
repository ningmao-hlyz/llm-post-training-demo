# -*- coding: utf-8 -*-
"""
SFT 训练: Qwen2.5-0.5B-Instruct + LoRA, 基于 TRL SFTTrainer。

用法:
    python scripts/train_sft.py [--output_dir output/qwen-nlu-lora]

要点(对应概念):
- 数据: data/train.jsonl 的 "messages" 列是 conversational 格式,
  TRL 会自动套用 tokenizer 的 chat template, 并只对 assistant
  回复部分计算 loss (SFT 的核心)。
- LoRA: 冻结全部基座参数, 只在 7 类线性层旁挂低秩矩阵 (r=16)。
- 产物: output_dir 下是 LoRA adapter 权重 (~几十MB, 可独立上传/合并)。
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_file", default="data/train.jsonl")
    p.add_argument("--output_dir", default="output/qwen-nlu-lora")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--lora_r", type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()

    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer
    import torch

    # 1. 加载基座与 tokenizer (bf16: 卡支持就用, 不支持退回 fp16)
    dtype = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=getattr(torch, dtype), device_map="auto"
    )
    print(f"[model] {MODEL_ID} | dtype={dtype}")

    # 2. 数据集: conversational 格式, TRL 自动 apply chat template
    ds = load_dataset("json", data_files=args.data_file, split="train")
    ds = ds.remove_columns([c for c in ds.column_names if c != "messages"])
    print(f"[data] {len(ds)} samples")

    # 3. LoRA 配置: 挂到 Transformer 全部 7 类线性层的旁路
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,          # 常用 2r 缩放
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",   # attention
            "gate_proj", "up_proj", "down_proj",      # FFN
        ],
    )
    trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"[lora] r={args.lora_r} (trainable params 会在 trainer 初始化后打印)")

    # 4. 训练配置
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        save_strategy="no",                   # demo 不存中间 checkpoint
        report_to="none",
        bf16=(dtype == "bf16"),
        fp16=(dtype == "fp16"),
        gradient_checkpointing=False,         # 0.5B 不需要
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds,
        peft_config=peft_config,
    )

    # 可训练参数占比 (见证 LoRA 的省钱效果)
    total = sum(p.numel() for p in trainer.model.parameters())
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"[lora] trainable={trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # 5. 训练
    trainer.train()

    # 6. 保存 adapter + tokenizer
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[done] adapter saved -> {args.output_dir}")

    # 7. 画 loss 曲线
    hist = trainer.state.log_history
    steps = [h["step"] for h in hist if "loss" in h]
    losses = [h["loss"] for h in hist if "loss" in h]
    if steps:
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
        ax.plot(steps, losses, color="#4B3FE3", linewidth=1.8)
        ax.set_xlabel("step")
        ax.set_ylabel("training loss")
        ax.set_title("SFT training loss (Qwen2.5-0.5B + LoRA)")
        ax.grid(alpha=0.3)
        out = results_dir / "loss_curve.png"
        fig.tight_layout()
        fig.savefig(out)
        print(f"[plot] loss curve -> {out}")
        final = losses[-1]
        print(f"[stats] final train loss = {final:.4f}")


if __name__ == "__main__":
    main()
