# -*- coding: utf-8 -*-
"""
对比推理: 同一批问题, base 模型 vs LoRA 微调后模型, 生成 Markdown 对比表。

用法:
    python scripts/infer_compare.py [--path output/qwen-nlu-lora] [--n 10]
"""
import argparse
import json
from pathlib import Path

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--path", default="output/qwen-nlu-lora")
    p.add_argument("--eval_file", default="data/eval.jsonl")
    p.add_argument("--n", type=int, default=10, help="每个意图抽几条")
    return p.parse_args()


def main():
    args = parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    use_bf16 = torch.cuda.is_bf16_supported()
    torch_dtype = torch.bfloat16 if use_bf16 else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch_dtype, device_map="auto"
    )
    base.eval()
    lora = PeftModel.from_pretrained(base, args.path)
    lora.eval()

    # 每个意图抽 n 条, 保证覆盖所有意图
    samples = [json.loads(l) for l in open(args.eval_file, encoding="utf-8")]
    by_intent = {}
    for s in samples:
        by_intent.setdefault(s["meta"]["intent"], []).append(s)
    picked = []
    for intent, group in by_intent.items():
        picked.extend(group[: args.n])

    def gen(model, messages):
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs, max_new_tokens=96, do_sample=False,
            temperature=None, top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
        gen_part = out[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(gen_part, skip_special_tokens=True).strip()

    lines = [
        "# Base vs LoRA-SFT 对比样例\n",
        f"- 基座: `{MODEL_ID}`",
        f"- 样例来源: `{args.eval_file}` (句式与训练集不相交)\n",
    ]
    for s in picked:
        msgs = s["messages"][:2]
        gt = json.dumps({"intent": s["meta"]["intent"], "slots": s["meta"]["slots"]},
                        ensure_ascii=False)
        # 坑: PeftModel.from_pretrained 是原地注入, base 引用的层已被替换,
        # 直接 gen(base) 走的也是激活 adapter 的前向(对照组被污染)。
        # 必须在 disable_adapter() 上下文中推理才是真正的基座行为。
        with lora.disable_adapter():
            out_base = gen(lora, msgs)
        out_lora = gen(lora, msgs)
        lines += [
            f"## {s['meta']['intent']}",
            f"**用户:** {msgs[1]['content']}",
            "",
            f"- **期望输出:** `{gt}`",
            f"- **base:** `{out_base}`",
            f"- **lora:** `{out_lora}`",
            "",
        ]
        print(f"[{s['meta']['intent']}] user: {msgs[1]['content']}")
        print(f"  base: {out_base[:80]}")
        print(f"  lora: {out_lora[:80]}")

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    out = results_dir / "compare_examples.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[done] -> {out}")


if __name__ == "__main__":
    main()
