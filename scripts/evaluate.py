# -*- coding: utf-8 -*-
"""
评测脚本: 在 data/eval.jsonl (句式与训练集不相交) 上对比评测。

用法:
    python scripts/evaluate.py --model base
    python scripts/evaluate.py --model lora --path output/qwen-nlu-lora

指标(三层):
1. format_valid_rate : 输出能解析为合法 JSON 且含 intent/slots 字段
2. intent_accuracy   : intent 标签精确匹配
3. slot_p / slot_r / slot_f1 : 槽位键值对精确匹配 (微平均)

可选 --regression: 跑几个通用通识问题, 人工检查是否灾难性遗忘。
"""
import argparse
import json
import re
from pathlib import Path

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
REGRESSION_QUESTIONS = [
    "用一句话介绍你自己。",
    "中国最长的河流是哪条？",
    "把下面的话翻译成英文：今天天气真好。",
    "9.11 和 9.8 哪个大？",
    "写一个 Python 函数计算两数之和。",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["base", "lora"], required=True)
    p.add_argument("--path", default="output/qwen-nlu-lora", help="LoRA adapter 目录")
    p.add_argument("--eval_file", default="data/eval.jsonl")
    p.add_argument("--max_new_tokens", type=int, default=96)
    p.add_argument("--regression", action="store_true", help="附带跑通用能力回归")
    return p.parse_args()


def load_model_and_tokenizer(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    use_bf16 = torch.cuda.is_bf16_supported()
    torch_dtype = torch.bfloat16 if use_bf16 else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch_dtype, device_map="auto"
    )
    name = "base"
    if args.model == "lora":
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.path)
        name = "lora"
    model.eval()
    return model, tokenizer, name


def extract_json(text: str) -> dict | None:
    """容错解析: 截取输出中第一个平衡的 {...} 再 json.loads。"""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        # schema 校验: intent 是字符串且 slots 是 dict, 否则视为格式不合法
        if (
            isinstance(obj, dict)
            and isinstance(obj.get("intent"), str)
            and isinstance(obj.get("slots"), dict)
        ):
            return obj
    except json.JSONDecodeError:
        pass
    return None


def generate(model, tokenizer, messages, max_new_tokens):
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,          # 评测要确定性输出
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def evaluate(args):
    model, tokenizer, name = load_model_and_tokenizer(args)
    samples = [json.loads(l) for l in open(args.eval_file, encoding="utf-8")]

    n_valid = n_intent_ok = 0
    slot_tp = slot_fp = slot_fn = 0
    bad_cases = []

    for i, s in enumerate(samples):
        msgs = s["messages"][:2]          # system + user
        gt = s["meta"]
        pred_text = generate(model, tokenizer, msgs, args.max_new_tokens)
        pred = extract_json(pred_text)

        if pred is not None:
            n_valid += 1
            if pred["intent"] == gt["intent"]:
                n_intent_ok += 1
            # 槽位: 键值对精确匹配
            gt_pairs = set(gt["slots"].items())
            pred_pairs = set(
                (k, v) for k, v in (pred["slots"] or {}).items()
                if isinstance(v, (str, int, float))
            )
            slot_tp += len(gt_pairs & pred_pairs)
            slot_fp += len(pred_pairs - gt_pairs)
            slot_fn += len(gt_pairs - pred_pairs)
        else:
            bad_cases.append({"user": msgs[1]["content"], "raw_output": pred_text[:200]})

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(samples)} ...")

    n = len(samples)
    precision = slot_tp / (slot_tp + slot_fp) if slot_tp + slot_fp else 0.0
    recall = slot_tp / (slot_tp + slot_fn) if slot_tp + slot_fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    report = {
        "model": name,
        "n": n,
        "format_valid_rate": round(n_valid / n, 4),
        "intent_accuracy": round(n_intent_ok / n, 4),
        "slot_precision": round(precision, 4),
        "slot_recall": round(recall, 4),
        "slot_f1": round(f1, 4),
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if bad_cases:
        print(f"\n格式解析失败 {len(bad_cases)} 条, 示例:")
        for b in bad_cases[:3]:
            print(f"  user: {b['user']}")
            print(f"  raw : {b['raw_output']}")

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    out_file = results_dir / f"eval_{name}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[done] report -> {out_file}")

    # ---- 可选: 通用能力回归 ----
    if args.regression:
        print("\n===== 通用能力回归(人工检查输出是否正常) =====")
        reg_path = results_dir / f"regression_{name}.md"
        with open(reg_path, "w", encoding="utf-8") as f:
            f.write(f"# 通用能力回归 - {name}\n\n")
            for q in REGRESSION_QUESTIONS:
                ans = generate(
                    model, tokenizer,
                    [{"role": "user", "content": q}],
                    max_new_tokens=128,
                )
                print(f"\nQ: {q}\nA: {ans[:200]}")
                f.write(f"**Q:** {q}\n\n**A:** {ans}\n\n")
        print(f"\n[done] regression -> {reg_path}")


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
