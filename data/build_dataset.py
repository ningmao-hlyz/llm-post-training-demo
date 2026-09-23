# -*- coding: utf-8 -*-
"""
程序化构建语音助手 NLU(意图识别+槽位抽取) SFT 数据集。

任务: 用户对语音助手说一句话, 模型输出结构化 JSON:
      {"intent": "<意图标签>", "slots": {...}}

设计要点:
1. 训练集 / 评测集的【句式模板完全不相交】(held-out templates),
   评测考察泛化能力, 防止模型死记训练句式。
2. 时间口语表达统一规范化为 HH:MM(如 "早上八点" -> 08:00),
   模型需要学到语义映射而非逐字复制。
3. 槽位存在性由话语内容决定(话语中明确提及才输出),
   评测按槽位键值对精确匹配计算 P/R/F1。

输出:
    data/train.jsonl  # {"messages": [...], "meta": {ground truth}}
    data/eval.jsonl
"""
import json
import random
from pathlib import Path

random.seed(42)

# ---------------------------------------------------------------------------
# 词表 (无歧义: 同一口语表达只对应一个标准值)
# ---------------------------------------------------------------------------

# (口语表达, 规范化HH:MM)
TIMES = [
    ("早上六点", "06:00"), ("六点半", "06:30"), ("早上七点", "07:00"), ("七点一刻", "07:15"),
    ("早上七点半", "07:30"), ("早上八点", "08:00"), ("早上八点半", "08:30"),
    ("九点", "09:00"), ("上午九点半", "09:30"), ("十点", "10:00"), ("上午十点半", "10:30"),
    ("十一点", "11:00"), ("中午十二点", "12:00"), ("十二点半", "12:30"),
    ("下午一点", "13:00"), ("下午两点", "14:00"), ("下午两点半", "14:30"),
    ("下午三点", "15:00"), ("三点一刻", "15:15"), ("下午四点", "16:00"),
    ("下午四点半", "16:30"), ("下午五点半", "17:30"), ("傍晚六点", "18:00"),
    ("晚上七点", "19:00"), ("晚上七点半", "19:30"), ("晚上八点", "20:00"),
    ("晚上八点半", "20:30"), ("晚上九点", "21:00"), ("晚上十点", "22:00"),
    ("晚上十点半", "22:30"), ("晚上十一点", "23:00"),
]
DATES = ["今天", "明天", "后天", "周一", "周二", "周三", "周四", "周五", "周六", "周日",
         "这周末", "下周一", "下周三", "下周五", "下周日", "大后天"]
CITIES = ["北京", "上海", "广州", "深圳", "杭州", "成都", "重庆", "武汉", "西安", "南京",
          "苏州", "长沙", "青岛", "厦门", "昆明", "大连", "天津", "合肥", "郑州", "佛山"]
SONGS = ["晴天", "七里香", "稻香", "夜曲", "青花瓷", "小幸运", "光年之外", "起风了",
         "平凡之路", "漠河舞厅", "孤勇者", "晚风", "成都", "贝加尔湖畔", "后来",
         "突然好想你", "倔强", "温柔", "山海", "如愿"]
ARTISTS = ["周杰伦", "林俊杰", "陈奕迅", "邓紫棋", "五月天", "薛之谦", "毛不易", "李荣浩",
           "Taylor Swift", "王菲", "刘德华", "孙燕姿", "朴树", "许嵩", "赵雷"]
CONTACTS = ["妈妈", "爸爸", "老王", "张伟", "李娜", "老板", "女朋友", "弟弟", "姐姐",
            "王经理", "小李", "陈医生", "刘老师", "大姨", "同学", "同事"]
DEVICES = [("客厅的灯", "light.livingroom"), ("卧室的灯", "light.bedroom"),
           ("空调", "ac"), ("电视", "tv"), ("加湿器", "humidifier"),
           ("热水器", "water_heater"), ("窗帘", "curtain"), ("扫地机器人", "robot_vacuum"),
           ("客厅的空调", "ac.livingroom"), ("台灯", "desk_lamp")]
REMINDER_CONTENTS = ["喝水", "吃药", "开会", "取快递", "还信用卡", "给客户回电话",
                     "健身", "倒垃圾", "浇花", "交房租", "订机票", "复盘周报",
                     "给妈妈打电话", "取干洗的衣服", "复核合同"]
SCHEDULE_KEYWORDS = ["日程", "安排", "会议", "行程"]

# 口语前后缀, 增加表达多样性
PREFIXES = ["", "麻烦", "帮我", "帮我", "麻烦帮我", "嘿", "那个", "", "", ""]
SUFFIXES = ["", "", "", "谢谢", "呗", "好嘛", "行不行", "可以吗", "呀"]

# ---------------------------------------------------------------------------
# 意图定义: 模板分 train / eval 两个不相交池
# ---------------------------------------------------------------------------

INTENTS = {
    "alarm.set": {
        "train": [
            "帮我定一个{time_raw}的闹钟",
            "定个{date}{time_raw}的闹钟",
            "设个闹钟{date}{time_raw}",
            "闹钟定在{time_raw}",
            "我{date}{time_raw}要早起, 定个闹钟",
            "提醒我{date}{time_raw}起床, 定闹钟",
            "给我上个{time_raw}的闹钟",
            "设置{time_raw}闹钟",
            "要赶车, 闹钟设{time_raw}",
        ],
        "eval": [
            "帮我设个{date}{time_raw}的闹铃",
            "闹钟调到{time_raw}",
            "我{date}{time_raw}要起, 弄个闹钟",
            "整一个{time_raw}的闹钟",
            "安排一个{time_raw}的闹钟提醒我",
        ],
    },
    "weather.query": {
        "train": [
            "{city}明天天气怎么样",
            "查一下{city}{date}的天气",
            "明天{city}会下雨吗",
            "帮我看看{date}{city}的天气",
            "{city}天气如何",
            "跟我说说{city}{date}的天气预报",
            "查天气, {city}, {date}",
        ],
        "eval": [
            "看看{date}{city}天气咋样",
            "帮我查查{city}的天气预报, {date}的",
            "想知道{city}{date}冷不冷",
            "天气预报, {city}, {date}",
        ],
    },
    "music.play": {
        "train": [
            "播放{artist}的{song}",
            "我想听{song}",
            "来一首{artist}的歌",
            "放一下{artist}的{song}",
            "单曲循环{song}",
            "随机播放{artist}的歌",
            "我想听{artist}的{song}",
            "放首{song}听听",
        ],
        "eval": [
            "给我来点{artist}的音乐",
            "唱一下{song}",
            "能不能放{artist}的{song}",
            "音乐, 放{song}",
        ],
    },
    "volume.set": {
        "train": [
            "声音调{direction_zh}一点",
            "音量调到{value}",
            "调{direction_zh}音量",
            "把音量{direction_zh}调到{value}",
            "音量{direction_zh}",
            "太吵了, 声音调{direction_zh}",
            "听不清, 音量调{direction_zh}",
        ],
        "eval": [
            "声音{direction_zh}些",
            "帮我把声音调到{value}",
            "调{direction_zh}一点声音",
            "音量设置成{value}",
        ],
    },
    "reminder.set": {
        "train": [
            "提醒我{time_raw}{content}",
            "{date}提醒我{content}",
            "帮我设个提醒, {time_raw}要{content}",
            "别忘了{date}{time_raw}{content}",
            "记一下, {time_raw}提醒我{content}",
            "定个提醒{date}{time_raw}{content}",
        ],
        "eval": [
            "提醒我{date}{content}",
            "设置提醒, {time_raw}该{content}了",
            "到{time_raw}叫我{content}",
        ],
    },
    "schedule.query": {
        "train": [
            "我{date}有什么{kw}",
            "看看{date}的{kw}",
            "帮我查下{date}的{kw}",
            "{date}我忙不忙, 看看{kw}",
            "查一下我的{kw}, {date}的",
            "我{date}的{kw}有哪些",
            "看看我有什么{kw}",
        ],
        "eval": [
            "{date}的{kw}给我看看",
            "帮我看看{date}我都安排了什么",
            "查{kw}, {date}",
        ],
    },
    "call.make": {
        "train": [
            "打电话给{contact}",
            "给{contact}打个电话",
            "呼叫{contact}",
            "帮我拨通{contact}的电话",
            "我要给{contact}打电话",
            "电话打给{contact}",
            "联系一下{contact}, 打电话",
        ],
        "eval": [
            "拨号给{contact}",
            "跟{contact}通话",
            "打给{contact}",
            "帮我和{contact}建立通话",
        ],
    },
    "device.control": {
        "train": [
            "把{device_raw}{action_zh}了",
            "{action_zh}{device_raw}",
            "帮我{action_zh}{device_raw}",
            "{device_raw}给我{action_zh}",
            "我回来了, {action_zh}{device_raw}",
            "麻烦把{device_raw}{action_zh}一下",
        ],
        "eval": [
            "{device_raw}, {action_zh}",
            "去{action_zh}{device_raw}",
            "能不能把{device_raw}{action_zh}一下",
        ],
    },
}

SYSTEM_PROMPT = (
    "你是一个语音助手的语义理解模块。根据用户的话语, 输出意图和槽位的 JSON, "
    '格式为 {"intent": "意图标签", "slots": {...}}, 不要输出任何其他内容。'
)


def render(template: str) -> tuple[str, dict]:
    """填充模板占位符, 返回 (用户话语, 词表采样值字典)。"""
    time_raw, time_std = random.choice(TIMES)
    device_raw, device_id = random.choice(DEVICES)
    direction_zh, direction = random.choice([("大", "up"), ("小", "down")])
    action_zh, action = random.choice([("开", "on"), ("关", "off")])
    v = {
        "time_std": time_std, "time_raw": time_raw, "date": random.choice(DATES),
        "city": random.choice(CITIES), "song": random.choice(SONGS),
        "artist": random.choice(ARTISTS), "contact": random.choice(CONTACTS),
        "device_id": device_id, "device_raw": device_raw,
        "content": random.choice(REMINDER_CONTENTS),
        "kw": random.choice(SCHEDULE_KEYWORDS),
        "direction": direction, "value": str(random.choice([10, 20, 30, 40, 50, 60, 70, 80])),
        "action": action, "direction_zh": direction_zh, "action_zh": action_zh,
    }
    text = template.format(**v)
    text = random.choice(PREFIXES) + text + random.choice(SUFFIXES)
    return text, v


def build_sample(intent: str, template: str) -> dict:
    """槽位存在性由模板占位符决定: 话语中明确提及的才进入 ground truth。"""
    text, v = render(template)
    slots = {}

    if intent == "alarm.set":
        slots["time"] = v["time_std"]
        if "{date}" in template:
            slots["date"] = v["date"]
    elif intent == "weather.query":
        slots["city"] = v["city"]
        if "{date}" in template:
            slots["date"] = v["date"]
    elif intent == "music.play":
        if "{song}" in template:
            slots["song"] = v["song"]
        if "{artist}" in template:
            slots["artist"] = v["artist"]
    elif intent == "volume.set":
        slots["direction"] = v["direction"]
        if "{value}" in template:
            slots["value"] = v["value"]
    elif intent == "reminder.set":
        slots["content"] = v["content"]
        if "{time_raw}" in template:
            slots["time"] = v["time_std"]
            if "{date}" in template:
                slots["date"] = v["date"]
        elif "{date}" in template:
            slots["date"] = v["date"]
    elif intent == "schedule.query":
        if "{date}" in template:
            slots["date"] = v["date"]
    elif intent == "call.make":
        slots["contact"] = v["contact"]
    elif intent == "device.control":
        slots["device"] = v["device_id"]
        slots["action"] = v["action"]

    answer = json.dumps({"intent": intent, "slots": slots}, ensure_ascii=False)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
            {"role": "assistant", "content": answer},
        ],
        "meta": {"intent": intent, "slots": slots},
    }


def main():
    out_dir = Path(__file__).parent
    n_train, n_eval = 1200, 200
    intents = list(INTENTS.keys())

    def sample_many(templates, n, seen):
        pool = []
        guard = 0
        while len(pool) < n and guard < n * 100:
            guard += 1
            intent = intents[len(pool) % len(intents)]  # 先按意图轮转保证均衡
            s = build_sample(intent, random.choice(templates[intent]))
            key = s["messages"][1]["content"]
            if key in seen:
                continue
            seen.add(key)
            pool.append(s)
        return pool

    train = sample_many({k: v["train"] for k, v in INTENTS.items()}, n_train, set())
    evalset = sample_many({k: v["eval"] for k, v in INTENTS.items()}, n_eval, set())
    random.shuffle(train)
    random.shuffle(evalset)

    with open(out_dir / "train.jsonl", "w", encoding="utf-8") as f:
        for s in train:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    with open(out_dir / "eval.jsonl", "w", encoding="utf-8") as f:
        for s in evalset:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    from collections import Counter
    c = Counter(s["meta"]["intent"] for s in train)
    print(f"train: {len(train)} 条 -> {out_dir / 'train.jsonl'}")
    print(f"eval : {len(evalset)} 条 -> {out_dir / 'eval.jsonl'}")
    print("意图分布(train):", dict(sorted(c.items())))
    print("\n样例:")
    for s in train[:3]:
        print(" user:", s["messages"][1]["content"])
        print(" gt  :", s["messages"][2]["content"])


if __name__ == "__main__":
    main()
