# Base vs LoRA-SFT 对比样例

- 基座: `Qwen/Qwen2.5-0.5B-Instruct`
- 样例来源: `data/eval.jsonl` (句式与训练集不相交)
- 解码: greedy (do_sample=False)
- 完整 80 条见云端运行输出, 此处为代表性选录

## weather.query

**用户:** 帮我查查天津的天气预报, 周三的

- **期望输出:** `{"intent": "weather.query", "slots": {"city": "天津", "date": "周三"}}`
- **base:** `{"intent": "查询天气", "slots": {"city": "天津", "date": "周三"} }` — 意图自编中文标签
- **lora:** `{"intent": "weather.query", "slots": {"city": "天津", "date": "周三"}}`

**用户:** 嘿想知道北京下周三冷不冷呀

- **期望输出:** `{"intent": "weather.query", "slots": {"city": "北京", "date": "下周三"}}`
- **base:** `{"intent": "询问天气", "slots": {"location": ["北京"], "time": ["下周三"]}}` — 键名自编 + 值包 list
- **lora:** `{"intent": "weather.query", "slots": {"city": "北京", "date": "下周三"}}`

## call.make

**用户:** 打给老板好嘛

- **期望输出:** `{"intent": "call.make", "slots": {"contact": "老板"}}`
- **base:** `{"intent": "请求", "slots": {"request_type": ["给老板打电话"], "reason": ["询问如何表达感谢或礼貌地向老板打招呼"...` — 完全跑偏
- **lora:** `{"intent": "call.make", "slots": {"contact": "老板"}}`

**用户:** 拨号给弟弟好嘛

- **期望输出:** `{"intent": "call.make", "slots": {"contact": "弟弟"}}`
- **base:** `{"intent": "请求", "slots": {"phone_number": ["电话号码"]}}` — 猜不到联系人
- **lora:** `{"intent": "call.make", "slots": {"contact": "弟弟"}}`

## reminder.set

**用户:** 到晚上十点叫我复盘周报呗

- **期望输出:** `{"intent": "reminder.set", "slots": {"content": "复盘周报", "time": "22:00"}}`
- **base:** `{"intent": "提醒", "slots": {"时间": ["晚上十点"]}}` — 时间未规范化为 22:00, 内容槽丢失
- **lora:** `{"intent": "reminder.set", "slots": {"content": "复盘周报", "time": "22:00"}}`

**用户:** 提醒我下周三给客户回电话

- **期望输出:** `{"intent": "reminder.set", "slots": {"content": "给客户回电话", "date": "下周三"}}`
- **base:** `{"intent": "通知", "slots": {"date": ["下周三"], "time": ["给客户回电话"]}}` — 槽位语义错位
- **lora:** `{"intent": "reminder.set", "slots": {"content": "给客户回电话", "date": "下周三"}}`

## music.play

**用户:** 帮我音乐, 放如愿

- **期望输出:** `{"intent": "music.play", "slots": {"song": "如愿"}}`
- **base:** `{"intent": "播放音乐", "slots": {"artist": "", "genre": ""}}` — 歌名丢失
- **lora:** `{"intent": "music.play", "slots": {"song": "如愿"}}`

## volume.set

**用户:** 音量设置成40呗

- **期望输出:** `{"intent": "volume.set", "slots": {"value": "40"}}`
- **base:** `{"intent": "音量控制", "slots": ["音量"]}` — slots 为 list, schema 全错
- **lora:** `{"intent": "volume.set", "slots": {"value": "40"}}`

## alarm.set

**用户:** 帮我整一个七点一刻的闹钟谢谢

- **期望输出:** `{"intent": "alarm.set", "slots": {"time": "07:15"}}`
- **base:** `{"intent": "设置闹钟", "slots": {"时间": ["七点一刻"]}}` — 口语时间未映射为 07:15
- **lora:** `{"intent": "alarm.set", "slots": {"time": "07:15"}}`

---

## 结论

base 模型能模仿 JSON 的"形"(格式合法率 97%), 但**不知道任务协议**: 意图标签自编(`查询天气`/`请求`)、
槽位键名自编(`location`/`时间`)、值不做规范化(`晚上十点` 而非 `22:00`)、类型不稳(值包 list)。

LoRA-SFT 用 1.75% 可训练参数把"任务协议"烧进权重: 意图命中私有命名空间、槽位键值对齐 schema、
口语时间规范化为标准时间。评测指标(intent 0% -> 98.5%, slot F1 13.9% -> 96.4%)与之完全一致。
