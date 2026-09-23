# Base vs LoRA-SFT 对比样例

- 基座: `Qwen/Qwen2.5-0.5B-Instruct`
- 样例来源: `data/eval.jsonl` (句式与训练集不相交)

## weather.query
**用户:** 帮我查查天津的天气预报, 周三的

- **期望输出:** `{"intent": "weather.query", "slots": {"city": "天津", "date": "周三"}}`
- **base:** `{"intent": "查询天气", "slots": {"city": "天津", "date": "周三"} }`
- **lora:** `{"intent": "weather.query", "slots": {"city": "天津", "date": "周三"}}`

## weather.query
**用户:** 那个帮我查查合肥的天气预报, 周四的呀

- **期望输出:** `{"intent": "weather.query", "slots": {"city": "合肥", "date": "周四"}}`
- **base:** `{"intent": "查询天气", "slots": {"city": ["合肥"], "date": ["周四"]}}`
- **lora:** `{"intent": "weather.query", "slots": {"city": "合肥", "date": "周四"}}`

## weather.query
**用户:** 帮我天气预报, 苏州, 周六可以吗

- **期望输出:** `{"intent": "weather.query", "slots": {"city": "苏州", "date": "周六"}}`
- **base:** `{"intent": "查询天气", "slots": {"city": "苏州", "time": "周六"}}`
- **lora:** `{"intent": "weather.query", "slots": {"city": "苏州", "date": "周六"}}`

## weather.query
**用户:** 那个天气预报, 厦门, 周一行不行

- **期望输出:** `{"intent": "weather.query", "slots": {"city": "厦门", "date": "周一"}}`
- **base:** `{"intent": "查询天气", "slots": {"city": ["厦门"], "time": []}}`
- **lora:** `{"intent": "weather.query", "slots": {"city": "厦门", "date": "周一"}}`

## weather.query
**用户:** 麻烦帮我天气预报, 南京, 下周五行不行

- **期望输出:** `{"intent": "weather.query", "slots": {"city": "南京", "date": "下周五"}}`
- **base:** `{"intent": "查询天气", "slots": {"city": "南京", "time": "下周五"}}`
- **lora:** `{"intent": "weather.query", "slots": {"city": "南京", "date": "下周五"}}`

## weather.query
**用户:** 看看大后天天津天气咋样行不行

- **期望输出:** `{"intent": "weather.query", "slots": {"city": "天津", "date": "大后天"}}`
- **base:** `{"intent": "询问天气信息", "slots": {"date": ["大后天"], "city": ["天津"]}}`
- **lora:** `{"intent": "weather.query", "slots": {"city": "天津", "date": "大后天"}}`

## weather.query
**用户:** 帮我查查杭州的天气预报, 周三的呀

- **期望输出:** `{"intent": "weather.query", "slots": {"city": "杭州", "date": "周三"}}`
- **base:** `{"intent": "查询天气", "slots": {"city": "杭州", "date": "周三"}}`
- **lora:** `{"intent": "weather.query", "slots": {"city": "杭州", "date": "周三"}}`

## weather.query
**用户:** 嘿想知道北京下周三冷不冷呀

- **期望输出:** `{"intent": "weather.query", "slots": {"city": "北京", "date": "下周三"}}`
- **base:** `{"intent": "询问天气", "slots": {"location": ["北京"], "time": ["下周三"]}}`
- **lora:** `{"intent": "weather.query", "slots": {"city": "北京", "date": "下周三"}}`

## weather.query
**用户:** 那个天气预报, 青岛, 周四

- **期望输出:** `{"intent": "weather.query", "slots": {"city": "青岛", "date": "周四"}}`
- **base:** `{"intent": "查询天气信息", "slots": {"city": ["青岛"], "time": ["周四"]}}`
- **lora:** `{"intent": "weather.query", "slots": {"city": "青岛", "date": "周四"}}`

## weather.query
**用户:** 帮我天气预报, 西安, 周五呗

- **期望输出:** `{"intent": "weather.query", "slots": {"city": "西安", "date": "周五"}}`
- **base:** `{"intent": "查询天气", "slots": {"city": "西安", "date": "周五"}}`
- **lora:** `{"intent": "weather.query", "slots": {"city": "西安", "date": "周五"}}`

## call.make
**用户:** 打给老板好嘛

- **期望输出:** `{"intent": "call.make", "slots": {"contact": "老板"}}`
- **base:** `{"intent": "请求", "slots": {"request_type": ["给老板打电话"], "reason": ["询问如何表达感谢或礼貌地向老板打招呼"}}`
- **lora:** `{"intent": "call.make", "slots": {"contact": "老板"}}`

## call.make
**用户:** 打给陈医生行不行

- **期望输出:** `{"intent": "call.make", "slots": {"contact": "陈医生"}}`
- **base:** `{"intent": "请求", "slots": {"doctor": ["陈医生"]}}`
- **lora:** `{"intent": "call.make", "slots": {"contact": "陈医生"}}`

## call.make
**用户:** 帮我拨号给老王谢谢

- **期望输出:** `{"intent": "call.make", "slots": {"contact": "老王"}}`
- **base:** `{"intent": "请求", "slots": {"oldwang": ["老王"]}}`
- **lora:** `{"intent": "call.make", "slots": {"contact": "老王"}}`

## call.make
**用户:** 麻烦帮我打给弟弟行不行

- **期望输出:** `{"intent": "call.make", "slots": {"contact": "弟弟"}}`
- **base:** `{"intent": "请求帮助", "slots": {"receiver": ["弟弟"]}}`
- **lora:** `{"intent": "call.make", "slots": {"contact": "弟弟"}}`

## call.make
**用户:** 麻烦帮我帮我和王经理建立通话

- **期望输出:** `{"intent": "call.make", "slots": {"contact": "王经理"}}`
- **base:** `{"intent": "help", "slots": {"name": ["帮助对象"], "manager": ["王经理"]}}`
- **lora:** `{"intent": "call.make", "slots": {"contact": "王经理"}}`

## call.make
**用户:** 麻烦帮我跟同学通话

- **期望输出:** `{"intent": "call.make", "slots": {"contact": "同学"}}`
- **base:** `{"intent": "请求帮助", "slots": {"同学": ["同学"], "问题": ["通话"}}`
- **lora:** `{"intent": "call.make", "slots": {"contact": "同学"}}`

## call.make
**用户:** 跟陈医生通话好嘛

- **期望输出:** `{"intent": "call.make", "slots": {"contact": "陈医生"}}`
- **base:** `{"intent": "询问意愿", "slots": {"doctor": ["陈医生"]}}`
- **lora:** `{"intent": "call.make", "slots": {"contact": "陈医生"}}`

## call.make
**用户:** 那个跟小李通话

- **期望输出:** `{"intent": "call.make", "slots": {"contact": "小李"}}`
- **base:** `{"intent": "联系人", "slots": {"name": ["小李"], "type": ["电话"]}}`
- **lora:** `{"intent": "call.make", "slots": {"contact": "小李"}}`

## call.make
**用户:** 跟王经理通话

- **期望输出:** `{"intent": "call.make", "slots": {"contact": "王经理"}}`
- **base:** `{"intent": "与人沟通", "slots": {"person": ["王经理"], "topic": []}}`
- **lora:** `{"intent": "call.make", "slots": {"contact": "王经理"}}`

## call.make
**用户:** 拨号给弟弟好嘛

- **期望输出:** `{"intent": "call.make", "slots": {"contact": "弟弟"}}`
- **base:** `{"intent": "请求", "slots": {"phone_number": ["电话号码"]}}`
- **lora:** `{"intent": "call.make", "slots": {"contact": "弟弟"}}`

## reminder.set
**用户:** 帮我到下午五点半叫我吃药行不行

- **期望输出:** `{"intent": "reminder.set", "slots": {"content": "吃药", "time": "17:30"}}`
- **base:** `{"intent": "请求帮助", "slots": {"time": ["下午五点半"]}}`
- **lora:** `{"intent": "reminder.set", "slots": {"content": "吃药", "time": "17:30"}}`

## reminder.set
**用户:** 嘿提醒我后天倒垃圾谢谢

- **期望输出:** `{"intent": "reminder.set", "slots": {"content": "倒垃圾", "date": "后天"}}`
- **base:** `{"intent": "请求", "slots": {"date": ["后天"], "reason": ["倒垃圾"]}}`
- **lora:** `{"intent": "reminder.set", "slots": {"content": "倒垃圾", "date": "后天"}}`

## reminder.set
**用户:** 到晚上十点叫我复盘周报呗

- **期望输出:** `{"intent": "reminder.set", "slots": {"content": "复盘周报", "time": "22:00"}}`
- **base:** `{"intent": "提醒", "slots": {"时间": ["晚上十点"]}}`
- **lora:** `{"intent": "reminder.set", "slots": {"content": "复盘周报", "time": "22:00"}}`

## reminder.set
**用户:** 提醒我下周三给客户回电话

- **期望输出:** `{"intent": "reminder.set", "slots": {"content": "给客户回电话", "date": "下周三"}}`
- **base:** `{"intent": "通知", "slots": {"date": ["下周三"], "time": ["给客户回电话"]}}`
- **lora:** `{"intent": "reminder.set", "slots": {"content": "给客户回电话", "date": "下周三"}}`

## reminder.set
**用户:** 提醒我明天给妈妈打电话行不行

- **期望输出:** `{"intent": "reminder.set", "slots": {"content": "给妈妈打电话", "date": "明天"}}`
- **base:** `{"intent": "请求", "slots": {"时间": ["明天"], "对象": ["妈妈"]}}`
- **lora:** `{"intent": "reminder.set", "slots": {"content": "给妈妈打电话", "date": "明天"}}`

## reminder.set
**用户:** 麻烦提醒我周日吃药好嘛

- **期望输出:** `{"intent": "reminder.set", "slots": {"content": "吃药", "date": "周日"}}`
- **base:** `{"intent": "提醒", "slots": {"time": ["周日"], "medicine": []}}`
- **lora:** `{"intent": "reminder.set", "slots": {"content": "吃药", "date": "周日"}}`

## reminder.set
**用户:** 嘿设置提醒, 早上八点该交房租了

- **期望输出:** `{"intent": "reminder.set", "slots": {"content": "交房租", "time": "08:00"}}`
- **base:** `{"intent": "设置提醒", "slots": {"time": ["早上八点"], "reason": []}}`
- **lora:** `{"intent": "reminder.set", "slots": {"content": "交房租", "time": "08:00"}}`

## reminder.set
**用户:** 设置提醒, 晚上七点半该喝水了

- **期望输出:** `{"intent": "reminder.set", "slots": {"content": "喝水", "time": "19:30"}}`
- **base:** `{"intent": "设置提醒", "slots": {"time": ["晚上七点半"], "action": ["喝水"]}}`
- **lora:** `{"intent": "reminder.set", "slots": {"content": "喝水了", "time": "19:30"}}`

## reminder.set
**用户:** 提醒我这周末订机票

- **期望输出:** `{"intent": "reminder.set", "slots": {"content": "订机票", "date": "这周末"}}`
- **base:** `{"intent": "通知", "slots": {"date": ["日期"], "time": ["时间"], "destination": ["目的地"], "flight": ["航班"]}}`
- **lora:** `{"intent": "reminder.set", "slots": {"content": "订机票", "date": "这周末"}}`

## reminder.set
**用户:** 嘿设置提醒, 早上八点该复核合同了好嘛

- **期望输出:** `{"intent": "reminder.set", "slots": {"content": "复核合同", "time": "08:00"}}`
- **base:** `{"intent": "设置提醒", "slots": {"time": ["早上八点"], "task": ["复核合同"]}}`
- **lora:** `{"intent": "reminder.set", "slots": {"content": "复核合同", "time": "08:00"}}`

## music.play
**用户:** 麻烦给我来点周杰伦的音乐

- **期望输出:** `{"intent": "music.play", "slots": {"artist": "周杰伦"}}`
- **base:** `{"intent": "播放音乐", "slots": {"artist": "周杰伦"}}`
- **lora:** `{"intent": "music.play", "slots": {"artist": "周杰伦"}}`

## music.play
**用户:** 帮我给我来点孙燕姿的音乐

- **期望输出:** `{"intent": "music.play", "slots": {"artist": "孙燕姿"}}`
- **base:** `{"intent": "播放音乐", "slots": {"artist": ["孙燕姿"], "genre": ["音乐"]}}`
- **lora:** `{"intent": "music.play", "slots": {"artist": "孙燕姿"}}`

## music.play
**用户:** 能不能放赵雷的起风了

- **期望输出:** `{"intent": "music.play", "slots": {"song": "起风了", "artist": "赵雷"}}`
- **base:** `{"intent": "播放音乐", "slots": {"artist": ["赵雷"], "song": ["起风了"]}}`
- **lora:** `{"intent": "music.play", "slots": {"song": "起风了", "artist": "赵雷"}}`

## music.play
**用户:** 帮我给我来点陈奕迅的音乐好嘛

- **期望输出:** `{"intent": "music.play", "slots": {"artist": "陈奕迅"}}`
- **base:** `{"intent": "推荐音乐", "slots": {"artist": ["陈奕迅"], "genre": []}}`
- **lora:** `{"intent": "music.play", "slots": {"artist": "陈奕迅"}}`

## music.play
**用户:** 音乐, 放起风了好嘛

- **期望输出:** `{"intent": "music.play", "slots": {"song": "起风了"}}`
- **base:** `{"intent": "播放音乐", "slots": {"music": ["音乐"]}}`
- **lora:** `{"intent": "music.play", "slots": {"song": "起风了"}}`

## music.play
**用户:** 帮我给我来点邓紫棋的音乐行不行

- **期望输出:** `{"intent": "music.play", "slots": {"artist": "邓紫棋"}}`
- **base:** `{"intent": "播放音乐", "slots": {"artist": ["邓紫棋"], "genre": []}}`
- **lora:** `{"intent": "music.play", "slots": {"artist": "邓紫棋"}}`

## music.play
**用户:** 帮我音乐, 放如愿

- **期望输出:** `{"intent": "music.play", "slots": {"song": "如愿"}}`
- **base:** `{"intent": "播放音乐", "slots": {"artist": "", "genre": ""}}`
- **lora:** `{"intent": "music.play", "slots": {"song": "如愿"}}`

## music.play
**用户:** 给我来点赵雷的音乐呀

- **期望输出:** `{"intent": "music.play", "slots": {"artist": "赵雷"}}`
- **base:** `{"intent": "请求音乐", "slots": {"artist": ["赵雷"], "genre": ["音乐"]}}`
- **lora:** `{"intent": "music.play", "slots": {"artist": "赵雷"}}`

## music.play
**用户:** 帮我音乐, 放贝加尔湖畔

- **期望输出:** `{"intent": "music.play", "slots": {"song": "贝加尔湖畔"}}`
- **base:** `{"intent": "播放音乐", "slots": {"genre": ["贝加尔湖畔"], "artist": []}}`
- **lora:** `{"intent": "music.play", "slots": {"song": "贝加尔湖畔"}}`

## music.play
**用户:** 帮我唱一下小幸运行不行

- **期望输出:** `{"intent": "music.play", "slots": {"song": "小幸运"}}`
- **base:** `{"intent": "唱歌", "slots": {"歌曲": ["小幸运行不行"], "歌手": []}}`
- **lora:** `{"intent": "music.play", "slots": {"song": "小幸运"}}`

## device.control
**用户:** 帮我去关空调

- **期望输出:** `{"intent": "device.control", "slots": {"device": "ac", "action": "off"}}`
- **base:** `{"intent": "设置设备", "slots": {"device": ["空调"], "action": ["关"]}}`
- **lora:** `{"intent": "device.control", "slots": {"device": "ac", "action": "off"}}`

## device.control
**用户:** 麻烦能不能把加湿器关一下

- **期望输出:** `{"intent": "device.control", "slots": {"device": "humidifier", "action": "off"}}`
- **base:** `{"intent": "请求", "slots": {"设备类型": ["加湿器"], "状态": ["关闭"]}}`
- **lora:** `{"intent": "device.control", "slots": {"device": "humidifier", "action": "off"}}`

## device.control
**用户:** 嘿能不能把热水器开一下可以吗

- **期望输出:** `{"intent": "device.control", "slots": {"device": "water_heater", "action": "on"}}`
- **base:** `{"intent": "指令请求", "slots": {"热水器": ["打开"]}}`
- **lora:** `{"intent": "device.control", "slots": {"device": "water_heater", "action": "on"}}`

## device.control
**用户:** 电视, 开行不行

- **期望输出:** `{"intent": "device.control", "slots": {"device": "tv", "action": "on"}}`
- **base:** `{"intent": "指令", "slots": {"command": "开行"}}`
- **lora:** `{"intent": "device.control", "slots": {"device": "tv", "action": "on"}}`

## device.control
**用户:** 帮我加湿器, 开

- **期望输出:** `{"intent": "device.control", "slots": {"device": "humidifier", "action": "on"}}`
- **base:** `{"intent": "add humidifier", "slots": {"name": "", "type": ""}}`
- **lora:** `{"intent": "device.control", "slots": {"device": "humidifier", "action": "on"}}`

## device.control
**用户:** 麻烦帮我去关客厅的灯可以吗

- **期望输出:** `{"intent": "device.control", "slots": {"device": "light.livingroom", "action": "off"}}`
- **base:** `{"intent": "请求帮助", "slots": {"room": ["客厅"], "action": ["关"]}}`
- **lora:** `{"intent": "device.control", "slots": {"device": "light.livingroom", "action": "off"}}`

## device.control
**用户:** 去关窗帘谢谢

- **期望输出:** `{"intent": "device.control", "slots": {"device": "curtain", "action": "off"}}`
- **base:** `{"intent": "请求", "slots": {"窗帘": ["关"]}}`
- **lora:** `{"intent": "device.control", "slots": {"device": "curtain", "action": "off"}}`

## device.control
**用户:** 帮我能不能把电视关一下呗

- **期望输出:** `{"intent": "device.control", "slots": {"device": "tv", "action": "off"}}`
- **base:** `{"intent": "请求帮助", "slots": {"设备类型": ["电视"], "功能需求": ["关闭"]}}`
- **lora:** `{"intent": "device.control", "slots": {"device": "tv", "action": "off"}}`

## device.control
**用户:** 客厅的空调, 关

- **期望输出:** `{"intent": "device.control", "slots": {"device": "ac.livingroom", "action": "off"}}`
- **base:** `{"intent": "设置", "slots": {"room": ["客厅"], "device": ["空调"]}}`
- **lora:** `{"intent": "device.control", "slots": {"device": "ac.livingroom", "action": "off"}}`

## device.control
**用户:** 麻烦空调, 关行不行

- **期望输出:** `{"intent": "device.control", "slots": {"device": "ac", "action": "off"}}`
- **base:** `{"intent": "请求", "slots": {"空调": [], "行": []}}`
- **lora:** `{"intent": "device.control", "slots": {"device": "ac", "action": "off"}}`

## schedule.query
**用户:** 麻烦查行程, 下周日好嘛

- **期望输出:** `{"intent": "schedule.query", "slots": {"date": "下周日"}}`
- **base:** `{"intent": "查询行程", "slots": {"date": ["下周日"], "reason": []}}`
- **lora:** `{"intent": "schedule.query", "slots": {"date": "下周日"}}`

## schedule.query
**用户:** 麻烦帮我周四的会议给我看看谢谢

- **期望输出:** `{"intent": "schedule.query", "slots": {"date": "周四"}}`
- **base:** `{"intent": "询问", "slots": {"time": ["Thursday"]}}`
- **lora:** `{"intent": "schedule.query", "slots": {"date": "周四"}}`

## schedule.query
**用户:** 麻烦帮我看看周一我都安排了什么谢谢

- **期望输出:** `{"intent": "schedule.query", "slots": {"date": "周一"}}`
- **base:** `{"intent": "需求获取", "slots": {"时间": ["周一"], "任务类型": ["安排"]}}`
- **lora:** `{"intent": "schedule.query", "slots": {"date": "周一"}}`

## schedule.query
**用户:** 查安排, 后天谢谢

- **期望输出:** `{"intent": "schedule.query", "slots": {"date": "后天"}}`
- **base:** `{"intent": "查询", "slots": {"time": ["后天"]}}`
- **lora:** `{"intent": "schedule.query", "slots": {"date": "后天"}}`

## schedule.query
**用户:** 下周三的会议给我看看行不行

- **期望输出:** `{"intent": "schedule.query", "slots": {"date": "下周三"}}`
- **base:** `{"intent": "询问", "slots": {"date": ["下周三"], "reason": []}}`
- **lora:** `{"intent": "schedule.query", "slots": {"date": "下周三"}}`

## schedule.query
**用户:** 麻烦帮我查安排, 周六好嘛

- **期望输出:** `{"intent": "schedule.query", "slots": {"date": "周六"}}`
- **base:** `{"intent": "查询", "slots": {"date": ["周六"]}}`
- **lora:** `{"intent": "schedule.query", "slots": {"date": "周六"}}`

## schedule.query
**用户:** 帮我看看周日我都安排了什么

- **期望输出:** `{"intent": "schedule.query", "slots": {"date": "周日"}}`
- **base:** `{"intent": "询问信息", "slots": {"weekend": ["安排"]}}`
- **lora:** `{"intent": "schedule.query", "slots": {"date": "周日"}}`

## schedule.query
**用户:** 查日程, 下周一好嘛

- **期望输出:** `{"intent": "schedule.query", "slots": {"date": "下周一"}}`
- **base:** `{"intent": "查询", "slots": {"day": "下周一"}}`
- **lora:** `{"intent": "schedule.query", "slots": {"date": "下周一"}}`

## schedule.query
**用户:** 麻烦周三的行程给我看看好嘛

- **期望输出:** `{"intent": "schedule.query", "slots": {"date": "周三"}}`
- **base:** `{"intent": "查询行程", "slots": {"date": ["周三"], "time": ["行程"]}}`
- **lora:** `{"intent": "schedule.query", "slots": {"date": "周三"}}`

## schedule.query
**用户:** 今天的行程给我看看

- **期望输出:** `{"intent": "schedule.query", "slots": {"date": "今天"}}`
- **base:** `{"intent": "查询行程", "slots": {}}`
- **lora:** `{"intent": "schedule.query", "slots": {"date": "今天"}}`

## volume.set
**用户:** 麻烦帮我声音大些

- **期望输出:** `{"intent": "volume.set", "slots": {"direction": "up"}}`
- **base:** `{"intent": "指令", "slots": {"volume": ["大"]}}`
- **lora:** `{"intent": "volume.set", "slots": {"direction": "up"}}`

## volume.set
**用户:** 音量设置成40呗

- **期望输出:** `{"intent": "volume.set", "slots": {"value": "40"}}`
- **base:** `{"intent": "设置", "slots": {"volume": 40}}`
- **lora:** `{"intent": "volume.set", "slots": {"value": "40"}}`

## volume.set
**用户:** 帮我帮我把声音调到50行不行

- **期望输出:** `{"intent": "volume.set", "slots": {"value": "50"}}`
- **base:** `{"intent":"语音控制","slots":["音量"]}`
- **lora:** `{"intent": "volume.set", "slots": {"direction": "up", "value": "50"}}`

## volume.set
**用户:** 那个音量设置成70谢谢

- **期望输出:** `{"intent": "volume.set", "slots": {"value": "70"}}`
- **base:** `{"intent": "请求", "slots": {"volume": 1}}`
- **lora:** `{"intent": "volume.set", "slots": {"value": "70"}}`

## volume.set
**用户:** 调大一点声音呗

- **期望输出:** `{"intent": "volume.set", "slots": {"direction": "up"}}`
- **base:** `{"intent": "调整音量", "slots": {}}`
- **lora:** `{"intent": "volume.set", "slots": {"direction": "up"}}`

## volume.set
**用户:** 帮我帮我把声音调到80

- **期望输出:** `{"intent": "volume.set", "slots": {"value": "80"}}`
- **base:** `{"intent": "指令执行", "slots": {"volume": 80}}`
- **lora:** `{"intent": "volume.set", "slots": {"value": "80"}}`

## volume.set
**用户:** 那个声音小些好嘛

- **期望输出:** `{"intent": "volume.set", "slots": {"direction": "down"}}`
- **base:** `{"intent": "需求", "slots": {"sound": ["声音大小"]}}`
- **lora:** `{"intent": "volume.set", "slots": {"direction": "down"}}`

## volume.set
**用户:** 麻烦音量设置成30可以吗

- **期望输出:** `{"intent": "volume.set", "slots": {"value": "30"}}`
- **base:** `{"intent": "设置", "slots": {"volume": 30}}`
- **lora:** `{"intent": "volume.set", "slots": {"value": "30"}}`

## volume.set
**用户:** 那个音量设置成20

- **期望输出:** `{"intent": "volume.set", "slots": {"value": "20"}}`
- **base:** `{"intent": "设置音量", "slots": {"volume": 20}}`
- **lora:** `{"intent": "volume.set", "slots": {"value": "20"}}`

## volume.set
**用户:** 帮我把声音调到30行不行

- **期望输出:** `{"intent": "volume.set", "slots": {"value": "30"}}`
- **base:** `{"intent":"语音控制","slots":["音量"]}`
- **lora:** `{"intent": "volume.set", "slots": {"direction": "up", "value": "30"}}`

## alarm.set
**用户:** 嘿我下周日早上八点半要起, 弄个闹钟

- **期望输出:** `{"intent": "alarm.set", "slots": {"time": "08:30", "date": "下周日"}}`
- **base:** `{"intent": "提醒", "slots": {"time": ["8:30"]}}`
- **lora:** `{"intent": "alarm.set", "slots": {"time": "08:30", "date": "下周日"}}`

## alarm.set
**用户:** 安排一个傍晚六点的闹钟提醒我好嘛

- **期望输出:** `{"intent": "alarm.set", "slots": {"time": "18:00"}}`
- **base:** `{"intent": "安排闹钟", "slots": {"time": ["12:00"], "reason": ["安排"]}}`
- **lora:** `{"intent": "alarm.set", "slots": {"time": "18:00"}}`

## alarm.set
**用户:** 帮我整一个七点一刻的闹钟谢谢

- **期望输出:** `{"intent": "alarm.set", "slots": {"time": "07:15"}}`
- **base:** `{"intent": "设置时间", "slots": {"hour": 7, "minute": 1}}`
- **lora:** `{"intent": "alarm.set", "slots": {"time": "07:15"}}`

## alarm.set
**用户:** 帮我设个后天傍晚六点的闹铃

- **期望输出:** `{"intent": "alarm.set", "slots": {"time": "18:00", "date": "后天"}}`
- **base:** `{"intent": "设置闹钟", "slots": {"time": "后天傍晚六点"}}`
- **lora:** `{"intent": "alarm.set", "slots": {"time": "18:00", "date": "后天"}}`

## alarm.set
**用户:** 麻烦整一个下午四点半的闹钟好嘛

- **期望输出:** `{"intent": "alarm.set", "slots": {"time": "16:30"}}`
- **base:** `{"intent": "设置时间", "slots": {"time": ["4:30"]}}`
- **lora:** `{"intent": "alarm.set", "slots": {"time": "16:30"}}`

## alarm.set
**用户:** 嘿闹钟调到晚上十一点可以吗

- **期望输出:** `{"intent": "alarm.set", "slots": {"time": "23:00"}}`
- **base:** `{"intent": "设置时间", "slots": {"time": ["10:00"]}}`
- **lora:** `{"intent": "alarm.set", "slots": {"time": "23:00"}}`

## alarm.set
**用户:** 我周四中午十二点要起, 弄个闹钟行不行

- **期望输出:** `{"intent": "alarm.set", "slots": {"time": "12:00", "date": "周四"}}`
- **base:** `{"intent": "提醒", "slots": {"time": ["12:00"]}}`
- **lora:** `{"intent": "alarm.set", "slots": {"time": "12:00", "date": "周四"}}`

## alarm.set
**用户:** 麻烦帮我设个周三十一点的闹铃好嘛

- **期望输出:** `{"intent": "alarm.set", "slots": {"time": "11:00", "date": "周三"}}`
- **base:** `{"intent": "设置闹钟", "slots": {"time": ["13:00"]}}`
- **lora:** `{"intent": "alarm.set", "slots": {"time": "13:00"}}`

## alarm.set
**用户:** 麻烦帮我我大后天早上八点要起, 弄个闹钟好嘛

- **期望输出:** `{"intent": "alarm.set", "slots": {"time": "08:00", "date": "大后天"}}`
- **base:** `{"intent": "提醒事项", "slots": {"时间": ["8:00"], "地点": ["起床"]}}`
- **lora:** `{"intent": "alarm.set", "slots": {"time": "08:00", "date": "大后天"}}`

## alarm.set
**用户:** 闹钟调到三点一刻呀

- **期望输出:** `{"intent": "alarm.set", "slots": {"time": "15:15"}}`
- **base:** `{"intent": "设置时间", "slots": {"time": ["3:00"]}}`
- **lora:** `{"intent": "alarm.set", "slots": {"time": "15:15"}}`

---

## 结论

base 模型能模仿 JSON 的"形"(格式合法率 97%), 但不知道任务协议, 典型错误模式:

1. **意图自编标签**: `查询天气`/`请求`/`指令执行`(而非 weather.query/call.make/volume.set)
2. **槽位键名自编**: `location`/`时间`/`volume`(而非 city/date/value)
3. **时间不规范化**: `后天傍晚六点`/`3:00`(而非 18:00/15:15, "三点一刻"映射错误)
4. **值类型不稳**: 槽位值包 list(`["8:30"]`), 甚至 slots 整个是 list

LoRA-SFT(1.75% 可训练参数)把任务协议烧进权重后, 以上问题全部消失。
量化指标与之一致: 意图准确率 0% -> 98.5%, 槽位 F1 13.9% -> 96.4%。

