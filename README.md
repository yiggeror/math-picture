# math-picture
none.

## 傅里叶旋转圆绘画 (`index.html`)

用浏览器直接打开 `index.html`。把一条闭合曲线采样成复数点列，做离散傅里叶变换，
每一项是一个按自身频率旋转的圆，首尾相接，末端的笔尖就会画出原图。

- 图案：猫、心形、五角星、任意文字、手绘一笔
- 「逐级增加圆」：1、2、4、8…个圆依次绘制，看图案从模糊的椭圆变成清晰的轮廓

## 星月夜 · 傅里叶竖屏视频 (`video/`)

13 万个首尾相接的旋转圆，一笔画出梵高《星月夜》。1080×1920、60 fps、48 秒，配乐同步。

- 开场镜头贴在漩涡中心的笔尖上，看不出在画什么；
- 画面向外一圈圈生长，镜头随之拉远，笔速按指数加快；
- 最后一笔闭合：圆炸开，一道冲击波把彩虹色线条变成原画的颜色；
- 揭晓后，光波随节拍从漩涡中心扩散。

```bash
pip install numpy scipy pillow       # 一次即可
cd video
python3 make_path.py    # 原画 → 笔触流线 → 一笔画 → FFT，输出 data/
python3 music.py        # 合成配乐 music.wav
node render.js          # 逐帧渲染并混入音乐 → starry_night.mp4
node render.js --stills 600,1800     # 只导出几帧 PNG 预览
node render.js --mux                 # 改了配乐后只重新合成音轨
```

- `make_path.py`：按原画笔触方向场追踪 7000 多条流线（每条像一笔油画笔触），从漩涡中心向外逐环串成一条闭合曲线，采样 262144 点做 FFT，保留最大的 131072 个圆。笔触之间的跳线是“提笔”，不画出来。
- `scene.html`：Canvas 渲染器，镜头、配色、冲击波、光晕都在这里。
- `music.py`：用 numpy 合成的原创配乐，时间点读 `config.json`，和画面同步。
- `config.json`：时长、高潮时间、节拍。
- 原画《星月夜》为公有领域作品（图片来自 Wikimedia Commons）。
