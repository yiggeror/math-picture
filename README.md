# math-picture
none.

## 傅里叶旋转圆绘画 (`index.html`)

用浏览器直接打开 `index.html`。把一条闭合曲线采样成复数点列，做离散傅里叶变换，
每一项是一个按自身频率旋转的圆，首尾相接，末端的笔尖就会画出原图。

- 图案：猫、心形、五角星、任意文字、手绘一笔
- 「逐级增加圆」：1、2、4、8…个圆依次绘制，看图案从模糊的椭圆变成清晰的轮廓

## 傅里叶画蒙娜丽莎 · 竖屏视频 (`video/`)

1 → 2 → 4 → … → 16384 个旋转圆，每一轮圆的数量翻倍，从一团乱线逐渐收敛成《蒙娜丽莎》。
输出 1080×1920、60 fps 的 mp4。

```bash
pip install numpy pillow scikit-image   # 一次即可
cd video
python3 make_path.py assets/mona_lisa.jpg terms.json   # 图片 → 等高线 → 一笔画 → FFT
node render.js                                         # 逐帧渲染 → fourier_mona_lisa.mp4
node render.js --stills 600,1800                       # 只导出几帧 PNG 预览
```

- `make_path.py`：把图片亮度的等高线连成一条闭合曲线，等弧长采样 32768 点做 FFT，保留最大的 16384 个圆。轮廓之间的跳线记为“提笔”，圆够多以后画得很淡。
- `scene.html`：Canvas 渲染器，每个圆是一个旋转的复数相量，笔尖是它们的和。
- `render.js`：用无头 Chromium 逐帧截图，经 ffmpeg 编码成 H.264。
- 换图：把 `assets/` 里的图片换掉，调 `make_path.py` 顶部的 `CROP`、`LEVELS`。
- 原画《蒙娜丽莎》为公有领域作品（图片来自 Wikimedia Commons）。
