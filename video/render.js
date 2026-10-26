// Render scene.html frame by frame in headless Chromium and pipe the frames into ffmpeg.
//
//   node render.js                          -> fourier_mona_lisa.mp4
//   node render.js --stills 300,1200,2500   -> PNG stills of those frames, no video
const { chromium } = require('playwright');
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

const args = process.argv.slice(2);
const opt = name => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : null; };
const out = opt('--out') || path.join(__dirname, 'fourier_mona_lisa.mp4');
const stills = opt('--stills') ? opt('--stills').split(',').map(Number) : null;

(async () => {
  const data = JSON.parse(fs.readFileSync(path.join(__dirname, 'terms.json'), 'utf8'));
  const font = fs.readFileSync(path.join(__dirname, 'assets/title.woff2')).toString('base64');

  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1080, height: 1920 } });
  page.on('pageerror', e => { console.error(e); process.exit(1); });
  await page.goto('file://' + path.join(__dirname, 'scene.html'));
  const { total, rounds } = await page.evaluate(([d, f]) => init(d, f), [data, font]);
  console.log(`${total} frames (${(total / 60).toFixed(1)}s), rounds: ${rounds.join(' ')}`);

  const last = stills ? Math.max(...stills) : total - 1;
  let ff = null;
  if (!stills) {
    ff = spawn('ffmpeg', ['-y', '-loglevel', 'error', '-f', 'image2pipe', '-framerate', '60', '-c:v', 'mjpeg', '-i', '-',
      '-c:v', 'libx264', '-preset', 'slow', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', out],
      { stdio: ['pipe', 'inherit', 'inherit'] });
  }

  const t0 = Date.now();
  for (let f = 0; f <= last; f++) {
    const want = !stills || stills.includes(f);
    const b64 = await page.evaluate(([f, want, type]) => {
      renderFrame(f);
      return want ? document.getElementById('c').toDataURL(type, 0.95).split(',')[1] : null;
    }, [f, want, stills ? 'image/png' : 'image/jpeg']);
    if (b64 && stills) fs.writeFileSync(path.join(__dirname, `still_${f}.png`), Buffer.from(b64, 'base64'));
    else if (b64 && !ff.stdin.write(Buffer.from(b64, 'base64'))) await new Promise(r => ff.stdin.once('drain', r));
    if (f % 300 === 0) console.log(`frame ${f}/${last}  ${((Date.now() - t0) / 1000).toFixed(0)}s`);
  }
  await browser.close();
  if (ff) {
    ff.stdin.end();
    await new Promise(r => ff.on('close', r));
    console.log('wrote', out);
  }
})();
