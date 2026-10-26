// Render scene.html frame by frame in headless Chromium and pipe the frames into ffmpeg.
//
//   node render.js                           -> starry_night_video.mp4, then starry_night.mp4 with music.wav
//   node render.js --mux                     -> only redo the music mux
//   node render.js --stills 300,1200,2500    -> still_<frame>.png only, no video
const { chromium } = require('playwright');
const { spawn } = require('child_process');
const http = require('http');
const fs = require('fs');
const path = require('path');

const args = process.argv.slice(2);
const opt = name => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : null; };
const silent = path.join(__dirname, 'starry_night_video.mp4');
const out = opt('--out') || path.join(__dirname, 'starry_night.mp4');
const stills = opt('--stills') ? opt('--stills').split(',').map(Number) : null;
const music = path.join(__dirname, 'music.wav');

const TYPES = { '.html': 'text/html', '.json': 'application/json', '.bin': 'application/octet-stream' };
function serve() {
  const server = http.createServer((req, res) => {
    const file = path.join(__dirname, decodeURIComponent(req.url.split('?')[0]));
    if (!file.startsWith(__dirname) || !fs.existsSync(file)) { res.writeHead(404); return res.end(); }
    res.writeHead(200, { 'Content-Type': TYPES[path.extname(file)] || 'application/octet-stream' });
    fs.createReadStream(file).pipe(res);
  });
  return new Promise(r => server.listen(0, '127.0.0.1', () => r(server)));
}

function run(cmd, argv) {
  return new Promise((ok, fail) => spawn(cmd, argv, { stdio: 'inherit' }).on('close', c => c ? fail(new Error(cmd + ' ' + c)) : ok()));
}

async function mux() {
  if (!fs.existsSync(music)) return console.log('no music.wav, video is silent:', silent);
  await run('ffmpeg', ['-y', '-loglevel', 'error', '-i', silent, '-i', music, '-c:v', 'copy',
    '-c:a', 'aac', '-b:a', '256k', '-shortest', '-movflags', '+faststart', out]);
  console.log('wrote', out);
}

(async () => {
  if (args.includes('--mux')) return mux();
  const server = await serve();
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1080, height: 1920 } });
  page.on('pageerror', e => { console.error(e); process.exit(1); });
  await page.goto(`http://127.0.0.1:${server.address().port}/scene.html`);
  const { total, fps } = await page.evaluate(() => init());
  console.log(`${total} frames, ${(total / fps).toFixed(1)}s`);

  let ff = null;
  if (!stills) {
    ff = spawn('ffmpeg', ['-y', '-loglevel', 'error', '-f', 'image2pipe', '-framerate', String(fps), '-c:v', 'mjpeg', '-i', '-',
      '-c:v', 'libx264', '-preset', 'slow', '-crf', '17', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', silent],
      { stdio: ['pipe', 'inherit', 'inherit'] });
  }

  const last = stills ? Math.max(...stills) : total - 1;
  const t0 = Date.now();
  for (let f = 0; f <= last; f++) {
    const want = !stills || stills.includes(f);
    const b64 = await page.evaluate(([f, want, type]) => {
      renderFrame(f, want);
      return want ? document.getElementById('c').toDataURL(type, 0.95).split(',')[1] : null;
    }, [f, want, stills ? 'image/png' : 'image/jpeg']);
    if (b64 && stills) fs.writeFileSync(path.join(__dirname, `still_${f}.png`), Buffer.from(b64, 'base64'));
    else if (b64 && !ff.stdin.write(Buffer.from(b64, 'base64'))) await new Promise(r => ff.stdin.once('drain', r));
    if (!stills && f % 300 === 0) console.log(`frame ${f}/${last}  ${((Date.now() - t0) / 1000).toFixed(0)}s`);
  }
  await browser.close();
  server.close();
  if (ff) {
    ff.stdin.end();
    await new Promise(r => ff.on('close', r));
    console.log('wrote', silent);
    await mux();
  }
})();
