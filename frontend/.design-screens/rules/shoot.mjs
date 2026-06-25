// Headless-Chrome screenshot harness for the standalone rules page.
//
// Serves the built dist/ directory over http (so the page's fetch("cards.json")
// works — we drop a cards.json copy into dist for the shoot), then captures the
// rules page in English + Chinese, desktop + mobile, full-page.
//
// Usage: node shoot.mjs <label>      e.g. node shoot.mjs before / node shoot.mjs after
import http from "http";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import puppeteer from "/Users/tgoh/.npm/_npx/55158e48eb5c59f7/node_modules/puppeteer-core/lib/esm/puppeteer/puppeteer-core.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const FRONTEND = path.resolve(__dirname, "..", "..");
const DIST = path.join(FRONTEND, "dist");
const OUT = __dirname;
const CHROME =
  "/Users/tgoh/.cache/puppeteer/chrome/mac_arm-147.0.7727.57/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing";

const label = process.argv[2] || "shot";

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".png": "image/png",
  ".mp3": "audio/mpeg",
  ".map": "application/json",
};

const server = http.createServer((req, res) => {
  let urlPath = decodeURIComponent(req.url.split("?")[0]);
  if (urlPath === "/") urlPath = "/index.html";
  const filePath = path.join(DIST, urlPath);
  if (!filePath.startsWith(DIST) || !fs.existsSync(filePath)) {
    res.writeHead(404);
    res.end("not found: " + urlPath);
    return;
  }
  const ext = path.extname(filePath);
  res.writeHead(200, { "Content-Type": MIME[ext] || "application/octet-stream" });
  fs.createReadStream(filePath).pipe(res);
});

const configs = [
  { name: "en-desktop", lang: "en", width: 1280, height: 900, dpr: 1 },
  { name: "en-mobile", lang: "en", width: 390, height: 844, dpr: 2 },
  { name: "zh-desktop", lang: "zh", width: 1280, height: 900, dpr: 1 },
  { name: "zh-mobile", lang: "zh", width: 390, height: 844, dpr: 2 },
];

async function main() {
  // Make sure cards.json is available next to rules.html for the runtime fetch.
  const cardsSrc = path.join(FRONTEND, "src", "generated", "cards.json");
  fs.copyFileSync(cardsSrc, path.join(DIST, "cards.json"));

  await new Promise((r) => server.listen(0, r));
  const port = server.address().port;
  const base = `http://127.0.0.1:${port}`;

  const browser = await puppeteer.launch({
    executablePath: CHROME,
    headless: "shell",
    args: ["--no-sandbox", "--font-render-hinting=none"],
  });

  for (const c of configs) {
    const page = await browser.newPage();
    await page.setViewport({
      width: c.width,
      height: c.height,
      deviceScaleFactor: c.dpr,
    });
    const url = `${base}/rules.html?lang=${c.lang}`;
    await page.goto(url, { waitUntil: "networkidle0", timeout: 30000 });
    // Give async card rendering + fonts a beat to settle.
    await new Promise((r) => setTimeout(r, 700));
    const file = path.join(OUT, `rules-${label}-${c.name}.png`);
    await page.screenshot({ path: file, fullPage: true });
    console.log("wrote", file);
    await page.close();
  }

  await browser.close();
  server.close();
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
