#!/usr/bin/env node
/** Adapter: Code de Vie shot list → comfyui-studio `npm run orchestrate -- --manifest`.
 *  One ai-lb lease per shot/batch (renew does NOT extend TTL). Resumable: skips shots with gen/<id>/<id>.mp4.
 *  --dry-run writes the orchestrator manifest and prints the command without calling the GPU.
 */
import { readFile, writeFile, mkdir, access } from "node:fs/promises";
import { spawn } from "node:child_process";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parse as parseYaml } from "yaml";

const __dirname = dirname(fileURLToPath(import.meta.url));
const PIPE = resolve(__dirname, "..");
const DATA = process.env.CDV_DATA || join(process.env.HOME, "video-data/code-de-vie");
const STUDIO = process.env.COMFYUI_STUDIO || join(process.env.HOME, "github/comfyui-studio");
const dry = process.argv.includes("--dry-run");
const force = process.argv.includes("--force");
const only = (() => { const i = process.argv.indexOf("--only"); return i >= 0 ? process.argv[i + 1].split(",") : null; })();

const raw = await readFile(join(PIPE, "shots.yaml"), "utf8");
const doc = parseYaml(raw);
const shots = doc.shots.filter((s) => !only || only.includes(s.id) || only.includes(String(s.shot)));

const orchShots = [];
for (const s of shots) {
  const outDir = join(DATA, "gen", s.id);
  const outMp4 = join(outDir, `${s.id}.mp4`);
  const outMp4Flat = join(DATA, "gen", `${s.id}.mp4`);
  let exists = false;
  try { await access(outMp4); exists = true; } catch {}
  try { await access(outMp4Flat); exists = true; } catch {}
  if (exists && !force) {
    console.log(`skip ${s.id} (cached ${outMp4})`);
    continue;
  }
  const world = s.world === "nurse" ? "nurse" : "coder";
  const localStill = join(DATA, "shots", `shot_${String(s.shot).padStart(2,"0")}`, `shot_${String(s.shot).padStart(2,"0")}_still.png`);
  const genStill = join(outDir, `${s.id}_still.png`);
  let character = localStill;
  try { await access(genStill); character = genStill; } catch {}
  try { await access(localStill); character = localStill; } catch {}
  if (s.still_ref && (s.still_ref.startsWith("outputs/") || s.still_ref.startsWith("/"))) {
    const cand = join(STUDIO, s.still_ref.replace(/^\.\//, ""));
    try { await access(cand); character = cand; } catch {}
  }
  orchShots.push({
    id: s.id,
    start: s.start,
    end: s.end,
    world,
    seed: (s.still_seeds && s.still_seeds[0]) || 42,
    prompt: s.video_prompt || "subtle natural motion, photorealistic",
    workflow: s.workflow,
    audio_slice: !!s.audio_slice,
    lease_minutes: 30,
    image: character,
    character,
  });
}

const manifest = {
  song: {
    audio: join(DATA, doc.song?.audio || "audio/code-de-vie_v4_5_reverbtail.wav"),
    fps: doc.fps || 24,
    beats: join(PIPE, "config/beats_v3.json"),
  },
  out: join(DATA, "gen"),
  lease_minutes: 30,
  worlds: {
    coder: { workflow: "workflows/wan22-i2v-a14b-gguf-4step.json", post: ["conform"] },
    nurse: { workflow: "workflows/wan22-i2v-a14b-gguf-4step.json", post: ["conform"] },
  },
  shots: orchShots,
};

await mkdir(join(DATA, "manifests"), { recursive: true });
const manPath = join(DATA, "manifests/orchestrate.yaml");
const { stringify } = await import("yaml");
await writeFile(manPath, stringify(manifest));
console.log(`wrote ${manPath} (${orchShots.length} jobs)`);

if (!orchShots.length) {
  console.log("nothing to generate");
  process.exit(0);
}

const args = ["run", "orchestrate", "--", "--manifest", manPath, "--continue-on-error"];
if (force) args.push("--force");
console.log(`$ npm ${args.join(" ")}   (cwd=${STUDIO})`);
if (dry) {
  console.log("dry-run: not calling ComfyUI / ai-lb");
  process.exit(0);
}

const child = spawn("npm", args, { cwd: STUDIO, stdio: "inherit", shell: false });
child.on("exit", (code) => process.exit(code ?? 1));
