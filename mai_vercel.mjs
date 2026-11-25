import { existsSync } from 'node:fs';
import { readFile, rename, writeFile } from 'node:fs/promises';
import { resolve, join } from 'node:path';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';

const folder = process.argv[2];
const plan = (await readFile(join(folder, 'plan.jsonl'), 'utf8')).trim().split('\n').map(line => JSON.parse(line));
const pending = plan.filter(row => !existsSync(join(folder, row.audio_path)));
if (pending.length === 0) {
  console.log('Complete:', plan.length, 'WAVs');
  process.exit(0);
}
if (!process.env.AI_GATEWAY_API_KEY) throw new Error('Export AI_GATEWAY_API_KEY before running the Vercel batch.');
const sdk = createRequire(resolve(process.env.MAI_SDK_DIR || 'outputs/mai-sdk', 'package.json'));
const { generateSpeech, gateway } = await import(pathToFileURL(sdk.resolve('ai')).href);
let done = plan.length - pending.length;
for (const row of pending) {
  const result = await generateSpeech({
    model: gateway.speechModel(row.model), voice: row.voice, text: row.question, outputFormat: 'wav',
    providerOptions: { azure: { style: row.style } },
  });
  if (result.warnings?.length) throw new Error('Speech provider reported warnings: ' + JSON.stringify(result.warnings));
  const wav = Buffer.from(result.audio.uint8Array);
  if (wav.toString('ascii', 0, 4) !== 'RIFF' || wav.toString('ascii', 8, 12) !== 'WAVE' || wav.length <= 44) {
    throw new Error('Expected a nonempty WAV for ' + row.question_id);
  }
  const path = join(folder, row.audio_path);
  await writeFile(path + '.tmp', wav);
  await rename(path + '.tmp', path);
  console.log(`[${++done}/${plan.length}] ${row.question_id} ${row.style}`);
}
