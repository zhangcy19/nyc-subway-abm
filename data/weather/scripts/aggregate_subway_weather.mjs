// Bundled Node + @oai/artifact-tool. Run from a directory with the runtime's
// node_modules junction. One station per workbook keeps memory bounded; the
// requested CSV is larger than Excel's single-sheet row limit.
import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';
import assert from 'node:assert/strict';

const require = createRequire(path.join(process.cwd(), '_runtime_loader.cjs'));
const { Workbook } = await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const dataRoot = path.dirname(root);
const output = path.join(root, 'subway_weather_hourly.csv');
const columns = ['station_complex_id', 'station_name', 'borough', 'latitude', 'longitude',
  'timestamp', 'rain_mm', 'temp_c', 'wind_ms', 'weather_available', 'weather_station'];
const fields = ['rain_mm', 'temp_c', 'wind_ms'];
const sha256 = value => createHash('sha256').update(value).digest('hex');
const inventory = JSON.parse(await fs.readFile(path.join(root, 'station_inventory.json'), 'utf8'));
const candidates = inventory.stations.filter(s => s.exported).sort((a, b) => a.station.localeCompare(b.station));
const pending = inventory.stations.filter(s => !s.exported).map(s => s.station);
const expectedTimes = Array.from({length: 2952}, (_, i) => new Date(Date.UTC(2023, 6, 1) + i * 3600000)
  .toISOString().slice(0, 19).replace('T', ' ')); // Local clock labels, not a UTC interpretation.

async function readCSV(filename, sheetName, width) {
  const text = await fs.readFile(filename, 'utf8');
  const wb = await Workbook.fromCSV(text, {sheetName});
  const rows = wb.worksheets.getItem(sheetName).getUsedRange().values;
  assert(rows.every(row => row.length === width), `${filename}: unexpected column count`);
  return {text, rows};
}
const stationSource = await readCSV(path.join(dataRoot, 'stations.csv'), 'Stations', 5);
assert.deepEqual(stationSource.rows[0], columns.slice(0, 5));
const stations = stationSource.rows.slice(1).map(row => [...row.slice(0, 3), Number(row[3]), Number(row[4])]);
assert.equal(stations.length, 424);
assert.equal(new Set(stations.map(row => row[0])).size, 424);
for (const row of stations) {
  assert(row.slice(0, 3).every(x => typeof x === 'string' && x.length));
  assert(Number.isFinite(row[3]) && Math.abs(row[3]) <= 90);
  assert(Number.isFinite(row[4]) && Math.abs(row[4]) <= 180);
}
assert(candidates.length >= 3);
const sourceHashes = [];
const values = [];
for (const s of candidates) {
  const source = await readCSV(path.join(root, s.filename), s.station, 6);
  assert.deepEqual(source.rows[0], columns.slice(5));
  assert.deepEqual(source.rows.slice(1).map(row => row[0]), expectedTimes);
  assert(source.rows.slice(1).every(row => row[5] === s.station));
  if (s.sha256) assert.equal(sha256(source.text), s.sha256, `${s.station}: source changed since validation`);
  values.push(source.rows.slice(1).map(row => row.slice(1, 4).map(x => {
    if (x === '' || x === null) return null;
    const value = Number(x);
    assert(Number.isFinite(value));
    return value;
  })));
  assert(source.rows.slice(1).every((row, h) => (row[4] === 'True') === values.at(-1)[h].every(x => x !== null)));
  sourceHashes.push({station: s.station, filename: s.filename, sha256: sha256(source.text),
    latitude: s.latitude, longitude: s.longitude});
}

export function distanceKm(lat1, lon1, lat2, lon2) {
  const radians = Math.PI / 180;
  const a = Math.sin((lat2-lat1)*radians/2)**2 + Math.cos(lat1*radians)*Math.cos(lat2*radians)*Math.sin((lon2-lon1)*radians/2)**2;
  return 2 * 6371.0088 * Math.asin(Math.sqrt(Math.max(0, Math.min(1, a))));
}

// Returns indices in the distance ranking, actual normalized weights, and method.
export function selectAndWeight(ranked, observed) {
  let selected = ranked.slice(0, 3).filter(item => observed[item.index] !== null);
  let method = selected.length === 3 ? 'top3_full' : 'top3_partial';
  if (!selected.length) {
    const fallback = ranked.slice(3).find(item => observed[item.index] !== null);
    if (!fallback) return {value: null, selected: [], weights: [], method: 'missing'};
    selected = [fallback];
    method = 'fallback';
  }
  const colocated = selected.filter(item => item.distance_km <= 1e-9);
  if (colocated.length) selected = colocated;
  const unscaled = selected.map(item => colocated.length ? 1 : 1 / item.distance_km**2);
  const total = unscaled.reduce((a, b) => a+b, 0);
  const weights = unscaled.map(w => w / total);
  return {value: selected.reduce((sum, item, i) => sum + weights[i]*observed[item.index], 0), selected, weights, method};
}

// Boundary checks exercise missingness decisions, not just normal weighted means.
const testRank = [1,2,3,4,5,6].map((d,index) => ({index,distance_km:d}));
assert.equal(selectAndWeight(testRank, [null,0,null,999,888,777]).value, 0);
assert.equal(selectAndWeight(testRank, [null,null,null,null,7,9]).value, 7);
assert.equal(selectAndWeight(testRank, [null,null,null,null,null,null]).value, null);
assert.equal(selectAndWeight([{index:0,distance_km:0}, ...testRank.slice(1)], [4,99,99,0,0,0]).value, 4);
assert(Math.abs(selectAndWeight(testRank, [0,10,null,999,999,999]).value - 2) < 1e-12);
assert(Math.abs(distanceKm(0,0,0,1) - 111.19508) < 0.0001);

function csvCell(value) {
  if (value === null || value === undefined) return '';
  const s = typeof value === 'boolean' ? (value ? 'True' : 'False') : String(value);
  return /[",\r\n]/.test(s) ? `"${s.replaceAll('"', '""')}"` : s;
}

const stats = Object.fromEntries(fields.map(key => [key, {top3_full:0, top3_partial:0, fallback:0, missing:0}]));
const sourceUse = Object.fromEntries(fields.map(key => [key, Object.fromEntries(candidates.map(s => [s.station,0]))]));
const catalog = [];
let rowsWritten = 0, completeRows = 0;
let wrotePreview = false;
const file = await fs.open(output + '.partial', 'w');
await file.write(columns.map(csvCell).join(',') + '\n');
try {
  for (let si=0; si<stations.length; si++) {
    const station = stations[si];
    const ranked = candidates.map((g,index) => ({index, code:g.station,
      distance_km:distanceKm(station[3],station[4],g.latitude,g.longitude)}))
      .sort((a,b) => a.distance_km-b.distance_km || a.code.localeCompare(b.code));
    const nominal = selectAndWeight(ranked, candidates.map(() => 1));
    catalog.push({station_complex_id:station[0], station_name:station[1],
      latitude:station[3], longitude:station[4],
      nearest_three:ranked.slice(0,3).map(g => g.code),
      base_weights:nominal.selected.map((g,i) => ({station:g.code,weight:nominal.weights[i]})),
      all_candidates:ranked.map(({code,distance_km}) => ({station:code,distance_km}))});
    const rows = [];
    for (let h=0; h<expectedTimes.length; h++) {
      const result = fields.map((field,v) => {
        const selected = selectAndWeight(ranked, values.map(g => g[h][v]));
        stats[field][selected.method]++;
        selected.selected.forEach(item => sourceUse[field][item.code]++);
        return selected;
      });
      const numeric = result.map(r => r.value === null ? null : Number(r.value.toFixed(6)));
      const available = numeric.every(x => x !== null);
      if (available) completeRows++;
      const provenance = result.map((r,v) => `${fields[v]}=${r.method === 'missing' ? 'NONE' :
        (r.method === 'fallback' ? 'NEAREST:' : 'IDW:') + r.selected.map(g => g.code).join('|')}`).join(';');
      rows.push([...station,expectedTimes[h],...numeric,available,provenance]);
    }
    // Author bounded batches through Artifact Tool; serialize values without
    // squeezing >1 million records into an Excel worksheet.
    let wb = Workbook.create();
    let sheet = wb.worksheets.add('Hourly');
    sheet.getRange(`A1:K${rows.length+1}`).values = [columns,...rows];
    wb.recalculate();
    let authored = sheet.getRange(`A2:K${rows.length+1}`).values;
    assert.deepEqual(authored, rows);
    await file.write(authored.map(row => row.map(csvCell).join(',')).join('\n') + '\n');
    rowsWritten += authored.length;
    if (!wrotePreview) {
      console.log((await wb.inspect({kind:'region',sheetId:'Hourly',range:'A1:K3',maxChars:1900})).ndjson);
      let preview = Workbook.create();
      const p = preview.worksheets.add('Preview');
      p.getRange('A1:K6').values = [columns,...rows.slice(0,5)];
      p.getRange('A1:K6').format.font = {name:'Arial',size:11};
      p.getRange('A1:K6').format.rowHeight = 48;
      p.getRange('A1:K1').format.fill = '#173B55';
      p.getRange('A1:K1').format.font = {bold:true,color:'#FFFFFF'};
      p.getRange('A1:K6').format.columnWidth = 15;
      p.getRange('A1:A6').format.columnWidth = 19;
      p.getRange('B1:B6').format.columnWidth = 29;
      p.getRange('F1:F6').format.columnWidth = 25;
      p.getRange('J1:J6').format.columnWidth = 23;
      p.getRange('K1:K6').format.columnWidth = 66;
      p.getRange('K1:K6').format.wrapText = true;
      p.getRange('G2:I6').setNumberFormat('0.000000');
      preview.recalculate();
      const rendered = await preview.render({sheetName:'Preview',autoCrop:'all',scale:1.5,format:'png'});
      await fs.writeFile(path.join(process.cwd(),'subway_weather_preview.png'),new Uint8Array(await rendered.arrayBuffer()));
      wrotePreview = true;
      preview = null;
    }
    authored = null; sheet = null; wb = null;
    if (global.gc) global.gc();
    if (si === 0 || (si+1)%25 === 0 || si === stations.length-1) {
      console.log(`Completed ${si+1}/${stations.length} stations; ${rowsWritten} rows; RSS ${Math.round(process.memoryUsage().rss/1024/1024)} MiB`);
    }
  }
} finally {
  await file.close();
}
assert.equal(rowsWritten, 1251648);
await fs.rename(output + '.partial', output);
await fs.writeFile(path.join(root,'subway_weather_weights.json'), JSON.stringify(catalog,null,2)+'\n','utf8');
const metadata = {created_at_utc:new Date().toISOString(),output:'subway_weather_hourly.csv',
  columns,station_count:stations.length,hours_per_station:expectedTimes.length,rows:rowsWritten,
  first_timestamp:expectedTimes[0],last_timestamp:expectedTimes.at(-1),timezone:'America/New_York',
  station_source:'../stations.csv',station_source_sha256:sha256(stationSource.text),weather_sources:sourceHashes,
  pending_stations:pending,complete_rows:completeRows,incomplete_rows:rowsWritten-completeRows,
  idw:{k:3,power:2,distance:'Haversine, Earth mean radius 6371.0088 km',
    fixed_candidates:'nearest 3 among the 6 downloaded stations, fixed across variables and time',
    partial_missing:'use the available subset of the fixed nearest 3; renormalize weights separately for each variable',
    fallback:'only if all fixed nearest 3 are missing for that variable, use the nearest available other gauge at the same hour',
    no_available_source:'leave that variable empty; weather_available=False',
    zero_distance:'only colocated valid observations are used, with equal weight if multiple',
    missing_definition:'empty/nonfinite source variable; input PNO filtering already applied',
    precision:'weighted calculations in double precision; output weather values rounded to 6 decimal places'},
  weather_station_encoding:'rain_mm=IDW:CODE|CODE;temp_c=NEAREST:CODE;wind_ms=NONE. IDW lists actual contributors in distance order; NEAREST is all-top3-missing fallback. Reconstruct weights from subway_weather_weights.json.',
  counts_by_variable:stats,contributor_counts_by_variable:sourceUse,
  limitations:['Only 6 of the requested 10 gauges are downloaded.',
    'Unflagged JRB precipitation retains the source values; residual undermeasurement is possible. Known PNO hours are missing and never treated as dry.',
    'Source accumulation windows are METAR reporting periods, not exact clock-hour rain measurements.',
    'These are interpolated weather estimates at subway coordinates, not on-site gauge observations.'],
  verification:'awaiting independent source and weight reconciliation'};
await fs.writeFile(path.join(root,'subway_weather_metadata.json'),JSON.stringify(metadata,null,2)+'\n','utf8');
console.log(JSON.stringify({rows:rowsWritten,completeRows,stats,output}));
