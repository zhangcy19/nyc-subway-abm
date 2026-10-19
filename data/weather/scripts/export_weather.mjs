// Run with bundled Node from a work directory whose node_modules junction
// points to the bundled runtime. CSVs retain the reference file's six columns.
import fs from 'node:fs/promises';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';

const runtimeRequire = createRequire(path.join(process.cwd(), '_runtime_loader.cjs'));
const { Workbook } = await import(pathToFileURL(runtimeRequire.resolve('@oai/artifact-tool')).href);
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const tables = JSON.parse(await fs.readFile(path.join(root, 'prepared_tables.json'), 'utf8'));
const inventoryPath = path.join(root, 'station_inventory.json');
const manifest = JSON.parse(await fs.readFile(inventoryPath, 'utf8'));
const previews = path.join(process.cwd(), 'previews');
await fs.mkdir(previews, { recursive: true });

function csvCell(value) {
  if (value === null || value === undefined) return '';
  const text = typeof value === 'boolean' ? (value ? 'True' : 'False') : String(value);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

const workbook = Workbook.create();
for (const [code, table] of Object.entries(tables)) {
  const sheet = workbook.worksheets.add(code);
  sheet.getRange(`A1:F${table.rows.length + 1}`).values = [table.headers, ...table.rows];
  // Formatting only affects the internal inspection preview, never the CSV.
  sheet.getRange('A1:F9').format.font = { name: 'Arial', size: 11 };
  sheet.getRange('A1:F1').format.fill = '#173B55';
  sheet.getRange('A1:F1').format.font = { bold: true, color: '#FFFFFF' };
  sheet.getRange('A1:F9').format.rowHeight = 24;
  sheet.getRange('A1:A9').format.columnWidth = 25;
  sheet.getRange('B1:D9').format.columnWidth = 15;
  sheet.getRange('E1:F9').format.columnWidth = 24;
  sheet.getRange('B2:D9').setNumberFormat('0.00000');
}
workbook.recalculate();

for (const [code, table] of Object.entries(tables)) {
  const sheet = workbook.worksheets.getItem(code);
  const values = sheet.getRange(`A1:F${table.rows.length + 1}`).values;
  if (JSON.stringify(values) !== JSON.stringify([table.headers, ...table.rows])) {
    throw new Error(`Artifact round trip changed values for ${code}`);
  }
  console.log((await workbook.inspect({ kind: 'region', sheetId: code, range: 'A1:F3', maxChars: 1200 })).ndjson);
  const csv = values.map(row => row.map(csvCell).join(',')).join('\n') + '\n';
  const filename = `weather_hourly_${code}.csv`;
  await fs.writeFile(path.join(root, filename), csv, 'utf8');
  const station = manifest.stations.find(item => item.station === code);
  station.exported = true;
  station.output_rows = table.rows.length;
  station.status = 'exported_with_explicit_missing_values';

  // A small bounded preview avoids rendering thousands of hourly rows.
  const previewWorkbook = Workbook.create();
  const previewSheet = previewWorkbook.worksheets.add(code);
  previewSheet.getRange('A1:F9').copyFrom(sheet.getRange('A1:F9'), 'all');
  previewSheet.getRange('A1:A9').format.columnWidth = 25;
  previewSheet.getRange('B1:D9').format.columnWidth = 15;
  previewSheet.getRange('E1:F9').format.columnWidth = 24;
  previewWorkbook.recalculate();
  const rendered = await previewWorkbook.render({sheetName: code, autoCrop: 'all', scale: 1.5, format: 'png'});
  await fs.writeFile(path.join(previews, `${code}.png`), new Uint8Array(await rendered.arrayBuffer()));
  console.log(`Exported ${filename}: ${table.rows.length} rows`);
}
manifest.exported_at_utc = new Date().toISOString();
await fs.writeFile(inventoryPath, JSON.stringify(manifest, null, 2) + '\n', 'utf8');
