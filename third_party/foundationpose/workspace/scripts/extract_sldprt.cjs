// Source-faithful extraction of the attached CAD's saved display mesh.
// Decoder: blussyya/sldprt-format-research (MIT); no external CAD service.
const fs = require('fs');
const crypto = require('crypto');
const path = require('path');
const decoder = require('../vendor/sldprt-format-research/src');
const [source, target, report] = process.argv.slice(2);
if (!source || !target || !report) throw new Error('usage: extract_sldprt.cjs INPUT OUTPUT.stl REPORT.json');
const bytes = fs.readFileSync(source);
const parsed = decoder.parse(bytes);
const audit = {
  source: path.resolve(source), source_sha256: crypto.createHash('sha256').update(bytes).digest('hex'),
  decoder: 'sldprt@0.5.0', decoder_commit: '93b49335cfcd8b47e93878a0d9935c376a86169d',
  representation: 'SolidWorks saved display tessellation', coordinate_unit: 'm',
  format: parsed.format, stats: parsed.stats, partial: parsed.partial,
  errors: parsed.errors, rejected: parsed.rejected, warnings: parsed.warnings,
  analytic_faces: parsed.faces.map(f => ({id: f.metadata?.rawId, type: f.metadata?.typeTag,
    parameters: f.metadata?.parameters, bounds: f.bounds})),
};
fs.writeFileSync(report, JSON.stringify(audit, null, 2) + '\n');
if (parsed.errors.length || parsed.partial || parsed.rejected.length || !parsed.faces.length)
  throw new Error('CAD tessellation is incomplete; inspect the report or export STEP from SolidWorks');
fs.writeFileSync(target, decoder.toSTL(bytes, {scale: 1}));
console.log(JSON.stringify({faces: parsed.stats.faces, triangles: parsed.stats.triangles, unit: 'm'}));
