// ════════════════════════════════════════════════════════════
//  Aura AI — document text extraction (runs in the browser)
//
//  Turns an uploaded File into plain text for the RAG pipeline.
//  The file type is detected from its content (magic bytes / zip
//  entries), falling back to the extension, so renamed files work.
//
//  Third-party parsers are loaded from /vendor by index.html:
//    pdf.js (pdfjsLib)   PDF
//    mammoth             .docx
//    SheetJS (XLSX)      .xlsx .xls .xlsb .ods .csv … and XLSX.CFB for
//                        legacy OLE files (.doc / .ppt readers below)
//    JSZip               .pptx .odt .odp .epub
//
//  Usage:  const {text, kind, note} = await AuraExtract.extractText(file, msg => …)
// ════════════════════════════════════════════════════════════
(function () {
'use strict';

class ExtractError extends Error {}

const IMAGE_EXT   = new Set(['png','jpg','jpeg','gif','webp','bmp','tif','tiff','heic','heif','ico','avif']);
const MEDIA_EXT   = new Set(['mp3','wav','m4a','aac','flac','ogg','opus','mp4','mov','avi','mkv','webm','wmv']);
const ARCHIVE_EXT = new Set(['zip','rar','7z','tar','gz','tgz','bz2','xz','dmg','iso','exe','dll','so','dylib','bin','jar','apk','class']);

const cp1252 = new TextDecoder('windows-1252');
const utf16  = new TextDecoder('utf-16le');

const extOf = name => { const i = name.lastIndexOf('.'); return i < 0 ? '' : name.slice(i + 1).toLowerCase(); };
const startsWith = (u8, bytes) => bytes.every((b, i) => u8[i] === b);
const toU8 = c => c instanceof Uint8Array ? c : Uint8Array.from(c || []);

// ── XML helpers (namespace-prefix agnostic) ──────────────────
const parseXml  = s => new DOMParser().parseFromString(s, 'application/xml');
const parseHtml = s => new DOMParser().parseFromString(s, 'text/html');   // inert: no scripts run, nothing loads
const byLocal   = (root, name) => root ? [...root.getElementsByTagNameNS('*', name)] : [];

function resolvePath(baseDir, target) {           // zip-internal relative path → absolute
  if (target.startsWith('/')) return target.slice(1);
  const parts = (baseDir + target).split('/'), out = [];
  for (const p of parts) { if (p === '..') out.pop(); else if (p !== '.') out.push(p); }
  return out.join('/');
}
const dirOf = p => p.slice(0, p.lastIndexOf('/') + 1);

// ════════════════════════════════════════════════════════════
//  ENTRY POINT
// ════════════════════════════════════════════════════════════
async function extractText(file, onStatus = () => {}) {
  onStatus('Reading file…');
  const buf = new Uint8Array(await file.arrayBuffer());
  if (!buf.length) throw new ExtractError('The file is empty.');
  const ext = extOf(file.name);
  const ctx = { buf, ext, onStatus };
  const kind = await detect(ctx);

  let r;
  switch (kind) {
    case 'pdf':   r = await readPdf(ctx); break;
    case 'docx':  r = await readDocx(ctx); break;
    case 'doc':   r = readLegacyDoc(ctx); break;
    case 'pptx':  r = await readPptx(ctx); break;
    case 'ppt':   r = readLegacyPpt(ctx); break;
    case 'sheet': r = readSheet(ctx); break;
    case 'odt':
    case 'odp':   r = await readOdf(ctx, kind); break;
    case 'epub':  r = await readEpub(ctx); break;
    case 'rtf':   r = rtfToText(decodeText(buf)); break;
    case 'html':  r = htmlToText(parseHtml(decodeText(buf))); break;
    case 'xml':   r = xmlToText(decodeText(buf)); break;
    case 'text':  r = readPlain(ctx); break;
    default:      throw new ExtractError(unsupportedMsg(kind, ext));
  }
  if (typeof r === 'string') r = { text: r };
  const text = clean(r.text || '');
  if (!text) throw new ExtractError('No readable text found in this file.');
  return { text, kind, note: r.note || '' };
}

function unsupportedMsg(kind, ext) {
  switch (kind) {
    case 'image':     return 'This is an image. Reading text from images (OCR) is not supported yet — upload a document with selectable text.';
    case 'media':     return 'Audio and video files are not supported.';
    case 'archive':   return 'Archives and programs are not supported. Unzip the files and upload them individually.';
    case 'apple':     return 'Apple Pages / Keynote files are not supported. In Pages or Keynote use File → Export To → PDF or Word/PowerPoint, then upload that.';
    case 'encrypted': return 'This file is password-protected. Remove the password and try again.';
    case 'ole':       return 'This Microsoft Office file type is not supported (only Word, PowerPoint and Excel).';
    default:          return `Unsupported file type${ext ? ' (.' + ext + ')' : ''}.`;
  }
}

// ── Format detection ────────────────────────────────────────
async function detect(ctx) {
  const { buf, ext } = ctx;
  if (startsWith(buf, [0x25, 0x50, 0x44, 0x46])) return 'pdf';                              // %PDF
  if (startsWith(buf, [0xD0, 0xCF, 0x11, 0xE0, 0xA1, 0xB1, 0x1A, 0xE1])) return detectOle(ctx);  // OLE2
  if (startsWith(buf, [0x50, 0x4B, 0x03, 0x04])) return detectZip(ctx);                     // PK zip
  if (startsWith(buf, [0x7B, 0x5C, 0x72, 0x74, 0x66])) return 'rtf';                        // {\rtf
  if (IMAGE_EXT.has(ext) || isImage(buf)) return 'image';
  if (MEDIA_EXT.has(ext))   return 'media';
  if (ARCHIVE_EXT.has(ext)) return 'archive';
  if (['csv', 'tsv'].includes(ext)) return 'sheet';
  if (['html', 'htm', 'xhtml', 'mhtml'].includes(ext)) return 'html';
  if (['xml', 'svg', 'rss', 'atom', 'plist'].includes(ext)) return 'xml';
  return 'text';
}

function isImage(b) {
  return startsWith(b, [0x89, 0x50, 0x4E, 0x47]) || startsWith(b, [0xFF, 0xD8, 0xFF]) ||
         startsWith(b, [0x47, 0x49, 0x46, 0x38]) ||
         (startsWith(b, [0x52, 0x49, 0x46, 0x46]) && startsWith(b.subarray(8), [0x57, 0x45, 0x42, 0x50]));
}

function detectOle(ctx) {
  let cfb;
  try { cfb = XLSX.CFB.read(ctx.buf, { type: 'array' }); }
  catch (e) { throw new ExtractError('This Office file is damaged and could not be opened.'); }
  ctx.cfb = cfb;
  const names = new Set(cfb.FileIndex.map(f => f.name));
  if (names.has('EncryptedPackage') || names.has('EncryptionInfo')) return 'encrypted';
  if (names.has('WordDocument'))        return 'doc';
  if (names.has('PowerPoint Document')) return 'ppt';
  if (names.has('Workbook') || names.has('Book')) return 'sheet';
  return 'ole';
}

async function detectZip(ctx) {
  let zip;
  try { zip = await JSZip.loadAsync(ctx.buf); }
  catch (e) { throw new ExtractError('This file looks like a zip-based document but could not be opened (damaged?).'); }
  ctx.zip = zip;
  const has = p => !!zip.file(p);
  if (has('word/document.xml'))        return 'docx';
  if (has('ppt/presentation.xml'))     return 'pptx';
  if (has('xl/workbook.xml') || has('xl/workbook.bin')) return 'sheet';
  if (has('mimetype')) {
    const mt = (await zip.file('mimetype').async('string')).trim();
    if (mt === 'application/epub+zip') return 'epub';
    if (mt.endsWith('opendocument.text') || mt.endsWith('opendocument.text-template')) return 'odt';
    if (mt.endsWith('opendocument.presentation')) return 'odp';
    if (mt.endsWith('opendocument.spreadsheet'))  return 'sheet';
  }
  if (has('Index/Document.iwa') || Object.keys(zip.files).some(n => n.startsWith('Index/')))
    return ctx.ext === 'numbers' ? 'sheet' : 'apple';
  return 'archive';
}

// ════════════════════════════════════════════════════════════
//  READERS
// ════════════════════════════════════════════════════════════

// ── PDF ──────────────────────────────────────────────────────
async function readPdf({ buf, onStatus }) {
  pdfjsLib.GlobalWorkerOptions.workerSrc = 'vendor/pdf.worker.min.js';
  let pdf;
  try {
    // isEvalSupported:false — never compile fonts into JS (guards against malicious PDFs)
    pdf = await pdfjsLib.getDocument({ data: buf, isEvalSupported: false }).promise;
  } catch (e) {
    if (e && e.name === 'PasswordException') throw new ExtractError(unsupportedMsg('encrypted'));
    throw new ExtractError('Could not open this PDF: ' + (e && e.message || e));
  }
  const pages = [];
  let chars = 0;
  try {
    for (let i = 1; i <= pdf.numPages; i++) {
      onStatus(`Reading page ${i}/${pdf.numPages}…`);
      const page = await pdf.getPage(i);
      const tc = await page.getTextContent();
      let s = '';
      for (const it of tc.items) {
        if (typeof it.str !== 'string') continue;
        s += it.str;
        if (it.hasEOL) s += '\n';
      }
      chars += s.replace(/\s/g, '').length;
      if (s.trim()) pages.push(`--- Page ${i} ---\n${s}`);
      page.cleanup();
    }
  } finally {
    pdf.destroy();
  }
  if (!chars)
    throw new ExtractError('This PDF has no selectable text — it looks like a scanned document (images only). Text recognition (OCR) is not supported yet.');
  const note = chars < 40 * pdf.numPages ? 'some pages look scanned — their text could not be read' : '';
  return { text: pages.join('\n\n'), note };
}

// ── Word .docx ───────────────────────────────────────────────
async function readDocx({ buf }) {
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  const r = await mammoth.extractRawText({ arrayBuffer: ab });
  return r.value;
}

// ── Word 97-2003 .doc (binary) ───────────────────────────────
// Reads the piece table (CLX) from the table stream and pulls each
// text piece out of the WordDocument stream. [MS-DOC] §2.4.1
function readLegacyDoc({ cfb }) {
  const stream = name => { const e = cfb.FileIndex.find(f => f.name === name); return e ? toU8(e.content) : null; };
  const wd = stream('WordDocument');
  const dv = new DataView(wd.buffer, wd.byteOffset, wd.byteLength);
  if (dv.getUint16(0, true) !== 0xA5EC) throw new ExtractError('Unrecognised Word file.');
  if (dv.getUint16(2, true) < 101) throw new ExtractError('Word 95 and older files are not supported. Open it in Word and save as .docx.');
  const flags = dv.getUint16(0x0A, true);
  if (flags & 0x0100) throw new ExtractError(unsupportedMsg('encrypted'));
  const tbl = stream(flags & 0x0200 ? '1Table' : '0Table');
  if (!tbl) throw new ExtractError('This Word file is damaged (missing table stream).');
  const tv = new DataView(tbl.buffer, tbl.byteOffset, tbl.byteLength);

  // FIB: FibBase(32) csw(2) FibRgW(csw*2) cslw(2) FibRgLw(cslw*4) cbRgFcLcb(2) FibRgFcLcb…
  const csw  = dv.getUint16(32, true);
  const cslw = dv.getUint16(34 + csw * 2, true);
  const rgFcLcb = 34 + csw * 2 + 2 + cslw * 4 + 2;
  const fcClx  = dv.getUint32(rgFcLcb + 66 * 4, true);        // fcClx is the 34th fc/lcb pair
  const lcbClx = dv.getUint32(rgFcLcb + 66 * 4 + 4, true);

  let p = fcClx;
  const end = fcClx + lcbClx;
  while (p < end && tbl[p] === 0x01) p += 3 + tv.getUint16(p + 1, true);   // skip Prc blocks
  if (tbl[p] !== 0x02) throw new ExtractError('This Word file is damaged (no piece table).');
  const lcb = tv.getUint32(p + 1, true);
  p += 5;
  const n = (lcb - 4) / 12;
  let raw = '';
  for (let i = 0; i < n; i++) {
    const cpStart = tv.getUint32(p + i * 4, true), cpEnd = tv.getUint32(p + (i + 1) * 4, true);
    const fc = tv.getUint32(p + (n + 1) * 4 + i * 8 + 2, true);
    const len = cpEnd - cpStart;
    if (fc & 0x40000000) {                                      // compressed: 8-bit cp1252
      const off = (fc & 0x3FFFFFFF) / 2;
      raw += cp1252.decode(wd.subarray(off, off + len));
    } else {                                                    // UTF-16LE
      raw += utf16.decode(wd.subarray(fc, fc + len * 2));
    }
  }

  // Drop field codes (keep field results) and map Word's special characters
  let out = '';
  const fields = [];
  for (const ch of raw) {
    if (ch === '\x13') { fields.push('code'); continue; }
    if (ch === '\x14') { if (fields.length) fields[fields.length - 1] = 'result'; continue; }
    if (ch === '\x15') { fields.pop(); continue; }
    if (fields.includes('code')) continue;
    switch (ch) {
      case '\r': case '\x0b': case '\x0c': case '\x0e': out += '\n'; break;
      case '\x07': out += '\t'; break;                          // table cell / row end
      case '\x1e': out += '-'; break;                           // non-breaking hyphen
      case '\x01': case '\x02': case '\x05': case '\x08': case '\x1f': break;  // objects, refs, soft hyphen
      default: out += ch;
    }
  }
  return out;
}

// ── PowerPoint .pptx ─────────────────────────────────────────
async function readPptx({ zip, onStatus }) {
  const xml = async path => { const f = zip.file(path); return f ? parseXml(await f.async('string')) : null; };
  const rels = async part => {                 // relationships of a part: [{type, target}]
    const doc = await xml(dirOf(part) + '_rels/' + part.slice(part.lastIndexOf('/') + 1) + '.rels');
    return byLocal(doc, 'Relationship').map(r => ({
      id: r.getAttribute('Id'), type: r.getAttribute('Type') || '',
      target: resolvePath(dirOf(part), r.getAttribute('Target') || '') }));
  };
  const R_NS = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships';

  // Slide order comes from presentation.xml; fall back to file numbering
  const presRels = await rels('ppt/presentation.xml');
  const relById  = Object.fromEntries(presRels.map(r => [r.id, r.target]));
  let slides = byLocal(await xml('ppt/presentation.xml'), 'sldId')
    .map(s => relById[s.getAttributeNS(R_NS, 'id') || s.getAttribute('r:id')]).filter(Boolean);
  if (!slides.length)
    slides = Object.keys(zip.files).filter(n => /^ppt\/slides\/slide\d+\.xml$/.test(n))
      .sort((a, b) => parseInt(a.match(/\d+/)) - parseInt(b.match(/\d+/)));

  const paras = doc => byLocal(doc, 'p')
    .map(p => byLocal(p, 't').map(t => t.textContent).join(''))
    .filter(s => s.trim());

  const out = [];
  for (const [i, path] of slides.entries()) {
    onStatus(`Reading slide ${i + 1}/${slides.length}…`);
    let s = paras(await xml(path)).join('\n');
    const notes = (await rels(path)).find(r => r.type.endsWith('/notesSlide'));
    if (notes) {
      const n = paras(await xml(notes.target)).filter(t => !/^\d+$/.test(t.trim()));  // drop slide-number field
      if (n.length) s += '\nSpeaker notes: ' + n.join('\n');
    }
    if (s.trim()) out.push(`--- Slide ${i + 1} ---\n${s}`);
  }
  return out.join('\n\n');
}

// ── PowerPoint 97-2003 .ppt (binary) ─────────────────────────
// Walks the record tree of the "PowerPoint Document" stream collecting
// TextCharsAtom / TextBytesAtom records. Depending on the PowerPoint
// version, slide text lives either in SlideListWithText (grouped per
// slide by SlidePersistAtom) or inside each Slide container's drawing.
function readLegacyPpt({ cfb }) {
  const e = cfb.FileIndex.find(f => f.name === 'PowerPoint Document');
  const s = toU8(e.content);
  const dv = new DataView(s.buffer, s.byteOffset, s.byteLength);
  const MASTER = /^(Click to edit (the )?Master .*|Second level|Third level|Fourth level|Fifth level|\*|‹#›)$/i;
  const listSlides = [], drawSlides = [], notes = [];

  const add = (list, t) => {
    if (!list) return;
    for (const line of t.split(/[\r\n\x0b]/)) {
      const l = line.trim();
      if (l && !MASTER.test(l) && !list.includes(l)) list.push(l);
    }
  };
  // target: array receiving text found at this level (null = ignore)
  (function walk(o, end, target, slwt) {
    while (o + 8 <= end) {
      const verInst = dv.getUint16(o, true), type = dv.getUint16(o + 2, true), len = dv.getUint32(o + 4, true);
      const body = o + 8, next = body + len;
      if (next > end) break;
      if ((verInst & 0xF) === 0xF) {                               // container
        switch (type) {
          case 0x0FF0: walk(body, next, null, verInst >> 4); break;  // SlideListWithText: 0 slides, 1 masters, 2 notes
          case 0x03EE: { const t = []; drawSlides.push(t); walk(body, next, t, -1); break; }  // Slide
          case 0x03F0: walk(body, next, notes, -1); break;            // Notes
          case 0x03F8: case 0x0FC9: break;                            // MainMaster, Handout: skip
          default:     walk(body, next, target, slwt);
        }
      } else if (type === 0x03F3 && slwt === 0) {                  // SlidePersistAtom → next slide
        target = []; listSlides.push(target);
      } else if (type === 0x0FA0 || type === 0x0FA8) {             // TextCharsAtom (UTF-16) / TextBytesAtom (8-bit)
        add(target, type === 0x0FA0 ? utf16.decode(s.subarray(body, next)) : cp1252.decode(s.subarray(body, next)));
      }
      o = next;
    }
  })(0, s.length, null, -1);

  // Merge both sources slide by slide (usually only one of them has text)
  const n = Math.max(listSlides.length, drawSlides.length), parts = [];
  for (let i = 0; i < n; i++) {
    const lines = [...(listSlides[i] || [])];
    for (const l of drawSlides[i] || []) if (!lines.includes(l)) lines.push(l);
    if (lines.length) parts.push(`--- Slide ${i + 1} ---\n${lines.join('\n')}`);
  }
  if (notes.length) parts.push(`--- Speaker notes ---\n${notes.join('\n')}`);
  return parts.join('\n\n');
}

// ── Spreadsheets: .xlsx .xls .xlsb .ods .csv .tsv .numbers ───
// Each row becomes "Header: value | Header: value" so a chunk stays
// understandable even without the header row.
function readSheet({ buf, ext, onStatus }) {
  let wb;
  try {
    wb = ['csv', 'tsv'].includes(ext)
      ? XLSX.read(decodeText(buf), { type: 'string', raw: true })
      : XLSX.read(buf, { type: 'array', cellDates: true, dense: true });
  } catch (e) {
    if (/password|encrypt/i.test(e.message)) throw new ExtractError(unsupportedMsg('encrypted'));
    throw new ExtractError('Could not read this spreadsheet: ' + e.message);
  }
  const out = [];
  for (const name of wb.SheetNames) {
    onStatus(`Reading sheet "${name}"…`);
    const ws = wb.Sheets[name];
    if (!ws) continue;
    const rows = XLSX.utils.sheet_to_json(ws, { header: 1, raw: false, blankrows: false, defval: '' })
      .map(r => r.map(v => String(v).trim()))
      .filter(r => r.some(Boolean));
    if (!rows.length) continue;
    const head = rows[0], filled = head.filter(Boolean);
    const isHeader = rows.length > 1 && filled.length >= 2 &&
                     filled.every(v => isNaN(Number(v.replace(/[,%$€£]/g, ''))));
    const lines = (isHeader ? rows.slice(1) : rows).map(r =>
      r.map((v, i) => v && (isHeader && head[i] ? `${head[i]}: ${v}` : v)).filter(Boolean).join(' | '));
    out.push(`--- Sheet: ${name} ---\n` + (isHeader ? `Columns: ${filled.join(', ')}\n` : '') + lines.join('\n'));
  }
  return out.join('\n\n');
}

// ── OpenDocument text / presentation (.odt .odp) ─────────────
async function readOdf({ zip }, kind) {
  const f = zip.file('content.xml');
  if (!f) throw new ExtractError('This OpenDocument file is damaged (no content.xml).');
  const doc = parseXml(await f.async('string'));

  const inline = el => {                     // text of one paragraph, honouring <text:s>, <text:tab>, …
    let s = '';
    for (const n of el.childNodes) {
      if (n.nodeType === 3) s += n.data;
      else if (n.nodeType === 1) {
        if (n.localName === 's') s += ' '.repeat(+n.getAttribute('text:c') || 1);
        else if (n.localName === 'tab') s += '\t';
        else if (n.localName === 'line-break') s += '\n';
        else s += inline(n);
      }
    }
    return s;
  };
  const paras = root => {                    // paragraphs & headings in document order
    const out = [];
    (function walk(el) {
      for (const c of el.children) {
        if (c.localName === 'p' || c.localName === 'h') { const t = inline(c); if (t.trim()) out.push(t); }
        else walk(c);
      }
    })(root);
    return out.join('\n');
  };

  if (kind === 'odp')
    return byLocal(doc, 'page').map((pg, i) => `--- Slide ${i + 1} ---\n${paras(pg)}`).join('\n\n');
  return paras(doc.documentElement);
}

// ── EPUB e-books ─────────────────────────────────────────────
async function readEpub({ zip, onStatus }) {
  const read = async p => { const f = zip.file(p); return f ? f.async('string') : null; };
  const container = parseXml(await read('META-INF/container.xml') || '<x/>');
  const opfPath = byLocal(container, 'rootfile')[0]?.getAttribute('full-path');
  if (!opfPath) throw new ExtractError('This EPUB is damaged (no package file).');
  const opf = parseXml(await read(opfPath));
  const manifest = Object.fromEntries(byLocal(opf, 'item').map(i => [i.getAttribute('id'), i.getAttribute('href')]));
  const spine = byLocal(opf, 'itemref').map(r => manifest[r.getAttribute('idref')]).filter(Boolean);
  const out = [];
  for (const [i, href] of spine.entries()) {
    onStatus(`Reading chapter ${i + 1}/${spine.length}…`);
    const html = await read(resolvePath(dirOf(opfPath), decodeURIComponent(href)));
    if (html) out.push(htmlToText(parseHtml(html)));
  }
  return out.join('\n\n');
}

// ── Plain text & code ────────────────────────────────────────
function readPlain({ buf }) {
  const text = decodeText(buf);
  const sample = text.slice(0, 20000);
  const junk = (sample.match(/[\x00-\x08\x0E-\x1F�]/g) || []).length;
  if (sample.includes('\0') || junk > sample.length * 0.05)
    throw new ExtractError('Unsupported file type — this looks like a binary file, not a document.');
  return text;
}

// Decodes bytes as UTF-8 / UTF-16 (BOM) and falls back to Windows-1252 for legacy text
function decodeText(buf) {
  if (buf[0] === 0xFF && buf[1] === 0xFE) return new TextDecoder('utf-16le').decode(buf);
  if (buf[0] === 0xFE && buf[1] === 0xFF) return new TextDecoder('utf-16be').decode(buf);
  const s = new TextDecoder('utf-8').decode(buf);
  const bad = (s.match(/�/g) || []).length;
  return bad > 0 && !buf.includes(0) && bad > s.length * 0.001 ? cp1252.decode(buf) : s;
}

// ── HTML / XML ───────────────────────────────────────────────
const BLOCK = new Set(['P','DIV','BR','LI','TR','H1','H2','H3','H4','H5','H6','SECTION','ARTICLE','HEADER',
  'FOOTER','BLOCKQUOTE','PRE','TABLE','UL','OL','DL','DT','DD','FIGURE','FIGCAPTION','HR','ADDRESS','NAV','ASIDE','MAIN','TITLE']);

function htmlToText(doc) {
  doc.querySelectorAll('script,style,noscript,template,svg,iframe,object').forEach(e => e.remove());
  let out = '';
  (function walk(n) {
    if (n.nodeType === 3) { out += n.data.replace(/\s+/g, ' '); return; }
    if (n.nodeType !== 1) return;
    const tag = n.tagName.toUpperCase(), block = BLOCK.has(tag);
    if (block) out += '\n';
    for (const c of n.childNodes) walk(c);
    if (tag === 'TD' || tag === 'TH') out += '\t';
    if (block) out += '\n';
  })(doc.body || doc.documentElement);
  return out;
}

function xmlToText(s) {
  const doc = parseXml(s);
  if (doc.getElementsByTagName('parsererror').length) return s;   // not well-formed: index it as-is
  const parts = [];
  const tw = doc.createTreeWalker(doc.documentElement, NodeFilter.SHOW_TEXT);
  while (tw.nextNode()) { const t = tw.currentNode.data.trim(); if (t) parts.push(t); }
  return parts.join('\n');
}

// ── RTF ──────────────────────────────────────────────────────
// Small tokenizer: skips non-text destinations (font tables, pictures, …),
// decodes \'hh (Windows-1252) and \uN escapes.
const RTF_SKIP = new Set(['fonttbl','colortbl','stylesheet','info','pict','object','themedata','colorschememapping',
  'latentstyles','datastore','xmlnstbl','listtable','listoverridetable','rsidtbl','generator','filetbl','revtbl',
  'fldinst','nonshppict','shppict','bkmkstart','bkmkend','xe','tc','private','pgdsctbl','mmathPr','header',
  'footer','headerl','headerr','headerf','footerl','footerr','footerf','listtext','pntext','expandedcolortbl']);
const RTF_CHARS = { par: '\n', line: '\n', sect: '\n\n', page: '\n\n', tab: '\t', cell: '\t', row: '\n',
  emdash: '—', endash: '–', bullet: '•', lquote: '‘', rquote: '’', ldblquote: '“', rdblquote: '”',
  emspace: ' ', enspace: ' ', qmspace: ' ' };

function rtfToText(rtf) {
  const re = /\\([a-z]{1,32})(-?\d{1,10})? ?|\\'([0-9a-f]{2})|\\([^a-z])|([{}])|[\r\n]+|([^\\{}\r\n]+)/gi;
  const stack = [];
  let out = '', ignorable = false, uc = 1, skip = 0, bytes = [];
  const flush = () => { if (bytes.length) { out += cp1252.decode(new Uint8Array(bytes)); bytes = []; } };
  for (const m of rtf.matchAll(re)) {
    const [, word, arg, hex, sym, brace, text] = m;
    if (hex !== undefined) {
      if (skip > 0) skip--; else if (!ignorable) bytes.push(parseInt(hex, 16));
      continue;
    }
    flush();
    if (brace) {
      if (brace === '{') stack.push([ignorable, uc]);
      else [ignorable, uc] = stack.pop() || [false, 1];
      skip = 0;
    } else if (sym) {
      if (sym === '*') ignorable = true;
      else if (ignorable) continue;
      else if (sym === '~') out += ' ';
      else if (sym === '\n' || sym === '\r') out += '\n';
      else if ('\\{}'.includes(sym)) out += sym;
    } else if (word) {
      if (RTF_SKIP.has(word)) ignorable = true;
      else if (ignorable) continue;
      else if (word === 'uc') uc = +arg || 0;
      else if (word === 'u') { let c = +arg; if (c < 0) c += 65536; out += String.fromCharCode(c); skip = uc; }
      else if (RTF_CHARS[word]) out += RTF_CHARS[word];
    } else if (text) {
      let t = text;
      if (skip > 0) { const k = Math.min(skip, t.length); t = t.slice(k); skip -= k; }
      if (!ignorable) out += t;
    }
  }
  flush();
  return out;
}

// ── Final clean-up ───────────────────────────────────────────
function clean(t) {
  return t.normalize('NFKC')   // NFKC also splits PDF ligatures (ﬁ → fi) and fixes lookalike CJK radicals
    .replace(/[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/g, '�')  // lone surrogates
    .replace(/\r\n?/g, '\n')
    .replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F ]/g, ' ')
    .replace(/[ \t]+\n/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

window.AuraExtract = { extractText, ExtractError };
})();
