// Minimal, safe Markdown for model output: paragraphs, lists, **bold**, *italic*.

export function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

const inline = (s) => escapeHtml(s)
  .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
  .replace(/(^|[^*])\*([^*\n]+)\*/g, '$1<em>$2</em>');

export function renderMarkdown(text) {
  const out = [];
  let list = null;
  const close = () => { if (list) { out.push(`</${list}>`); list = null; } };
  for (const raw of String(text).split('\n')) {
    const line = raw.trim();
    if (!line) { close(); continue; }
    const ol = line.match(/^\d+[.)]\s+(.*)/);
    const ul = line.match(/^[-*•]\s+(.*)/);
    if (ol || ul) {
      const tag = ol ? 'ol' : 'ul';
      if (list !== tag) { close(); out.push(`<${tag}>`); list = tag; }
      out.push(`<li>${inline((ol || ul)[1])}</li>`);
    } else {
      close();
      out.push(`<p>${inline(line.replace(/^#+\s*/, ''))}</p>`);
    }
  }
  close();
  return out.join('');
}
