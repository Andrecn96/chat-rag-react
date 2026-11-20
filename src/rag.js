// Motor de recuperación (la "R" de RAG).
// 1. Partimos cada documento en fragmentos (chunks).
// 2. Cuando llega una pregunta, calificamos cada fragmento con BM25
//    (el mismo algoritmo que usan buscadores como Elasticsearch).
// 3. Devolvemos los fragmentos con mejor puntuación para dárselos a la IA.

const STOPWORDS = new Set(
  `a al algo como con de del el ella ellos en entre era es esa ese eso esta este
  esto fue ha hay la las le les lo los mas me mi muy no nos o otra otro para pero
  por que quien se ser si sin sobre son su sus tambien te tiene tu un una uno unos
  y ya yo cual cuales cuando donde cuanto cuanta cuantos cuantas hace puedo puede
  the of and to is in it for on`.split(/\s+/)
);

// Raíz muy simple: quita plurales para que "perros" y "perro" coincidan.
function stem(word) {
  if (word.length > 4 && word.endsWith('es')) return word.slice(0, -2);
  if (word.length > 3 && word.endsWith('s')) return word.slice(0, -1);
  return word;
}

// Quita acentos, pasa a minúsculas y separa en palabras sin stopwords.
export function tokenize(text) {
  return text
    .toLowerCase()
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .split(/[^a-z0-9ñ]+/)
    .filter((w) => w.length > 1 && !STOPWORDS.has(w))
    .map(stem);
}

// Divide el texto en fragmentos de ~maxChars respetando los párrafos.
// Si un párrafo es muy largo, se corta por oraciones.
export function chunkText(text, maxChars = 400) {
  const paragraphs = text
    .split(/\n\s*\n/)
    .map((p) => p.trim())
    .filter(Boolean);

  const chunks = [];
  let current = '';

  const push = () => {
    if (current.trim()) chunks.push(current.trim());
    current = '';
  };

  for (const p of paragraphs) {
    if (p.length > maxChars) {
      push();
      const sentences = p.match(/[^.!?]+[.!?]*\s*/g) ?? [p];
      for (const s of sentences) {
        if ((current + s).length > maxChars) push();
        current += s;
      }
      push();
    } else if ((current + '\n\n' + p).length > maxChars) {
      push();
      current = p;
    } else {
      current = current ? current + '\n\n' + p : p;
    }
  }
  push();
  return chunks;
}

// Construye el índice a partir de la lista de documentos [{ name, text }].
export function buildIndex(documents) {
  const chunks = [];
  documents.forEach((doc) => {
    chunkText(doc.text).forEach((text, i) => {
      const tokens = tokenize(text);
      const tf = new Map();
      tokens.forEach((t) => tf.set(t, (tf.get(t) ?? 0) + 1));
      chunks.push({ id: `${doc.name}#${i + 1}`, source: doc.name, text, tf, length: tokens.length });
    });
  });

  // df = en cuántos fragmentos aparece cada palabra
  const df = new Map();
  chunks.forEach((c) => c.tf.forEach((_, term) => df.set(term, (df.get(term) ?? 0) + 1)));

  const avgLength = chunks.reduce((sum, c) => sum + c.length, 0) / (chunks.length || 1);
  return { chunks, df, avgLength };
}

// BM25: premia fragmentos que contienen las palabras de la pregunta,
// sobre todo las palabras raras (idf alto), y penaliza fragmentos muy largos.
export function search(index, query, topK = 3) {
  const { chunks, df, avgLength } = index;
  const N = chunks.length;
  const k1 = 1.5;
  const b = 0.75;
  const terms = [...new Set(tokenize(query))];

  return chunks
    .map((chunk) => {
      let score = 0;
      for (const term of terms) {
        const f = chunk.tf.get(term);
        if (!f) continue;
        const n = df.get(term);
        const idf = Math.log(1 + (N - n + 0.5) / (n + 0.5));
        score += idf * ((f * (k1 + 1)) / (f + k1 * (1 - b + b * (chunk.length / avgLength))));
      }
      return { ...chunk, score };
    })
    .filter((c) => c.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);
}
