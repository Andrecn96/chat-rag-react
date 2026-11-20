import { useMemo, useRef, useState, useEffect } from 'react';
import { buildIndex, search } from './rag';
import { buildSystemPrompt, streamChat } from './api';
import { SAMPLE_DOCS } from './sampleDocs';
import './App.css';

const DEFAULT_MODEL = import.meta.env.VITE_OPENROUTER_MODEL || 'openrouter/free';
// La clave del .env solo se usa en desarrollo (npm run dev), nunca en un build publicado.
const ENV_KEY = import.meta.env.DEV ? import.meta.env.VITE_OPENROUTER_API_KEY || '' : '';

function readStorage(key, fallback) {
  try {
    return localStorage.getItem(key) || fallback;
  } catch {
    return fallback;
  }
}

export default function App() {
  const [documents, setDocuments] = useState(SAMPLE_DOCS);
  const [messages, setMessages] = useState([]); // { role, content, sources? }
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [apiKey, setApiKey] = useState(() => readStorage('openrouter_key', ENV_KEY));
  const [model, setModel] = useState(() => readStorage('openrouter_model', DEFAULT_MODEL));
  const [pasteName, setPasteName] = useState('');
  const [pasteText, setPasteText] = useState('');
  const abortRef = useRef(null);
  const endRef = useRef(null);

  // El índice se reconstruye solo cuando cambian los documentos.
  const index = useMemo(() => buildIndex(documents), [documents]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    try {
      localStorage.setItem('openrouter_key', apiKey);
      localStorage.setItem('openrouter_model', model);
    } catch {
      /* sin almacenamiento disponible */
    }
  }, [apiKey, model]);

  async function handleFiles(e) {
    const files = [...e.target.files];
    const loaded = await Promise.all(files.map(async (f) => ({ name: f.name, text: await f.text() })));
    setDocuments((docs) => [...docs, ...loaded]);
    e.target.value = '';
  }

  function addPasted() {
    if (!pasteText.trim()) return;
    const name = pasteName.trim() || `texto-${documents.length + 1}.txt`;
    setDocuments((docs) => [...docs, { name, text: pasteText }]);
    setPasteName('');
    setPasteText('');
  }

  function removeDoc(name) {
    setDocuments((docs) => docs.filter((d) => d.name !== name));
  }

  async function send(e) {
    e.preventDefault();
    const question = input.trim();
    if (!question || loading) return;
    if (!apiKey) {
      setError('Falta la clave de OpenRouter (ponla en Configuración).');
      return;
    }

    setError('');
    setInput('');

    // 1) Recuperar: buscamos los fragmentos más parecidos a la pregunta.
    const sources = search(index, question, 3);

    // 2) Aumentar: metemos esos fragmentos en el prompt de sistema.
    const history = [...messages, { role: 'user', content: question }];
    const apiMessages = [
      { role: 'system', content: buildSystemPrompt(sources) },
      ...history.slice(-6).map(({ role, content }) => ({ role, content })),
    ];

    setMessages([...history, { role: 'assistant', content: '', sources }]);
    setLoading(true);

    // 3) Generar: la IA responde con base en ese contexto.
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      await streamChat({
        apiKey,
        model,
        messages: apiMessages,
        signal: controller.signal,
        onToken: (text) =>
          setMessages((msgs) => {
            const copy = [...msgs];
            copy[copy.length - 1] = { ...copy[copy.length - 1], content: text };
            return copy;
          }),
      });
    } catch (err) {
      if (err.name !== 'AbortError') {
        setError(err.message);
        setMessages((msgs) => msgs.slice(0, -1)); // quitamos la burbuja vacía
      }
    } finally {
      setLoading(false);
      abortRef.current = null;
    }
  }

  const totalChunks = index.chunks.length;

  return (
    <div className="layout">
      <aside className="sidebar">
        <h1>Chat RAG</h1>
        <p className="muted">
          Pregunta sobre tus documentos. La app busca los fragmentos relevantes y se los pasa a la IA.
        </p>

        <section>
          <h2>Documentos ({documents.length})</h2>
          <p className="muted small">{totalChunks} fragmentos indexados</p>
          <ul className="doc-list">
            {documents.map((d) => (
              <li key={d.name}>
                <span title={d.name}>{d.name}</span>
                <button className="link" onClick={() => removeDoc(d.name)} aria-label={`Quitar ${d.name}`}>
                  quitar
                </button>
              </li>
            ))}
          </ul>

          <label className="file-btn">
            Subir .txt o .md
            <input type="file" accept=".txt,.md,.csv,.json" multiple onChange={handleFiles} />
          </label>

          <details>
            <summary>Pegar texto</summary>
            <input
              placeholder="Nombre (opcional)"
              value={pasteName}
              onChange={(e) => setPasteName(e.target.value)}
            />
            <textarea
              rows={5}
              placeholder="Pega aquí el contenido…"
              value={pasteText}
              onChange={(e) => setPasteText(e.target.value)}
            />
            <button onClick={addPasted}>Agregar documento</button>
          </details>
        </section>

        <section>
          <details>
            <summary>Configuración</summary>
            <label>
              Clave de OpenRouter
              <input type="password" value={apiKey} onChange={(e) => setApiKey(e.target.value)} placeholder="sk-or-…" />
            </label>
            <label>
              Modelo
              <input value={model} onChange={(e) => setModel(e.target.value)} />
            </label>
          </details>
        </section>
      </aside>

      <main className="chat">
        <div className="messages">
          {messages.length === 0 && (
            <div className="empty">
              <p>Prueba con:</p>
              {['¿A qué hora abren los sábados?', '¿Cuánto cuesta el desayuno y hasta qué hora lo sirven?', '¿Puedo llevar a mi perro?'].map(
                (q) => (
                  <button key={q} className="suggestion" onClick={() => setInput(q)}>
                    {q}
                  </button>
                )
              )}
            </div>
          )}

          {messages.map((m, i) => (
            <div key={i} className={`msg ${m.role}`}>
              <div className="bubble">{m.content || (loading && i === messages.length - 1 ? 'Pensando…' : '')}</div>
              {m.role === 'assistant' && m.sources && (
                <details className="sources">
                  <summary>
                    {m.sources.length ? `Fuentes usadas (${m.sources.length})` : 'Sin fuentes relevantes'}
                  </summary>
                  {m.sources.map((s, j) => (
                    <blockquote key={s.id}>
                      <strong>
                        [{j + 1}] {s.id}
                      </strong>{' '}
                      <span className="muted small">puntaje {s.score.toFixed(2)}</span>
                      <p>{s.text}</p>
                    </blockquote>
                  ))}
                </details>
              )}
            </div>
          ))}
          <div ref={endRef} />
        </div>

        {error && <p className="error">{error}</p>}

        <form className="composer" onSubmit={send}>
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Escribe tu pregunta…"
            disabled={loading}
          />
          {loading ? (
            <button type="button" onClick={() => abortRef.current?.abort()}>
              Detener
            </button>
          ) : (
            <button type="submit" disabled={!input.trim()}>
              Enviar
            </button>
          )}
        </form>
      </main>
    </div>
  );
}
