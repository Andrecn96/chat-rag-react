// Conexión con la API de IA (OpenRouter, compatible con el formato de OpenAI).
// Esta es la "G" de RAG: la IA genera la respuesta usando el contexto recuperado.

const API_URL = 'https://openrouter.ai/api/v1/chat/completions';

// Arma el mensaje de sistema con los fragmentos encontrados.
export function buildSystemPrompt(contextChunks) {
  const context = contextChunks.length
    ? contextChunks.map((c, i) => `[${i + 1}] (fuente: ${c.source})\n${c.text}`).join('\n\n---\n\n')
    : '(No se encontró información relevante en los documentos.)';

  return `Eres un asistente que responde preguntas usando ÚNICAMENTE el contexto de abajo.
Reglas:
- Si la respuesta está en el contexto, respóndela de forma clara y breve, en español.
- Escribe en texto plano, sin Markdown (nada de ** ni #).
- Cita los fragmentos que usaste con su número, por ejemplo [1] o [2].
- Si el contexto no contiene la respuesta, di que no lo encuentras en los documentos. No inventes.

CONTEXTO:
${context}`;
}

// Envía la conversación y va entregando el texto en partes (streaming)
// mediante onToken, para que la respuesta aparezca mientras se escribe.
export async function streamChat({ apiKey, model, messages, onToken, signal }) {
  const res = await fetch(API_URL, {
    method: 'POST',
    signal,
    headers: {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
      'X-Title': 'Chat RAG React',
    },
    body: JSON.stringify({ model, messages, stream: true, temperature: 0.2 }),
  });

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const data = await res.json();
      detail = data.error?.message ?? detail;
    } catch {
      /* la respuesta no era JSON */
    }
    throw new Error(`Error ${res.status}: ${detail}`);
  }

  // La respuesta llega como Server-Sent Events: líneas "data: {...}"
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let full = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const lines = buffer.split('\n');
    buffer = lines.pop(); // la última línea puede venir incompleta

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') return full;
      try {
        const json = JSON.parse(payload);
        if (json.error) throw new Error(json.error.message);
        const token = json.choices?.[0]?.delta?.content ?? '';
        if (token) {
          full += token;
          onToken(full);
        }
      } catch (err) {
        if (err instanceof SyntaxError) continue; // fragmento JSON incompleto
        throw err;
      }
    }
  }
  return full;
}
