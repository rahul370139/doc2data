# Doc2Data — Frontend

Next.js 14 + TypeScript + Tailwind interface for the Doc2Data LangGraph pipeline.

## Features

- Drag-and-drop upload for PDFs / images
- Server-Sent-Events **live progress** from the agentic graph
  (`load → identify → plan → [A|B|C] → validate → reflect → rescue → finalize`)
- Results tabs: **Fields · Business schema · Validation · Debug**
- Downloadable raw JSON (Reducto-compatible)

## Development

```bash
cd doc2data/frontend
cp .env.example .env.local
# edit .env.local to point at your backend:
#   API_BASE_URL=http://localhost:8000
npm install
npm run dev
```

The dev server listens on [http://localhost:3000](http://localhost:3000) and
proxies `/api/backend/*` to `API_BASE_URL` — so the frontend never needs CORS.

## Production build

```bash
npm run build
npm run start
```

Or use the Docker image built by `deploy_and_run.sh`, which serves the
compiled `standalone` output on port 3000 behind the same container
network as the FastAPI backend (port 8000).
