
  # Website with Chatbot and Finances



  ## Running the code

  Run `npm i` to install the dependencies.

Run `npm run dev` to start the development server.

## Backend service

The chatbot logic and audio features now live in a Python FastAPI service under `backend/`.

1. Install dependencies (preferably inside a virtualenv):
   ```bash
   pip install -r backend/requirements.txt
   ```
2. Export your OpenAI credentials (or add them to a `.env` file consumed by your process manager):
   ```bash
   set OPENAI_API_KEY=your-key
   set OPENAI_BASE_URL=https://openai-hub.neuraldeep.tech/v1
   ```
   Adjust `OPENAI_BASE_URL` if you are using a different gateway. `BACKEND_CORS_ORIGINS` can be set to a comma-separated list of origins if you need to tighten CORS (defaults to `*`).
3. Start the service:
   ```bash
   uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
   ```

## Frontend ↔ Backend connection

The React client looks for `VITE_API_BASE_URL` (defaults to `http://localhost:8000`). Create a `.env` file in the project root if you need to point at a different host/port:

```
VITE_API_BASE_URL=http://localhost:8000
```

Voice features rely on the browser MediaRecorder API, which is only available on secure origins (https or `http://localhost`).  
