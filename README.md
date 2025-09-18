## Response Envelope (v0.2)

Every agent returns a JSON object (the “envelope”) for the UI to render:

{
  "version": "0.2",
  "display_text": "Main answer (markdown allowed)",
  "snippets": [{ "doc_id": "...", "page": 4, "header_path": "§...", "text": "..." }],
  "structured": { "mime": "application/json", "content": "{...}" },   // optional
  "tables": [{ "title": "...", "columns": [...], "rows": [[...], ...] }], // optional
  "alerts": [{ "level": "info|warning|error", "text": "..." }],       // optional
  "telemetry": { "route_to": "...", "route_score": 0.62, "route_sim": 0.41 } // optional
}
