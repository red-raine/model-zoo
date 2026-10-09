# model-zoo frontend-wasm (D7)

The Model Zoo Builder frontend: **describe a model → get a pipeline plan**,
running in any browser via a Go→WASM assembler module + optional LiteLLM agent
plan.

## Files
| File | Purpose |
|------|---------|
| `asm/main.go` | the assembler brain compiled to WASM (`asmPlan(spec)` → JSON stage chain) |
| `asm.wasm` | the compiled module (GOOS=js GOARCH=wasm) |
| `wasm_exec.js` | Go's wasm runtime glue (from Go lib/wasm) |
| `index.html` | the single-page chat UI |

## Run
```bash
# serve this dir (any static server)
python -m http.server 8000 --directory frontend-wasm
# open http://localhost:8000 ; enter LiteLLM key when prompted
```
Buttons:
- **Plan pipeline** — runs the WASM assembler locally (offline, instant).
- **Ask LiteLLM agent** — sends the spec to `nyx-qwen25-1.5b` (or any proxy
  model) for the richer 8-stage plan.

## Rebuild the wasm
```bash
cd frontend-wasm/asm
$env:GOOS="js"; $env:GOARCH="wasm"
go build -o ../asm.wasm .
```

## Contract
`asmPlan(spec) -> {spec, intent, stages:[{name,tool,cost,notes}], recipe_refs}`
Intents: code/chat/reasoning/judge/ternary/distill/default.