// asm — the model-zoo assembler compiled to WebAssembly (GOOS=js GOARCH=wasm).
//
// Exposes: plan(spec string) -> JSON pipeline plan.
// The WASM module is the "brain" that runs in the browser; the LiteLLM LLM
// plan generation (richer) is a browser-side fetch to the local proxy.
package main

import (
	"encoding/json"
	"strings"
	"syscall/js"
)

type Stage struct {
	Name  string `json:"name"`
	Tool  string `json:"tool"`
	Cost  string `json:"cost"`
	Notes string `json:"notes"`
}

type Plan struct {
	Spec    string  `json:"spec"`
	Intent  string  `json:"intent"`
	Stages  []Stage `json:"stages"`
	Recipes []string `json:"recipe_refs"`
}

var intentMap = []struct{ pattern, intent string }{
	{"coder", "code"}, {"code", "code"}, {"programming", "code"},
	{"chat", "chat"}, {"assistant", "chat"}, {"instruct", "chat"},
	{"reason", "reasoning"}, {"think", "reasoning"}, {"r1", "reasoning"},
	{"judge", "judge"}, {"classif", "judge"}, {"gate", "judge"},
	{"ternary", "ternary"}, {"bitnet", "ternary"}, {"1.58", "ternary"},
	{"distill", "distill"}, {"tiny", "distill"}, {"compress", "distill"}, {"kd", "distill"},
}

var defaultStages = []Stage{
	{"spec", "assembler", "$0", "normalize user intent -> spec"},
	{"research", "nanoresearch mcp", "$0", "literature/gap research"},
	{"recipe", "recipe_engine run", "$0", "pick build recipe"},
	{"data", "pipeline + dlane + tm", "$0", "traces -> balance -> dvc"},
	{"version", "dvc", "$0", "versioned dataset"},
	{"train", "trl/mlflow", "$0", "SFT/DPO/GRPO"},
	{"distill", "tiny ladder", "$0", "compress to tiny gguf"},
	{"eval", "trace_eval + gates", "$0", "pass/fail"},
	{"serve", "llama.cpp/ollama", "$0", "OpenAI endpoint"},
}

func classify(spec string) string {
	low := strings.ToLower(spec)
	for _, m := range intentMap {
		if strings.Contains(low, m.pattern) {
			return m.intent
		}
	}
	return "default"
}

func plan(spec string) Plan {
	intent := classify(spec)
	p := Plan{Spec: spec, Intent: intent, Stages: defaultStages}
	if intent == "code" {
		p.Recipes = []string{"recipes/kaggle.yaml", "recipes/default.json"}
	} else if intent == "distill" {
		p.Recipes = []string{"recipes/kaggle.yaml"}
	}
	return p
}

func planJS(this js.Value, args []js.Value) any {
	spec := "a model"
	if len(args) > 0 {
		spec = args[0].String()
	}
	b, _ := json.Marshal(plan(spec))
	return string(b)
}

func main() {
	js.Global().Set("asmPlan", js.FuncOf(planJS))
	js.Global().Get("console").Call("log", "model-zoo assembler (wasm) loaded")
	select {}
}