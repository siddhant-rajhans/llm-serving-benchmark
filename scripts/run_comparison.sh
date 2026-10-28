#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
CONCURRENCY="${CONCURRENCY:-16}"
NUM_REQUESTS="${NUM_REQUESTS:-500}"
OUTPUT_DIR="${OUTPUT_DIR:-results}"

echo "============================================"
echo " LLM Serving Benchmark: Full Comparison"
echo "============================================"
echo "Model:       $MODEL"
echo "Concurrency: $CONCURRENCY"
echo "Requests:    $NUM_REQUESTS"
echo "Output:      $OUTPUT_DIR"
echo "============================================"
echo ""

llm-bench compare \
    --model "$MODEL" \
    --concurrency "$CONCURRENCY" \
    --num-requests "$NUM_REQUESTS" \
    --output-dir "$OUTPUT_DIR" \
    --patterns "uniform,bursty,ramp_up,cyclic,spike"

echo ""
echo "============================================"
echo " Routing cost estimate"
echo "============================================"
llm-bench estimate-routing \
    --model "$MODEL" \
    --large-model "meta-llama/Llama-3.1-70B-Instruct" \
    --num-samples 1000

echo ""
echo "Done. Results in $OUTPUT_DIR/"
