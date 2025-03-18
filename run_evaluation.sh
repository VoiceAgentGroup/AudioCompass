#!/bin/bash

# run_evaluation.sh - Automated script for running VoiceBench evaluations
# Usage: ./run_evaluation.sh <model> <dataset> <split> <modality>

if [ $# -lt 4 ]; then
    echo "Usage: $0 <model> <dataset> <split> <modality>"
    echo ""
    echo "Arguments:"
    echo "  model    - Model to evaluate (e.g., qwen2, diva)"
    echo "  dataset  - Dataset to use (e.g., alpacaeval, commoneval, sd-qa, ifeval, advbench, openbookqa, mmsu)"
    echo "  split    - Data split (test, or region code like 'usa' for sd-qa)"
    echo "  modality - Input modality (audio or text)"
    exit 1
fi

MODEL=$1
DATASET=$2
SPLIT=$3
MODALITY=$4

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}======================================================${NC}"
echo -e "${BLUE}    Running VoiceBench Evaluation for $MODEL    ${NC}"
echo -e "${BLUE}======================================================${NC}"
echo ""

# Step 1: Get the model's response
echo -e "${GREEN}[Step 1] Getting responses from the $MODEL model for $DATASET-$SPLIT-$MODALITY...${NC}"
python main.py --model $MODEL --data $DATASET --split $SPLIT --modality $MODALITY

# Define output and result filenames
OUTPUT_FILE="output/$MODEL-$DATASET-$SPLIT-$MODALITY.jsonl"
RESULT_FILE="result/$MODEL-$DATASET-$SPLIT-$MODALITY.jsonl"

# Check if output file was created
if [ ! -f "$OUTPUT_FILE" ]; then
    echo -e "${YELLOW}Error: Output file $OUTPUT_FILE was not created. Check for errors in the model generation step.${NC}"
    exit 1
fi

# Step 2: Run GPT-4 evaluation if needed
if [[ "$DATASET" == "alpacaeval" || "$DATASET" == "commoneval" || "$DATASET" == "sd-qa" ]]; then
    echo -e "${GREEN}[Step 2] Running GPT-4 evaluation on model responses...${NC}"
    python api_judge.py --src-file $OUTPUT_FILE

    # Check if result file was created
    if [ ! -f "$RESULT_FILE" ]; then
        echo -e "${YELLOW}Error: Result file $RESULT_FILE was not created. Check for errors in the GPT evaluation step.${NC}"
        exit 1
    fi

    # Step 3: Get final evaluation results
    echo -e "${GREEN}[Step 3] Generating final evaluation results...${NC}"
    
    # Select the appropriate evaluator based on the dataset
    if [[ "$DATASET" == "alpacaeval" || "$DATASET" == "commoneval" ]]; then
        EVALUATOR="open"
    elif [[ "$DATASET" == "sd-qa" ]]; then
        EVALUATOR="qa"
    fi
    
    python evaluate.py --src-file $RESULT_FILE --evaluator $EVALUATOR
else
    # For other datasets, skip GPT-4 evaluation and use the output file directly
    echo -e "${GREEN}[Step 2] Skipping GPT-4 evaluation (not needed for $DATASET)${NC}"
    
    # Step 3: Get final evaluation results
    echo -e "${GREEN}[Step 3] Generating final evaluation results...${NC}"
    
    # Select the appropriate evaluator based on the dataset
    if [[ "$DATASET" == "ifeval" ]]; then
        EVALUATOR="ifeval"
    elif [[ "$DATASET" == "advbench" ]]; then
        EVALUATOR="harm"
    elif [[ "$DATASET" == "openbookqa" || "$DATASET" == "mmsu" ]]; then
        EVALUATOR="mcq"
    else
        echo -e "${YELLOW}Warning: Unknown dataset $DATASET. Please manually specify the evaluator.${NC}"
        exit 1
    fi
    
    python evaluate.py --src-file $OUTPUT_FILE --evaluator $EVALUATOR
fi

echo -e "${BLUE}======================================================${NC}"
echo -e "${BLUE}    Evaluation Complete!    ${NC}"
echo -e "${BLUE}======================================================${NC}"
echo ""