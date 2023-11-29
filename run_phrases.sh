#!/bin/bash

# Check if the input file is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <input_file> --rev_author_dict_path <path_to_rev_auth> --key <key_value>"
    exit 1
fi

# Set the command to run for each phrase
COMMAND="torchrun --nproc_per_node 1 paper_search_llm_phrase.py --ckpt_dir llama-2-7b-chat/ --tokenizer_path tokenizer.model --max_seq_len 512 --max_batch_size 6"

# Parse additional arguments
while [ "$#" -gt 1 ]; do
    case "$2" in
        --key)
            KEY_VALUE="$3"
            COMMAND+=" --key $KEY_VALUE"
            shift 2
            ;;
        *)
            echo "Invalid argument: $2"
            exit 1
            ;;
    esac
done

# Read each line from the input file and execute the command
while IFS= read -r phrase; do
    echo "Running command for phrase: $phrase"
    $COMMAND --phrase "$phrase"
done < "$1"

echo "Script completed."
