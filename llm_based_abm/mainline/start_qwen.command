#!/bin/zsh
# Run Qwen in its own visible Terminal with a saved transcript.
cd "${0:A:h}" || exit 1
export DASHSCOPE_API_KEY="$(/opt/anaconda3/envs/dev/bin/python -c 'print(open("../key-qwen.txt").read().strip().strip(chr(34)+chr(39)))')"
mkdir -p results/qwen38_flash_mainline_run01/private
run_log="results/qwen38_flash_mainline_run01/private/terminal_$(date +%Y%m%d_%H%M%S).log"
print -r -- 'Qwen3.8 Flash mainline: thinking disabled; press Ctrl+C once to save and stop.'
print -r -- "Log: $run_log"
/usr/bin/script -q "$run_log" /opt/anaconda3/envs/dev/bin/python -u run.py llm --config configs/qwen_flash.local.json
run_exit=$?
print -r -- "Run ended (exit code $run_exit). Results and cached replies are saved."
read 'run_reply?Press Enter to close this session.'
exit "$run_exit"
