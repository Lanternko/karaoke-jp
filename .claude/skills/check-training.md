# check-training

Check the status of a running training job (CE+CTC retrain or other GPU training).

## Usage

`/check-training` — check all active tmux training sessions.

## Instructions

1. **List tmux sessions**:
   ```bash
   tmux list-sessions 2>/dev/null
   ```

2. **For each training session**, capture the last ~30 lines of output:
   ```bash
   tmux capture-pane -t <session> -p -S -30
   ```

3. **Parse and report**:
   - Current epoch / total epochs
   - Train loss and valid loss trends
   - Whether it survived past known failure points (e.g., CE+CTC run4 OOM at epoch 5)
   - Estimated time remaining if possible (from per-epoch timing)
   - GPU memory usage: `nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader`

4. **For CE+CTC retrain specifically** (the active training as of 2026-06-12):
   - Check if `run4` directory exists: `ls ~/side_projects/music-ai/karaoke-jp/benchmarks/ctc_ce_run4/`
   - Report whether epoch 5+ was reached (prior runs OOM-killed there)
   - If training is complete, remind user about the posttrain eval chain:
     pick best 3 epochs → test-82 eval → threshold sweep → final retrained-on-RoFormer number
