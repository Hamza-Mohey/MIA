# Isolated training environment

The application and its GPU experiments share `.venv-training`, the single
supported Python 3.11 environment.

Python 3.11 and CUDA PyTorch were installed with:

```powershell
winget install --id Python.Python.3.11 --exact --scope user
& "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe" -m venv .venv-training
.\.venv-training\Scripts\python.exe -m pip install torch==2.12.1 `
  --index-url https://download.pytorch.org/whl/cu126
.\.venv-training\Scripts\python.exe -m pip install -r requirements-training.txt
.\.venv-training\Scripts\python.exe -m pip install -r requirements.txt
```

Cache the optimized Whisper Medium inference representation once:

```powershell
.\.venv-training\Scripts\python.exe -m modules.prefetch_faster_whisper
```

The Streamlit application launches Whisper, pyannote, and Qwen as isolated
one-shot workers. Process exit returns RAM and VRAM to Windows between stages;
training commands remain ordinary standalone processes.

Before using speaker diarization, open
<https://huggingface.co/pyannote/speaker-diarization-community-1>, accept the
conditions, create a read token, and run:

```powershell
.\.venv-training\Scripts\hf.exe auth login
```

Do not put the token in a source file or commit it.

The first Whisper adapter run should be a smoke test:

```powershell
.\.venv-training\Scripts\python.exe -m training.ami.train_whisper_lora --max-steps 20
```

Only after the smoke test and validation-loss check should a several-hour run
be started. The test partition is never passed to the trainer.

Compare decoded WER before promoting an adapter:

```powershell
.\.venv-training\Scripts\python.exe -m training.ami.evaluate_whisper_adapter
```

The first 20-step experiment did not change WER, so its adapter remains
disabled. See `training/EXPERIMENTS.md` for measured results.
