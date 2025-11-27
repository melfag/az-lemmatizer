# RunPod Training Quickstart - 900K Cleaned Dataset

This guide will help you train the Azerbaijani lemmatizer on RunPod with the cleaned 900K dataset.

## 📊 Dataset Info

- **Train:** 719,972 examples (cleaned)
- **Val:** 89,996 examples
- **Test:** 89,997 examples
- **Total:** 899,965 examples

## 💰 Expected Cost & Time

| GPU | Time | Cost | Recommendation |
|-----|------|------|----------------|
| RTX 4090 | ~3-4 hours | $2.50 | ⭐⭐⭐ Best value |
| A100 | ~2-3 hours | $3.50 | ⭐⭐ Fastest |
| A10 | ~5-6 hours | $2.50 | ⭐ Budget option |

## 🚀 Step-by-Step Setup

### 1. Create RunPod Account & Add Credit

```
1. Go to https://www.runpod.io
2. Sign up (requires credit card)
3. Add $10-20 credit
```

### 2. Deploy Pod

```
1. Click "Deploy" or "Rent GPUs"
2. Select GPU:
   - RTX 4090 (Recommended): $0.69/hr
   - A100: $1.10/hr
3. Select Template: "RunPod Pytorch 2.1"
4. Container Disk: 50GB
5. Volume Storage: 20GB (optional, for persistent storage)
6. Click "Deploy On-Demand"
```

### 3. Connect to Pod

Once deployed, click "Connect" → "Start Web Terminal" or use SSH:

```bash
# SSH option (shown in RunPod dashboard)
ssh root@<pod-id>.runpod.io -p <port>
```

### 4. Upload Project Files

**Option A: Clone from GitHub (if public)**
```bash
cd /workspace
git clone https://github.com/yourusername/azerbaijani-lemmatizer.git
cd azerbaijani-lemmatizer
```

**Option B: Upload via SCP (recommended for private repo)**

From your local machine:
```bash
# Upload entire project
scp -P <port> -r azerbaijani-lemmatizer root@<pod-id>.runpod.io:/workspace/

# Or upload just necessary files
scp -P <port> -r \
  data/processed/moraz_900k_train.json \
  data/processed/moraz_900k_val.json \
  data/processed/moraz_900k_test.json \
  data/processed/char_vocab.json \
  root@<pod-id>.runpod.io:/workspace/azerbaijani-lemmatizer/data/processed/
```

**Option C: Use RunPod's File Browser**
- In Web Terminal, use the file browser to drag & drop files

### 5. Install Dependencies

```bash
cd /workspace/azerbaijani-lemmatizer

# Verify GPU
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Install requirements
pip install -r requirements.txt
```

### 6. Create Training Config

Create `configs/runpod_900k_training.yaml`:

```yaml
# Data paths
data:
  train_path: "data/processed/moraz_900k_train.json"
  val_path: "data/processed/moraz_900k_val.json"
  test_path: "data/processed/moraz_900k_test.json"
  num_workers: 4

# Vocabulary
vocab_path: "data/processed/char_vocab.json"

# Model architecture
model:
  char_embedding_dim: 64
  char_hidden_dim: 128
  char_num_layers: 1
  bert_model_name: "allmalab/bert-base-aze"
  bert_freeze_layers: 8
  fusion_output_dim: 256
  decoder_hidden_dim: 256
  decoder_num_layers: 1
  dropout: 0.3
  use_copy: true
  copy_penalty: 0.3

# Training
training:
  num_epochs: 20
  batch_size: 64
  gradient_accumulation_steps: 1
  bert_lr: 1.0e-5
  other_lr: 3.0e-4
  weight_decay: 1.0e-5
  warmup_ratio: 0.15
  max_grad_norm: 1.0
  label_smoothing: 0.1
  teacher_forcing_ratio: 0.7
  teacher_forcing_schedule: 'linear'
  teacher_forcing_end_ratio: 0.1
  char_class_weight: 0.1
  attention_coverage_weight: 0.05
  transformation_loss_weight: 0.15
  early_stopping_patience: 8
  checkpoint_dir: "checkpoints/runpod_900k_training"
  save_every: 2

# Hardware
device: "cuda"
mixed_precision: true
```

### 7. Start Training (with tmux)

**IMPORTANT:** Use tmux so training continues even if you disconnect!

```bash
# Start tmux session
tmux new -s training

# Start training
python scripts/train.py --config configs/runpod_900k_training.yaml

# Detach from tmux: Press Ctrl+B, then press D
# Training continues in background!

# Later, reattach to see progress
tmux attach -t training
```

### 8. Monitor Training

```bash
# View logs
tail -f logs/training.log

# Check GPU usage
watch -n 1 nvidia-smi

# Reattach to tmux session
tmux attach -t training
```

### 9. Download Results

After training completes, download the best model:

**From your local machine:**
```bash
# Download best model
scp -P <port> root@<pod-id>.runpod.io:/workspace/azerbaijani-lemmatizer/checkpoints/runpod_900k_training/best_model.pt ./

# Download all checkpoints (optional)
scp -P <port> -r root@<pod-id>.runpod.io:/workspace/azerbaijani-lemmatizer/checkpoints/ ./
```

### 10. Stop Pod

**IMPORTANT:** Stop the pod when done to avoid charges!

```
1. Go to RunPod dashboard
2. Click "Stop" or "Terminate" on your pod
3. Verify it's stopped (no more charges)
```

## 📁 Files You Need to Upload

Minimum required files:
```
azerbaijani-lemmatizer/
├── data/processed/
│   ├── moraz_900k_train.json      (720K examples)
│   ├── moraz_900k_val.json        (90K examples)
│   ├── moraz_900k_test.json       (90K examples)
│   └── char_vocab.json
├── configs/
│   └── runpod_900k_training.yaml
├── scripts/
│   ├── train.py
│   └── evaluate.py
├── models/
│   ├── lemmatizer.py
│   ├── encoder.py
│   ├── decoder.py
│   ├── fusion_mechanism.py
│   └── components/
├── training/
│   ├── trainer.py
│   └── losses.py
├── utils/
│   └── metrics.py
└── requirements.txt
```

## ⚡ Quick Commands Reference

```bash
# Check GPU
nvidia-smi

# Start training in background
tmux new -s training
python scripts/train.py --config configs/runpod_900k_training.yaml
# Ctrl+B, D to detach

# Monitor progress
tmux attach -t training
tail -f logs/training.log

# Stop pod (in RunPod dashboard)
# Click "Stop" or "Terminate"
```

## 🐛 Troubleshooting

### Out of Memory Error
```yaml
# Reduce batch size in config
training:
  batch_size: 32  # Instead of 64
```

### Connection Lost
```bash
# Training continues in tmux!
# Just reconnect and reattach:
ssh root@<pod-id>.runpod.io -p <port>
tmux attach -t training
```

### Missing Dependencies
```bash
pip install transformers torch pyyaml tqdm
```

### CUDA Out of Memory
```yaml
# Disable mixed precision
mixed_precision: false
```

## 💡 Tips

1. **Always use tmux** - Training continues even if SSH disconnects
2. **Save checkpoints frequently** - Set `save_every: 1` for safety
3. **Monitor costs** - Check RunPod dashboard regularly
4. **Download results immediately** - Don't leave data on pod
5. **Stop pod when done** - Avoid unnecessary charges

## 📊 Expected Results

With 900K cleaned dataset:
- **Expected accuracy:** 75-85%
- **Training time:** 2-4 hours (depends on GPU)
- **Total cost:** $2-4

## 🎯 Next Steps After Training

1. Download the trained model
2. Evaluate on test set: `python scripts/evaluate.py --checkpoint checkpoints/runpod_900k_training/best_model.pt`
3. Compare with previous results
4. If results are good, proceed to thesis writing!

---

Good luck! 🚀
