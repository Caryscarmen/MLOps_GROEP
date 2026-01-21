import torch

# 1. Point to the file you found (Adjust path if needed!)
# You said it was in runs/mlp_baseline
checkpoint_path = "experiments/results/mlp_baseline/checkpoint_epoch_2.pt" 

print(f"🔍 Loading {checkpoint_path}...")

# 2. Load the file (map_location='cpu' ensures it works even without a GPU)
checkpoint = torch.load(checkpoint_path, map_location="cpu")

# 3. Extract the hidden config dictionary
config = checkpoint['config']

# 4. Print the proof
print("\n✅ CONFIGURATION FOUND INSIDE CHECKPOINT:")
print("-" * 40)
print(f"Seed:          {config.get('seed')}")
print(f"Learning Rate: {config['training']['learning_rate']}")
print(f"Batch Size:    {config['data']['batch_size']}")
print(f"Hidden Units:  {config['model']['hidden_units']}")
print(f"Patience:      {config['training']['scheduler']['patience']}")
print("-" * 40)

# 5. (Optional) Check the performance saved in that file
print(f"Saved F2-Score: {checkpoint.get('val_f2')}")