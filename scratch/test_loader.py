import torch
import torch.nn as nn

# Load checkpoint
ckpt = torch.load('results/sen2sr_lite_compound_finetuned.pth', map_location='cpu')
print(f"Total keys in checkpoint: {len(ckpt)}")
first_key = list(ckpt.keys())[0]
print(f"First key: {first_key}")

# Let's verify how to identify the matching module
class MockSubModule(nn.Module):
    def __init__(self):
        super().__init__()
        # Checkpoint has conv_1.sk.weight
        self.conv_1 = nn.Conv2d(4, 24, 1)
        self.conv_2 = nn.Conv2d(24, 24, 1)

mock = MockSubModule()
print(f"hasattr conv_1: {hasattr(mock, 'conv_1')}")
print(f"hasattr conv1: {hasattr(mock, 'conv1')}")

# Also check direct state dict keys match
mock_keys = set(k.split('.')[0] for k in ckpt.keys())
print(f"Top-level prefixes in checkpoint: {mock_keys}")
