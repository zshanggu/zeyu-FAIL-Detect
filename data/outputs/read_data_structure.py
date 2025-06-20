import torch

# Load the file
data = torch.load('can_data_diffusion.pt', map_location='cpu')

# Check the type and keys
print(type(data))
if isinstance(data, dict):
    print("Keys:", list(data.keys()))
    
# For state dictionaries, inspect parameter shapes
if isinstance(data, dict):
    for key, value in data.items():
        if torch.is_tensor(value):
            print(f"{key}: {value.shape}")

X, Y = data['X'], data['Y']

# Check data types and ranges
print("X stats:")
print(f"  Min: {X.min()}, Max: {X.max()}")
print(f"  Mean: {X.mean()}, Std: {X.std()}")

print("Y stats:")
print(f"  Min: {Y.min()}, Max: {Y.max()}")
print(f"  Mean: {Y.mean()}, Std: {Y.std()}")

# Look at a few samples
print("Sample X[0]:", X[0].shape)  # First 10 elements
print("Sample Y[0]:", Y[0].shape)  # First 10 elements