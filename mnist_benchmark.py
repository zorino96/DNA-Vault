import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ==========================================
# 1. DNA Partition Vault (The Modular Brain)
# ==========================================
class DNAPartitionLayer(nn.Module):
    def __init__(self, in_features, out_features, num_blocks=2, is_last_layer=False):
        super(DNAPartitionLayer, self).__init__()
        self.num_blocks = num_blocks
        self.is_last_layer = is_last_layer
        self.in_features = in_features
        self.out_features = out_features
        
        self.weights = nn.Parameter(torch.randn(out_features, in_features))
        nn.init.xavier_uniform_(self.weights)
        
        if is_last_layer:
            self.bias = nn.Parameter(torch.zeros(num_blocks, out_features))
        else:
            self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x, task_id):
        mask = torch.zeros_like(self.weights)
        
        if not self.is_last_layer:
            block_size = self.out_features // self.num_blocks
            start = task_id * block_size
            end = (task_id + 1) * block_size
            mask[start:end, :] = 1.0
            mask_bias = torch.zeros_like(self.bias)
            mask_bias[start:end] = 1.0
            masked_bias = self.bias * mask_bias
        else:
            block_size = self.in_features // self.num_blocks
            start = task_id * block_size
            end = (task_id + 1) * block_size
            mask[:, start:end] = 1.0
            masked_bias = self.bias[task_id]
            
        masked_weights = self.weights * mask
        return F.linear(x, masked_weights, masked_bias)

# ==========================================
# 2. Task-Free Continual Learning Architecture
# ==========================================
class AutoPartitionModel(nn.Module):
    def __init__(self):
        super(AutoPartitionModel, self).__init__()
        self.layer1 = DNAPartitionLayer(28*28, 512, num_blocks=2, is_last_layer=False)
        self.layer2 = DNAPartitionLayer(512, 10, num_blocks=2, is_last_layer=True)
        
        # Prototype Memory: Stores average feature representations per task
        self.register_buffer('prototypes', torch.zeros(2, 28*28))

    def forward(self, x, task_id=None):
        x_flat = x.view(x.size(0), -1)
        
        # L2 Normalization to prevent OOD logit explosions
        x_norm = F.normalize(x_flat, p=2, dim=1)
        
        if task_id is not None:
            # Training Phase: Update Prototype Memory via Moving Average
            if self.training:
                with torch.no_grad():
                    self.prototypes[task_id] = 0.9 * self.prototypes[task_id] + 0.1 * x_norm.mean(dim=0)
                    
            out = F.relu(self.layer1(x_norm, task_id))
            return self.layer2(out, task_id)
        else:
            # Inference Phase: Task-Free Prototype Routing
            dist_0 = torch.norm(x_norm - self.prototypes[0], dim=1)
            dist_1 = torch.norm(x_norm - self.prototypes[1], dim=1)
            
            # Autonomously route to the vault with the closest prototype
            use_vault_1 = (dist_1 < dist_0).unsqueeze(1)
            
            out_0 = self.layer2(F.relu(self.layer1(x_norm, 0)), 0)
            out_1 = self.layer2(F.relu(self.layer1(x_norm, 1)), 1)
            
            return torch.where(use_vault_1, out_1, out_0)

# ==========================================
# 3. Data Preparation & Training Loop
# ==========================================
if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(torch.utils.data.Subset(train_dataset, range(10000)), batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    model = AutoPartitionModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    def train(loader, task_id, inverted=False):
        model.train()
        for data, target in loader:
            if inverted: data = 1.0 - data
            optimizer.zero_grad()
            output = model(data, task_id=task_id) 
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

    def test_auto(loader, inverted=False):
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in loader:
                if inverted: data = 1.0 - data
                # No task_id provided -> Task-Free Inference
                output = model(data, task_id=None) 
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        return 100. * correct / len(loader.dataset)

    print("--- Starting DNA-Vault Split MNIST Benchmark ---")
    print("\n1. Training Task 0 (Normal MNIST)...")
    for epoch in range(3): train(train_loader, task_id=0, inverted=False)
    print(f"   [+] Task 0 Accuracy (Auto-Routed): {test_auto(test_loader, inverted=False):.2f}%")

    print("\n2. Training Task 1 (Inverted MNIST)...")
    for epoch in range(3): train(train_loader, task_id=1, inverted=True)
    print(f"   [+] Task 1 Accuracy (Auto-Routed): {test_auto(test_loader, inverted=True):.2f}%")

    print("\n3. Final Catastrophic Forgetting Benchmark...")
    print(f"   [!] Task 0 Accuracy (After learning Task 1): {test_auto(test_loader, inverted=False):.2f}%")
