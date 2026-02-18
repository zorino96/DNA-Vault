import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# DNA Partition Vault and CNN Extractor
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

class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
    def forward(self, x):
        features = self.conv(x)
        return features.view(features.size(0), -1)

# ==========================================
# Models: Baseline vs DNA-Vault
# ==========================================
class StandardCNN(nn.Module):
    def __init__(self):
        super(StandardCNN, self).__init__()
        self.features = CNNFeatureExtractor()
        self.classifier = nn.Sequential(nn.Linear(4096, 512), nn.ReLU(), nn.Linear(512, 10))
    def forward(self, x):
        return self.classifier(self.features(x))

class AutoPartitionCNN(nn.Module):
    def __init__(self):
        super(AutoPartitionCNN, self).__init__()
        self.features = CNNFeatureExtractor()
        self.layer1 = DNAPartitionLayer(4096, 512, num_blocks=2, is_last_layer=False)
        self.layer2 = DNAPartitionLayer(512, 10, num_blocks=2, is_last_layer=True)
        self.register_buffer('prototypes', torch.zeros(2, 4096))

    def forward(self, x, task_id=None):
        x_features = self.features(x)
        x_norm = F.normalize(x_features, p=2, dim=1)
        
        if task_id is not None:
            if self.training:
                with torch.no_grad():
                    self.prototypes[task_id] = 0.9 * self.prototypes[task_id] + 0.1 * x_norm.mean(dim=0)
            return self.layer2(F.relu(self.layer1(x_norm, task_id)), task_id)
        else:
            dist_0 = torch.norm(x_norm - self.prototypes[0], dim=1)
            dist_1 = torch.norm(x_norm - self.prototypes[1], dim=1)
            use_vault_1 = (dist_1 < dist_0).unsqueeze(1)
            out_0 = self.layer2(F.relu(self.layer1(x_norm, 0)), 0)
            out_1 = self.layer2(F.relu(self.layer1(x_norm, 1)), 1)
            return torch.where(use_vault_1, out_1, out_0)

# ==========================================
# Training & Benchmarking
# ==========================================
if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10('./data', train=False, transform=transform)

    train_loader = DataLoader(torch.utils.data.Subset(train_dataset, range(20000)), batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    baseline_model = StandardCNN().to(device)
    dna_model = AutoPartitionCNN().to(device)

    base_opt = torch.optim.Adam(baseline_model.parameters(), lr=0.001)
    dna_opt = torch.optim.Adam(dna_model.parameters(), lr=0.001)

    def train_models(task_id, inverted=False, epochs=5):
        baseline_model.train()
        dna_model.train()
        for epoch in range(epochs):
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                if inverted: data = -data 
                
                base_opt.zero_grad()
                F.cross_entropy(baseline_model(data), target).backward()
                base_opt.step()
                
                dna_opt.zero_grad()
                F.cross_entropy(dna_model(data, task_id=task_id), target).backward()
                dna_opt.step()

    def test_models(inverted=False):
        baseline_model.eval()
        dna_model.eval()
        base_correct = dna_correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                if inverted: data = -data
                
                base_pred = baseline_model(data).argmax(dim=1, keepdim=True)
                base_correct += base_pred.eq(target.view_as(base_pred)).sum().item()
                
                dna_pred = dna_model(data, task_id=None).argmax(dim=1, keepdim=True)
                dna_correct += dna_pred.eq(target.view_as(dna_pred)).sum().item()
                
        return 100. * base_correct / len(test_loader.dataset), 100. * dna_correct / len(test_loader.dataset)

    print("--- Starting DNA-Vault CIFAR-10 Scalability Benchmark ---")
    print("\n1. Training Task 0 (Normal CIFAR-10)...")
    train_models(task_id=0, inverted=False, epochs=5)
    
    print("\n2. Training Task 1 (Inverted CIFAR-10)...")
    train_models(task_id=1, inverted=True, epochs=5)
    
    print("\n3. Final Catastrophic Forgetting Benchmark...")
    base_final, dna_final = test_models(inverted=False)
    print(f"   [!] Baseline Task 0 Retention: {base_final:.2f}% (Suffers Forgetting)")
    print(f"   [+] DNA-Vault Task 0 Retention: {dna_final:.2f}% (Demonstrates CNN Feature Drift)")
