import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import os
import argparse
from tqdm import tqdm
import foolbox as fb
import eagerpy as ep

os.makedirs('models', exist_ok=True)
os.makedirs('figures', exist_ok=True)
torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Means and standard deviations of datasets
data_stats = {
    "mnist": {
        "mean": (0.1307,),
        "std": (0.3081,)
    },
    "fashion_mnist": {
        "mean": (0.2860,),
        "std": (0.3530,)
    },
    "cifar": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616)
    }
}

###############################################
# Part 1: Model Implementation
###############################################

# Simple CNN for MNIST
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(x)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# CNN for Fashion-MNIST
class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.dropout1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# Function to load MNIST dataset
def load_mnist(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(data_stats['mnist']['mean'], data_stats['mnist']['std'])
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# Function to load Fashion-MNIST dataset
def load_fashion_mnist(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(data_stats['fashion_mnist']['mean'], data_stats['fashion_mnist']['std'])
    ])
    
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# Function to load CIFAR-10 dataset
def load_cifar(batch_size=128):
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(data_stats['cifar']['mean'], data_stats['cifar']['std'])
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(data_stats['cifar']['mean'], data_stats['cifar']['std'])
    ])
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# Function to train models
def train_model(model, train_loader, dataset_name, epochs=10, lr=0.001, save_model=True, save_path=None):
    if save_path is None:
        save_path = f"models/{dataset_name}_model.pth"
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Track training metrics
    training_loss = []
    training_acc = []
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Progress bar for training
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for i, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': running_loss / (i + 1), 
                'acc': 100. * correct / total
            })
        
        # Record metrics for this epoch
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        training_loss.append(epoch_loss)
        training_acc.append(epoch_acc)
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%')
    
    # Plot and save training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), training_loss)
    plt.title(f'{dataset_name} Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs+1), training_acc)
    plt.title(f'{dataset_name} Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    
    plt.tight_layout()
    fig_path = f"figures/{dataset_name}_training_curves.png"
    plt.savefig(fig_path)
    plt.close()
    
    # Save the model
    if save_model:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'training_loss': training_loss,
            'training_acc': training_acc,
            'epochs': epochs
        }, save_path)
        print(f"Model saved to {save_path}")
    
    return model

# Function to load a saved model
def load_model(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from {path}")
    return model

# Function to evaluate model
def evaluate_model(model, test_loader, dataset_name):
    model.eval()
    all_preds = []
    all_labels = []
    all_confidences = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # Get predicted class and confidence score
            probs = F.softmax(outputs, dim=1)
            confidences, predicted = probs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    
    # Analyze confidence distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(all_confidences, bins=20)
    plt.title('Distribution of Confidence Scores')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    fig_path = f"figures/{dataset_name}_confidence_dist.png"
    plt.savefig(fig_path)
    plt.close()
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    cm_path = f"figures/{dataset_name}_confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    
    return accuracy, precision, recall, f1, all_confidences

###############################################
# Part 2: Adversarial Attack Methods
###############################################

# Fast Gradient Sign Method (FGSM)
def fgsm_attack(model, images, labels, epsilon=0.1):
    # Set requires_grad attribute of tensor (important for attack)
    images.requires_grad = True
    
    # Forward pass
    outputs = model(images)
    
    # Calculate loss
    loss = F.cross_entropy(outputs, labels)
    
    # Backward pass
    model.zero_grad()
    loss.backward()
    
    # Collect datagrad
    data_grad = images.grad.data
    
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_images = images + epsilon * data_grad.sign()
    
    # Adding clipping to maintain [0,1] range
    perturbed_images = torch.clamp(perturbed_images, 0, 1)
    
    return perturbed_images

# Projected Gradient Descent (PGD) Attack
def pgd_attack(model, images, labels, epsilon=0.1, alpha=0.01, iterations=10):
    original_images = images.clone().detach()
    perturbed_images = images.clone().detach()
    
    for i in range(iterations):
        perturbed_images.requires_grad = True
        outputs = model(perturbed_images)
        model.zero_grad()
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        
        # Create adversarial example
        adv_images = perturbed_images + alpha * perturbed_images.grad.sign()
        
        # Project back to epsilon ball
        eta = torch.clamp(adv_images - original_images, -epsilon, epsilon)
        perturbed_images = torch.clamp(original_images + eta, 0, 1).detach()
    
    return perturbed_images

# DeepFool Attack - New implementation
def deepfool_attack(model, images, labels, num_classes=10, max_iter=50, overshoot=0.02):
    """
    Implementation of DeepFool attack.
    
    Parameters:
    model (nn.Module): The model to attack
    images (Tensor): Batch of images to attack
    labels (Tensor): True labels of the images
    num_classes (int): Number of classes in the dataset
    max_iter (int): Maximum number of iterations
    overshoot (float): Parameter to enhance the noise
    
    Returns:
    Tensor: Adversarial examples
    """
    model.eval()
    device = images.device
    batch_size = images.shape[0]
    
    # Initialize adversarial images as the original ones
    adv_images = images.clone().detach().requires_grad_(True)
    
    # Arrays to store results
    r_tot = torch.zeros_like(images)
    
    # Iterate over each image in the batch
    for idx in range(batch_size):
        x = adv_images[idx:idx+1].clone().detach().requires_grad_(True)
        original_label = labels[idx].item()
        
        output = model(x)
        _, pred_label = output.max(1)
        
        # If prediction is wrong, no need to attack
        if pred_label.item() != original_label:
            continue
        
        # Initialize
        r_i = torch.zeros_like(x)
        loop_i = 0
        
        while pred_label.item() == original_label and loop_i < max_iter:
            output = model(x + r_i)
            pred_label = output.argmax(1)
            
            # Calculate gradient for all classes
            grad_orig = torch.autograd.grad(output[0, original_label], x, 
                                          retain_graph=True, create_graph=False)[0]
            
            # Find the closest hyperplane (class)
            min_dist = float('inf')
            w_attack = None
            f_attack = None
            
            for k in range(num_classes):
                if k == original_label:
                    continue
                
                # Calculate gradient for target class
                grad_k = torch.autograd.grad(output[0, k], x, 
                                         retain_graph=True, create_graph=False)[0]
                
                # Calculate w_k and f_k
                w_k = grad_k - grad_orig
                f_k = output[0, k] - output[0, original_label]
                
                # Calculate distance
                dist_k = abs(f_k.item()) / (w_k.flatten().norm().item() + 1e-8)
                
                # Update if closer
                if dist_k < min_dist:
                    min_dist = dist_k
                    w_attack = w_k
                    f_attack = f_k
            
            # Calculate r_i
            r_i = r_i + (abs(f_attack.item()) / (w_attack.flatten().norm().item() ** 2 + 1e-8)) * w_attack
            
            # Update for next iteration
            x = x.clone().detach().requires_grad_(True)
            output = model(x + r_i)
            _, pred_label = output.max(1)
            
            loop_i += 1
        
        # Apply overshoot
        r_tot[idx] = (1 + overshoot) * r_i.detach()
    
    # Create and return adversarial examples
    adv_images = torch.clamp(images + r_tot, 0, 1)
    return adv_images

# Function to implement Carlini-Wagner attack using Foolbox library
def cw_attack_foolbox(model, images, labels, targeted=False, c=0.1, steps=100, lr=0.01):
    # Convert PyTorch model to Foolbox model
    bounds = (0, 1)
    preprocessing = dict(mean=np.array([0, 0, 0]).reshape((3, 1, 1)), std=np.array([1, 1, 1]).reshape((3, 1, 1)))
    
    if images.shape[1] == 1:  # MNIST/Fashion-MNIST
        preprocessing = dict(mean=np.array([0]).reshape((1, 1, 1)), std=np.array([1]).reshape((1, 1, 1)))

    
    # Create Foolbox model
    fmodel = fb.PyTorchModel(model, bounds=bounds, preprocessing=preprocessing)
    
    # Convert inputs to EagerPy tensors
    images_ep = ep.astensor(images.detach().cpu())
    labels_ep = ep.astensor(labels.detach().cpu())
    # Create attack
    attack = fb.attacks.L2CarliniWagnerAttack(
        steps=steps,
        confidence=0,
        initial_const=c,
        binary_search_steps=5
    )
    
    # Run attack
    raw_advs, clipped_advs, success = attack(fmodel, images_ep, labels_ep, epsilons=None)
    
    # Convert back to PyTorch tensor and send to device
    adv_images = torch.from_numpy(clipped_advs.raw.numpy()).to(device)
    
    return adv_images

# Function to visualize adversarial examples
def visualize_adversarial_examples(original_images, adversarial_images, original_labels, predicted_labels, attack_name, dataset_name):
    num_examples = min(5, original_images.size(0))
    
    plt.figure(figsize=(15, 6))
    for i in range(num_examples):
        # Original images
        plt.subplot(2, num_examples, i + 1)
        orig_img = original_images[i].cpu().detach().numpy()
        
        # If grayscale, remove channel dimension
        if orig_img.shape[0] == 1:
            orig_img = orig_img.squeeze()
            plt.imshow(orig_img, cmap='gray')
        else:
            # For color images, transpose to HWC format for matplotlib
            orig_img = np.transpose(orig_img, (1, 2, 0))
            plt.imshow(orig_img)
        
        plt.title(f"Original: {original_labels[i].item()}")
        plt.axis('off')
        
        # Adversarial images
        plt.subplot(2, num_examples, i + 1 + num_examples)
        adv_img = adversarial_images[i].cpu().detach().numpy()
        
        if adv_img.shape[0] == 1:
            adv_img = adv_img.squeeze()
            plt.imshow(adv_img, cmap='gray')
        else:
            adv_img = np.transpose(adv_img, (1, 2, 0))
            plt.imshow(adv_img)
        
        plt.title(f"Adv: {predicted_labels[i].item()}")
        plt.axis('off')
    
    plt.tight_layout()
    fig_path = f"figures/{dataset_name}_{attack_name}_examples.png"
    plt.savefig(fig_path)
    plt.close()
    
    # Also save the perturbation visualization
    plt.figure(figsize=(15, 3))
    for i in range(num_examples):
        plt.subplot(1, num_examples, i + 1)
        
        orig_img = original_images[i].cpu().detach().numpy()
        adv_img = adversarial_images[i].cpu().detach().numpy()
        
        # Calculate perturbation
        if orig_img.shape[0] == 1:
            pert = np.abs(adv_img.squeeze() - orig_img.squeeze())
            plt.imshow(pert, cmap='viridis')
        else:
            pert = np.abs(np.transpose(adv_img, (1, 2, 0)) - np.transpose(orig_img, (1, 2, 0)))
            pert = np.mean(pert, axis=2)
            plt.imshow(pert, cmap='viridis')
        
        plt.colorbar()
        plt.title(f"Perturbation")
        plt.axis('off')
    
    plt.tight_layout()
    pert_path = f"figures/{dataset_name}_{attack_name}_perturbations.png"
    plt.savefig(pert_path)
    plt.close()


def unnormalize(tensor, mean, std):
    mean = torch.tensor(mean)
    std = torch.tensor(std)

    while mean.dim() < tensor.dim() - 1:
        mean = mean.unsqueeze(-1)
        std = std.unsqueeze(-1)
    if mean.dim() < tensor.dim():
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    return tensor * std + mean


# Function to test model against adversarial attacks
def test_adversarial(model, test_loader, attack_fn, dataset_name, attack_name, attack_params=None):
    if attack_params is None:
        attack_params = {}
    
    model.eval()
    correct = 0
    total = 0
    
    # Store a few examples for visualization
    orig_examples = []
    adv_examples = []
    orig_labels = []
    pred_labels = []
    
    for images, labels in tqdm(test_loader, desc=f"Testing {attack_name} attack"):
        images, labels = images.to(device), labels.to(device)
        images = unnormalize(images, data_stats[dataset_name]['mean'], data_stats[dataset_name]['std'])
        # Generate adversarial examples
        perturbed_images = attack_fn(model, images, labels, **attack_params)
        
        # Get model predictions
        outputs = model(perturbed_images)
        _, predicted = outputs.max(1)
        
        # Update counters
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Store examples for visualization (from first batch only)
        if len(orig_examples) < 5:
            indices = range(min(5 - len(orig_examples), images.size(0)))
            orig_examples.extend([images[i].detach().clone() for i in indices])
            adv_examples.extend([perturbed_images[i].detach().clone() for i in indices])
            orig_labels.extend([labels[i].detach().clone() for i in indices])
            pred_labels.extend([predicted[i].detach().clone() for i in indices])
    
    final_acc = 100. * correct / total
    print(f'Accuracy under {attack_name} attack: {final_acc:.2f}%')
    
    # Visualize adversarial examples
    if orig_examples:
        visualize_adversarial_examples(
            torch.stack(orig_examples),
            torch.stack(adv_examples),
            torch.stack(orig_labels),
            torch.stack(pred_labels),
            attack_name,
            dataset_name
        )
    
    return final_acc

# Compare different attack methods and parameters
def compare_attacks(model, test_loader, dataset_name):
    # Test CW attack with different parameters
    cw_params = [
        {'c': 0.1, 'steps': 50},
        {'c': 1.0, 'steps': 50},
        {'c': 10.0, 'steps': 50}
    ]
    cw_results = []
    
    print("\nTesting CW attack with different parameters:")
    for params in cw_params:
        print(f"\nC: {params['c']}, Steps: {params['steps']}")
        attack_name = f"CW_c{params['c']}"
        acc = test_adversarial(model, test_loader, cw_attack_foolbox, dataset_name, attack_name, params)
        cw_results.append((params, acc))

    # Test DeepFool attack
    df_params = [
        {'max_iter': 30, 'overshoot': 0.02},
        {'max_iter': 50, 'overshoot': 0.02},
        {'max_iter': 50, 'overshoot': 0.05}
    ]
    df_results = []
    
    print("\nTesting DeepFool attack with different parameters:")
    for params in df_params:
        print(f"\nMax Iterations: {params['max_iter']}, Overshoot: {params['overshoot']}")
        attack_name = f"DeepFool_iter{params['max_iter']}_over{params['overshoot']}"
        acc = test_adversarial(model, test_loader, deepfool_attack, dataset_name, attack_name, params)
        df_results.append((params, acc))
    
    # Test with different epsilons for FGSM
    epsilons = [0.05, 0.1, 0.15, 0.2]
    fgsm_results = []
   
    print("\nTesting FGSM attack with different epsilon values:")
    for eps in epsilons:
        print(f"\nEpsilon: {eps}")
        attack_name = f"FGSM_eps{eps}"
        acc = test_adversarial(model, test_loader, fgsm_attack, dataset_name, attack_name, {'epsilon': eps})
        fgsm_results.append((eps, acc))
    
    # Test with different epsilons and iterations for PGD
    pgd_params = [
        {'epsilon': 0.1, 'iterations': 5},
        {'epsilon': 0.1, 'iterations': 10},
        {'epsilon': 0.2, 'iterations': 5},
        {'epsilon': 0.2, 'iterations': 10}
    ]
    pgd_results = []
    
    print("\nTesting PGD attack with different parameters:")
    for params in pgd_params:
        print(f"\nEpsilon: {params['epsilon']}, Iterations: {params['iterations']}")
        attack_name = f"PGD_eps{params['epsilon']}_iter{params['iterations']}"
        acc = test_adversarial(model, test_loader, pgd_attack, dataset_name, attack_name, params)
        pgd_results.append((params, acc))

    # Plot results
    plt.figure(figsize=(15, 12))
    
    # Plot FGSM results
    plt.subplot(2, 2, 1)
    plt.plot([x[0] for x in fgsm_results], [x[1] for x in fgsm_results], 'o-')
    plt.title('FGSM Attack Results')
    plt.xlabel('Epsilon')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    
    # Plot PGD results
    plt.subplot(2, 2, 2)
    x_labels = [f"ε={p['epsilon']}, iter={p['iterations']}" for p, _ in pgd_results]
    plt.bar(x_labels, [acc for _, acc in pgd_results])
    plt.title('PGD Attack Results')
    plt.xlabel('Parameters')
    plt.ylabel('Accuracy (%)')
    plt.xticks(rotation=45)
    
    # Plot CW results
    plt.subplot(2, 2, 3)
    x_labels = [f"C={p['c']}" for p, _ in cw_results]
    plt.bar(x_labels, [acc for _, acc in cw_results])
    plt.title('CW Attack Results')
    plt.xlabel('C value')
    plt.ylabel('Accuracy (%)')
    
    # Plot DeepFool results
    plt.subplot(2, 2, 4)
    x_labels = [f"i={p['max_iter']}, o={p['overshoot']}" for p, _ in df_results]
    plt.bar(x_labels, [acc for _, acc in df_results])
    plt.title('DeepFool Attack Results')
    plt.xlabel('Parameters')
    plt.ylabel('Accuracy (%)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    fig_path = f"figures/{dataset_name}_all_attacks_comparison.png"
    plt.savefig(fig_path)
    plt.close()
    
    # Create a summary comparison of best results for each attack method
    best_fgsm = max(fgsm_results, key=lambda x: x[1])
    best_pgd = max(pgd_results, key=lambda x: x[1])
    best_cw = max(cw_results, key=lambda x: x[1])
    best_df = max(df_results, key=lambda x: x[1])
    
    attack_names = ['FGSM', 'PGD', 'CW', 'DeepFool']
    best_accuracies = [best_fgsm[1], best_pgd[1], best_cw[1], best_df[1]]

    plt.figure(figsize=(10, 6))
    plt.bar(attack_names, best_accuracies)
    plt.title(f'Best Accuracy by Attack Method - {dataset_name}')
    plt.xlabel('Attack Method')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    for i, acc in enumerate(best_accuracies):
        plt.text(i, acc + 1, f'{acc:.1f}%', ha='center')
    
    summary_path = f"figures/{dataset_name}_attack_summary.png"
    plt.savefig(summary_path)
    plt.close()
    
    return {
        'fgsm': fgsm_results,
        'pgd': pgd_results,
        'cw': cw_results,
        'deepfool': df_results
    }

###############################################
# Part 3: Main Execution with Command-line Arguments
###############################################

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Image Classification and Adversarial Attack Analysis')
    parser.add_argument('--no-train', action='store_true', help='Skip model training and load from checkpoints')
    parser.add_argument('--no-eval', action='store_true', help='Skip model evaluation')
    parser.add_argument('--eval-only', action='store_true', help='Only evaluate models without adversarial testing')
    parser.add_argument('--mnist-model', type=str, default='models/mnist_model.pth', help='Path to MNIST model checkpoint')
    parser.add_argument('--fashion-model', type=str, default='models/fashion_mnist_model.pth', help='Path to Fashion-MNIST model checkpoint')
    parser.add_argument('--cifar-model', type=str, default='models/cifar_model.pth', help='Path to CIFAR model checkpoint')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training and evaluation')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for training')
    
    args = parser.parse_args()
    
    print("Starting Image Classification and Adversarial Attack Analysis")
    print(f"Using device: {device}")
    
    # Dictionary to store model evaluation results
    results = {}
    # 1. MNIST Dataset and Model
    print("\n==== MNIST Classification ====")
    train_loader, test_loader = load_mnist(batch_size=args.batch_size)
    mnist_model = SimpleCNN().to(device)
    
    if args.no_train:
        try:
            mnist_model = load_model(mnist_model, args.mnist_model)
        except FileNotFoundError:
            print(f"Model file {args.mnist_model} not found. Training model...")
            mnist_model = train_model(mnist_model, train_loader, 'mnist', epochs=args.epochs)
    else:
        print("Training MNIST model...")
        mnist_model = train_model(mnist_model, train_loader, 'mnist', epochs=args.epochs)
    
    if not args.no_eval:
        print("\nEvaluating MNIST model...")
        mnist_metrics = evaluate_model(mnist_model, test_loader, 'mnist')
        results['mnist'] = mnist_metrics
    
    # 2. Fashion-MNIST Dataset and Model
    print("\n==== Fashion-MNIST Classification ====")
    fashion_train_loader, fashion_test_loader = load_fashion_mnist(batch_size=args.batch_size)
    fashion_model = FashionCNN().to(device)
    
    if args.no_train:
        try:
            fashion_model = load_model(fashion_model, args.fashion_model)
        except FileNotFoundError:
            print(f"Model file {args.fashion_model} not found. Training model...")
            fashion_model = train_model(fashion_model, fashion_train_loader, 'fashion_mnist', epochs=args.epochs)
    else:
        print("Training Fashion-MNIST model...")
        fashion_model = train_model(fashion_model, fashion_train_loader, 'fashion_mnist', epochs=args.epochs)
    
    if not args.no_eval:
        print("\nEvaluating Fashion-MNIST model...")
        fashion_metrics = evaluate_model(fashion_model, fashion_test_loader, 'fashion_mnist')
        results['fashion_mnist'] = fashion_metrics
        
    # 3. CIFAR-10 Dataset and Model (using pre-trained ResNet)
    print("\n==== CIFAR-10 Classification ====")
    cifar_train_loader, cifar_test_loader = load_cifar(batch_size=args.batch_size)
    
    # Load pre-trained ResNet model
    print("Setting up ResNet model...")
    resnet_model = models.resnet18(pretrained=True)
    
    # Modify the final layer for CIFAR-10 (10 classes)
    resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 10)
    resnet_model = resnet_model.to(device)
    
    if args.no_train:
        try:
            resnet_model = load_model(resnet_model, args.cifar_model)
        except FileNotFoundError:
            print(f"Model file {args.cifar_model} not found. Training model...")
            resnet_model = train_model(resnet_model, cifar_train_loader, 'cifar', epochs=args.epochs, lr=0.0001)
    else:
        print("Fine-tuning ResNet model...")
        resnet_model = train_model(resnet_model, cifar_train_loader, 'cifar', epochs=args.epochs, lr=0.0001)
    
    if not args.no_eval:
        print("\nEvaluating ResNet model...")
        resnet_metrics = evaluate_model(resnet_model, cifar_test_loader, 'cifar')
        results['cifar'] = resnet_metrics
    
    # Skip adversarial testing if only evaluation is requested
    if args.eval_only:
        print("\nSkipping adversarial testing as --eval-only flag is set")
        return
    
    # 4. Test adversarial attacks on MNIST model
    print("\n==== Testing Adversarial Attacks on MNIST Model ====")
    mnist_attack_results = compare_attacks(mnist_model, test_loader, 'mnist')
    
    # 5. Test adversarial attacks on Fashion-MNIST model
    print("\n==== Testing Adversarial Attacks on Fashion-MNIST Model ====")
    fashion_attack_results = compare_attacks(fashion_model, fashion_test_loader, 'fashion_mnist')
    
    # 6. Test adversarial attacks on ResNet model for CIFAR-10
    print("\n==== Testing Adversarial Attacks on ResNet Model (CIFAR-10) ====")
    resnet_attack_results = compare_attacks(resnet_model, cifar_test_loader, 'cifar')
    
    # 7. Compare robustness across models and attacks
    print("\n==== Cross-Model Robustness Comparison ====")
    
    # Get best accuracy for each attack type across models
    attacks = ['fgsm', 'pgd', 'cw', 'deepfool']
    model_names = ['MNIST', 'Fashion-MNIST', 'CIFAR-10']
    
    # Extract best results for each attack type
    mnist_best = [max([acc for _, acc in mnist_attack_results[attack]]) for attack in attacks]
    fashion_best = [max([acc for _, acc in fashion_attack_results[attack]]) for attack in attacks]
    resnet_best = [max([acc for _, acc in resnet_attack_results[attack]]) for attack in attacks]
    
    # Plot cross-model comparison
    plt.figure(figsize=(12, 8))
    x = np.arange(len(attacks))
    width = 0.25
    
    plt.bar(x - width, mnist_best, width, label='MNIST')
    plt.bar(x, fashion_best, width, label='Fashion-MNIST')
    plt.bar(x + width, resnet_best, width, label='CIFAR-10 (ResNet)')
    
    plt.xlabel('Attack Type')
    plt.ylabel('Best Accuracy (%)')
    plt.title('Cross-Model Robustness Comparison')
    plt.xticks(x, ['FGSM', 'PGD', 'CW', 'DeepFool'])
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    cross_model_path = f"figures/cross_model_comparison.png"
    plt.savefig(cross_model_path)
    plt.close()
    
    # Create a summary table
    print("\nRobustness Summary (Best Accuracy % by Attack Type):")
    print(f"{'Model':<15} {'FGSM':<10} {'PGD':<10} {'CW':<10} {'DeepFool':<10}")
    print(f"{'-'*55}")
    print(f"{'MNIST':<15} {mnist_best[0]:<10.2f} {mnist_best[1]:<10.2f} {mnist_best[2]:<10.2f} {mnist_best[3]:<10.2f}")
    print(f"{'Fashion-MNIST':<15} {fashion_best[0]:<10.2f} {fashion_best[1]:<10.2f} {fashion_best[2]:<10.2f} {fashion_best[3]:<10.2f}")
    print(f"{'CIFAR-10':<15} {resnet_best[0]:<10.2f} {resnet_best[1]:<10.2f} {resnet_best[2]:<10.2f} {resnet_best[3]:<10.2f}")
   
if __name__ == "__main__":
    main()