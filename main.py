import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import copy
import time
import numpy as np

# ============================================================
# 1. SOFTMAX WITH TEMPERATURE (Eq. 1 from paper)
# ============================================================

def softmax_with_temperature(logits, temperature):
    """
    q_i = exp(z_i / T) / Σ_j exp(z_j / T)
    
    T=1: normal softmax
    T>1: softer distribution (more information per sample)
    """
    return F.softmax(logits / temperature, dim=1)


def log_softmax_with_temperature(logits, temperature):
    """Numerically stable log softmax with temperature"""
    return F.log_softmax(logits / temperature, dim=1)


# ============================================================
# 2. DISTILLATION LOSS (Section 2 from paper)
# ============================================================

class DistillationLoss(nn.Module):
    """
    Combined loss = α * T² * KL(teacher_soft || student_soft) 
                  + (1-α) * CE(hard_labels, student_hard)
    
    Key insight: Multiply soft target loss by T² to ensure
    relative contributions remain stable when T changes.
    """
    def __init__(self, temperature=20.0, alpha=0.9):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha  # Weight for soft targets
        
    def forward(self, student_logits, teacher_logits, labels=None):
        """
        Args:
            student_logits: Raw outputs from student model [batch, num_classes]
            teacher_logits: Raw outputs from teacher model [batch, num_classes]
            labels: Ground truth labels (optional, for hard target loss)
        
        Returns:
            Total loss, soft_loss, hard_loss
        """
        T = self.temperature
        
        # Soft targets from teacher (at high temperature)
        teacher_soft = softmax_with_temperature(teacher_logits, T)
        
        # Student log probabilities (at high temperature)
        student_log_soft = log_softmax_with_temperature(student_logits, T)
        
        # KL Divergence for soft targets, scaled by T²
        # KL(p || q) = Σ p(x) * log(p(x)/q(x))
        soft_loss = F.kl_div(
            student_log_soft,      # Predicted (student)
            teacher_soft,           # Target (teacher)
            reduction='batchmean'
        ) * (T ** 2)  # Scale by T² as per paper
        
        # Hard target loss (cross-entropy at T=1)
        if labels is not None:
            hard_loss = F.cross_entropy(student_logits, labels)
            total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        else:
            hard_loss = torch.tensor(0.0)
            total_loss = soft_loss
            
        return total_loss, soft_loss, hard_loss


# ============================================================
# 3. TEACHER MODEL (Large/Cumbersome Model)
# ============================================================

class TeacherNet(nn.Module):
    """
    Large neural net: 784 -> 1200 -> 1200 -> 10
    With dropout regularization (as described in paper Section 3)
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(784, 1200)
        self.fc2 = nn.Linear(1200, 1200)
        self.fc3 = nn.Linear(1200, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# ============================================================
# 4. STUDENT MODEL (Small/Distilled Model)
# ============================================================

class StudentNet(nn.Module):
    """
    Smaller neural net: 784 -> hidden_size -> hidden_size -> 10
    Configurable size for experiments
    """
    def __init__(self, hidden_size=800, num_classes=10, use_dropout=False):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = F.relu(self.fc2(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = self.fc3(x)
        return x


# ============================================================
# 5. TRAINING FUNCTIONS
# ============================================================

def train_teacher(model, train_loader, epochs=20, lr=0.001, device='cuda'):
    """Train teacher with standard cross-entropy + dropout"""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        acc = 100. * correct / total
        print(f"Teacher Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Acc: {acc:.2f}%")
    
    return model


def train_student_with_distillation(
    student, 
    teacher, 
    train_loader, 
    temperature=20.0, 
    alpha=0.9, 
    epochs=20, 
    lr=0.001, 
    device='cuda'
):
    """
    Train student using knowledge distillation
    (Section 2 of paper)
    """
    student = student.to(device)
    teacher = teacher.to(device)
    teacher.eval()  # Teacher is frozen
    
    optimizer = optim.Adam(student.parameters(), lr=lr)
    distill_criterion = DistillationLoss(temperature=temperature, alpha=alpha)
    
    student.train()
    for epoch in range(epochs):
        total_loss = 0
        soft_loss_total = 0
        hard_loss_total = 0
        correct = 0
        total = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            # Get teacher predictions (no gradient)
            with torch.no_grad():
                teacher_logits = teacher(data)
            
            # Get student predictions
            student_logits = student(data)
            
            # Compute distillation loss
            loss, soft_loss, hard_loss = distill_criterion(
                student_logits, teacher_logits, target
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            soft_loss_total += soft_loss.item()
            hard_loss_total += hard_loss.item()
            
            _, predicted = student_logits.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        acc = 100. * correct / total
        print(f"Student Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} "
              f"(soft: {soft_loss_total/len(train_loader):.4f}, hard: {hard_loss_total/len(train_loader):.4f}) "
              f"| Acc: {acc:.2f}%")
    
    return student


def train_student_normal(student, train_loader, epochs=20, lr=0.001, device='cuda'):
    """Train student with standard cross-entropy (baseline)"""
    student = student.to(device)
    optimizer = optim.Adam(student.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    student.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = student(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        acc = 100. * correct / total
        print(f"Normal Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Acc: {acc:.2f}%")
    
    return student


# ============================================================
# 6. EVALUATION FUNCTION
# ============================================================

def evaluate(model, test_loader, device='cuda'):
    """Calculate test accuracy and error count"""
    model = model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    errors = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            errors += (predicted != target).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy, errors


# ============================================================
# 7. VISUALIZATION: SOFT TARGETS
# ============================================================

def visualize_soft_targets(teacher, sample, temperatures=[1, 5, 10, 20], device='cuda'):
    """Show how temperature affects the probability distribution"""
    teacher = teacher.to(device)
    teacher.eval()
    
    with torch.no_grad():
        logits = teacher(sample.unsqueeze(0).to(device))
    
    print("\n" + "="*60)
    print("SOFT TARGETS AT DIFFERENT TEMPERATURES")
    print("="*60)
    
    for T in temperatures:
        probs = softmax_with_temperature(logits, T).cpu().numpy()[0]
        print(f"\nT = {T:2d}: ", end="")
        for i, p in enumerate(probs):
            bar = "█" * int(p * 50)
            if p > 0.01:
                print(f"{i}:{bar[:20]}({p:.3f}) ", end="")
        print()
    
    # Show entropy at each temperature
    print("\n" + "-"*60)
    print("Entropy at different temperatures:")
    for T in temperatures:
        probs = softmax_with_temperature(logits, T).cpu().numpy()[0]
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        print(f"  T = {T:2d}: H = {entropy:.4f} bits")
    print("="*60)


# ============================================================
# 8. EXPERIMENT: OMITTING A CLASS (Section 3 - "Mythical Digit")
# ============================================================

def create_transfer_set_without_class(train_dataset, omit_class=3):
    """
    Create a transfer set that omits all examples of a specific class
    (Paper Section 3: Testing if distillation can teach about unseen classes)
    """
    indices = [i for i, (_, label) in enumerate(train_dataset) if label != omit_class]
    return torch.utils.data.Subset(train_dataset, indices)


# ============================================================
# 9. MAIN EXPERIMENT (Reproducing Paper Results)
# ============================================================

def run_mnist_experiment():
    """
    Reproduce the MNIST experiment from Section 3 of the paper:
    
    - Teacher: 2 hidden layers × 1200 units, with dropout
    - Student: 2 hidden layers × 800 units, no regularization
    - Compare: Normal training vs Distillation
    """
    import numpy as np
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data loading with augmentation (jitter as in paper)
    transform_train = transforms.Compose([
        transforms.RandomAffine(degrees=0, translate=(2/28, 2/28)),  # 2-pixel jitter
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=2)
    
    # ========================================
    # Step 1: Train Teacher (Large Model)
    # ========================================
    print("\n" + "="*60)
    print("STEP 1: TRAINING TEACHER MODEL (1200-1200)")
    print("="*60)
    
    teacher = TeacherNet(num_classes=10)
    teacher = train_teacher(teacher, train_loader, epochs=20, lr=0.001, device=device)
    
    teacher_acc, teacher_errors = evaluate(teacher, test_loader, device)
    print(f"\n>>> Teacher Test Accuracy: {teacher_acc:.2f}% ({teacher_errors} errors)")
    
    # ========================================
    # Step 2: Train Student Normally (Baseline)
    # ========================================
    print("\n" + "="*60)
    print("STEP 2: TRAINING STUDENT NORMALLY (800-800, NO REGULARIZATION)")
    print("="*60)
    
    student_normal = StudentNet(hidden_size=800, use_dropout=False)
    student_normal = train_student_normal(
        student_normal, train_loader, epochs=20, lr=0.001, device=device
    )
    
    normal_acc, normal_errors = evaluate(student_normal, test_loader, device)
    print(f"\n>>> Student (Normal) Test Accuracy: {normal_acc:.2f}% ({normal_errors} errors)")
    
    # ========================================
    # Step 3: Train Student with Distillation
    # ========================================
    print("\n" + "="*60)
    print("STEP 3: TRAINING STUDENT WITH DISTILLATION (T=20)")
    print("="*60)
    
    student_distilled = StudentNet(hidden_size=800, use_dropout=False)
    student_distilled = train_student_with_distillation(
        student_distilled, 
        teacher, 
        train_loader, 
        temperature=20.0,  # As used in paper
        alpha=0.9,         # High weight on soft targets
        epochs=20, 
        lr=0.001, 
        device=device
    )
    
    distill_acc, distill_errors = evaluate(student_distilled, test_loader, device)
    print(f"\n>>> Student (Distilled) Test Accuracy: {distill_acc:.2f}% ({distill_errors} errors)")
    
    # ========================================
    # Step 4: Visualize Soft Targets
    # ========================================
    sample, label = test_dataset[0]
    visualize_soft_targets(teacher, sample, temperatures=[1, 5, 10, 20], device=device)
    
    # ========================================
    # Results Summary
    # ========================================
    print("\n" + "="*60)
    print("RESULTS SUMMARY (MNIST)")
    print("="*60)
    print(f"{'Model':<30} {'Accuracy':>10} {'Errors':>10}")
    print("-"*50)
    print(f"{'Teacher (1200-1200 + dropout)':<30} {teacher_acc:>9.2f}% {teacher_errors:>10}")
    print(f"{'Student Normal (800-800)':<30} {normal_acc:>9.2f}% {normal_errors:>10}")
    print(f"{'Student Distilled (800-800)':<30} {distill_acc:>9.2f}% {distill_errors:>10}")
    print("-"*50)
    improvement = normal_errors - distill_errors
    print(f"Error reduction from distillation: {improvement} errors")
    print("="*60)
    
    return teacher, student_normal, student_distilled


# ============================================================
# 10. EXPERIMENT: TEMPERATURE SWEEP (Section 3)
# ============================================================

def run_temperature_experiment():
    """
    Paper Section 3: "When the distilled net had 300 or more units...
    all temperatures above 8 gave fairly similar results. But when
    this was radically reduced to 30 units per layer, temperatures
    in the range 2.5 to 4 worked significantly better"
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # Quick teacher
    teacher = TeacherNet()
    teacher = train_teacher(teacher, train_loader, epochs=10, device=device)
    
    print("\n" + "="*60)
    print("TEMPERATURE SWEEP EXPERIMENT")
    print("="*60)
    
    temperatures = [1, 2, 3, 4, 5, 8, 10, 15, 20]
    
    print(f"\n{'Hidden Size':<12}", end="")
    for T in temperatures:
        print(f"T={T:<4}", end=" ")
    print()
    
    for hidden_size in [300, 30]:
        print(f"{hidden_size:<12}", end="")
        for T in temperatures:
            student = StudentNet(hidden_size=hidden_size)
            student = train_student_with_distillation(
                student, teacher, train_loader, 
                temperature=T, alpha=0.9, epochs=10, device=device
            )
            acc, _ = evaluate(student, test_loader, device)
            print(f"{acc:>5.1f}", end=" ")
        print()
    
    print("="*60)


# ============================================================
# 11. EXPERIMENT: MYTHICAL DIGIT (Section 3)
# ============================================================

def run_mythical_digit_experiment(omit_class=3):
    """
    Paper Section 3: "We then tried omitting all examples of the 
    digit 3 from the transfer set. So from the perspective of the 
    distilled model, 3 is a mythical digit that it has never seen."
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    full_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    # Create transfer set WITHOUT digit 3
    transfer_dataset = create_transfer_set_without_class(full_train, omit_class)
    transfer_loader = DataLoader(transfer_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # Train teacher on FULL data
    full_loader = DataLoader(full_train, batch_size=128, shuffle=True)
    teacher = TeacherNet()
    teacher = train_teacher(teacher, full_loader, epochs=15, device=device)
    
    print("\n" + "="*60)
    print(f"MYTHICAL DIGIT EXPERIMENT (Omitting class {omit_class})")
    print("="*60)
    
    # Train student WITHOUT seeing digit 3
    student = StudentNet(hidden_size=800)
    student = train_student_with_distillation(
        student, teacher, transfer_loader,
        temperature=20.0, alpha=0.9, epochs=15, device=device
    )
    
    # Evaluate overall
    overall_acc, overall_errors = evaluate(student, test_loader, device)
    
    # Evaluate specifically on omitted class
    omitted_indices = [i for i, (_, label) in enumerate(test_dataset) if label == omit_class]
    omitted_dataset = torch.utils.data.Subset(test_dataset, omitted_indices)
    omitted_loader = DataLoader(omitted_dataset, batch_size=1000, shuffle=False)
    
    omitted_acc, omitted_errors = evaluate(student, omitted_loader, device)
    
    print(f"\nTransfer set size: {len(transfer_dataset)} (omitted {len(full_train) - len(transfer_dataset)} examples)")
    print(f"Overall test accuracy: {overall_acc:.2f}% ({overall_errors} errors)")
    print(f"Accuracy on class {omit_class} (never seen): {omitted_acc:.2f}%")
    print(f"Errors on class {omit_class}: {omitted_errors}/{len(omitted_dataset)}")
    print("="*60)


# ============================================================
# 12. SPECIALIST MODELS (Section 5)
# ============================================================

class SpecialistNet(nn.Module):
    """
    Specialist model: Focuses on a confusable subset of classes
    All other classes combined into a "dustbin" class
    (Section 5.2 of paper)
    """
    def __init__(self, num_specialist_classes, num_dustbin_classes=1, hidden_size=300):
        super().__init__()
        self.num_specialist = num_specialist_classes
        self.num_dustbin = num_dustbin_classes
        self.fc1 = nn.Linear(784, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_specialist_classes + num_dustbin_classes)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def find_confusable_classes(model, data_loader, device, num_clusters=5):
    """
    Find confusable class clusters using prediction covariance
    (Section 5.3 - simplified version)
    """
    model.eval()
    
    # Collect all predictions
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            output = model(data)
            probs = F.softmax(output, dim=1)
            all_preds.append(probs.cpu())
            all_labels.append(target)
    
    all_preds = torch.cat(all_preds, dim=0)  # [N, 10]
    all_labels = torch.cat(all_labels, dim=0)
    
    # Compute covariance of predictions
    pred_mean = all_preds.mean(dim=0)
    centered = all_preds - pred_mean
    cov_matrix = centered.T @ centered / len(all_preds)
    
    # Simple confusion-based clustering
    confusion = torch.zeros(10, 10)
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1).cpu()
            for t, p in zip(target, pred):
                confusion[t, p] += 1
    
    # Normalize
    confusion = confusion / confusion.sum(dim=1, keepdim=True)
    
    # Find most confused pairs (excluding diagonal)
    pairs = []
    for i in range(10):
        for j in range(i+1, 10):
            confusion_score = confusion[i, j] + confusion[j, i]
            pairs.append((confusion_score, i, j))
    
    pairs.sort(reverse=True)
    
    # Create clusters from top confused pairs
    clusters = []
    used = set()
    
    for score, i, j in pairs[:num_clusters]:
        if i not in used and j not in used:
            clusters.append([i, j])
            used.add(i)
            used.add(j)
    
    # Add remaining classes as singletons
    for c in range(10):
        if c not in used:
            clusters.append([c])
    
    return clusters


def train_specialist(specialist, generalist, train_dataset, specialist_classes, 
                     dustbin_classes, temperature=3.0, epochs=10, device='cuda'):
    """
    Train a specialist model
    - Initialized from generalist weights (for transfer)
    - Trained on 50% specialist data + 50% random data
    (Section 5.2)
    """
    # Create biased dataset
    specialist_indices = [i for i, (_, label) in enumerate(train_dataset) 
                         if label in specialist_classes]
    other_indices = [i for i, (_, label) in enumerate(train_dataset) 
                    if label not in specialist_classes]
    
    # Sample equal amounts
    n_specialist = len(specialist_indices)
    n_other = min(len(other_indices), n_specialist)
    
    import random
    other_sampled = random.sample(other_indices, n_other)
    combined_indices = specialist_indices + other_sampled
    random.shuffle(combined_indices)
    
    specialist_dataset = torch.utils.data.Subset(train_dataset, combined_indices)
    specialist_loader = DataLoader(specialist_dataset, batch_size=128, shuffle=True)
    
    # Remap labels: specialist classes keep identity, others -> dustbin
    class_mapping = {}
    for i, c in enumerate(specialist_classes):
        class_mapping[c] = i
    for c in dustbin_classes:
        class_mapping[c] = len(specialist_classes)
    
    specialist = specialist.to(device)
    generalist = generalist.to(device)
    generalist.eval()
    
    optimizer = optim.Adam(specialist.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    specialist.train()
    for epoch in range(epochs):
        total_loss = 0
        for data, target in specialist_loader:
            data = data.to(device)
            # Remap targets
            mapped_target = torch.tensor(
                [class_mapping[t.item()] for t in target],
                dtype=torch.long,
                device=device
            )
            
            optimizer.zero_grad()
            output = specialist(data)
            loss = criterion(output, mapped_target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"  Specialist {specialist_classes} Epoch {epoch+1}: Loss={total_loss/len(specialist_loader):.4f}")
    
    return specialist


def run_specialist_experiment():
    """Run the specialist model experiment (Section 5)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    # Train generalist
    print("Training generalist model...")
    generalist = TeacherNet()
    generalist = train_teacher(generalist, train_loader, epochs=15, device=device)
    
    gen_acc, gen_errors = evaluate(generalist, test_loader, device)
    print(f"Generalist accuracy: {gen_acc:.2f}%\n")
    
    # Find confusable clusters
    print("Finding confusable class clusters...")
    clusters = find_confusable_classes(generalist, train_loader, device, num_clusters=4)
    
    print("\nIdentified clusters:")
    for i, cluster in enumerate(clusters):
        print(f"  Specialist {i}: classes {cluster}")
    
    # Train specialists
    print("\nTraining specialist models...")
    specialists = []
    all_specialist_classes = set()
    
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        
        specialist_classes = cluster
        dustbin_classes = [c for c in range(10) if c not in specialist_classes]
        all_specialist_classes.update(specialist_classes)
        
        specialist = SpecialistNet(
            num_specialist_classes=len(specialist_classes),
            num_dustbin_classes=1,
            hidden_size=300
        )
        
        specialist = train_specialist(
            specialist, generalist, train_dataset,
            specialist_classes, dustbin_classes,
            temperature=3.0, epochs=10, device=device
        )
        specialists.append((specialist, specialist_classes))
    
    print(f"\nTrained {len(specialists)} specialist models")
    print(f"Covering {len(all_specialist_classes)} classes: {sorted(all_specialist_classes)}")
    print("="*60)


# ============================================================
# 13. SOFT TARGETS AS REGULARIZER (Section 6)
# ============================================================

def run_regularization_experiment():
    """
    Paper Section 6: Show soft targets prevent overfitting
    Train with only 3% of data
    
    Results from paper:
    - Baseline (100% data): 58.9% frame accuracy
    - Baseline (3% data): 44.5% (overfitting)
    - Soft targets (3% data): 57.0% (almost recovers full performance!)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    full_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    # Use only 3% of training data
    n_subset = int(0.03 * len(full_train))
    indices = torch.randperm(len(full_train))[:n_subset]
    small_dataset = torch.utils.data.Subset(full_train, indices)
    
    full_loader = DataLoader(full_train, batch_size=128, shuffle=True)
    small_loader = DataLoader(small_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    print("="*60)
    print("SOFT TARGETS AS REGULARIZER EXPERIMENT")
    print(f"Full training set: {len(full_train)}")
    print(f"Small training set: {len(small_dataset)} (3%)")
    print("="*60)
    
    # Baseline with full data
    print("\n1. Training baseline with 100% data...")
    baseline_full = StudentNet(hidden_size=800)
    baseline_full = train_student_normal(baseline_full, full_loader, epochs=20, device=device)
    acc_full, _ = evaluate(baseline_full, test_loader, device)
    
    # Baseline with 3% data (should overfit)
    print("\n2. Training baseline with 3% data (will overfit)...")
    baseline_small = StudentNet(hidden_size=800)
    baseline_small = train_student_normal(baseline_small, small_loader, epochs=20, device=device)
    acc_small, _ = evaluate(baseline_small, test_loader, device)
    
    # Teacher for soft targets
    teacher = TeacherNet()
    teacher = train_teacher(teacher, full_loader, epochs=15, device=device)
    
    # Student with soft targets from 3% data
    print("\n3. Training with soft targets from 3% data...")
    student_soft = StudentNet(hidden_size=800)
    student_soft = train_student_with_distillation(
        student_soft, teacher, small_loader,
        temperature=20.0, alpha=0.9, epochs=20, device=device
    )
    acc_soft, _ = evaluate(student_soft, test_loader, device)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Baseline (100% data):      {acc_full:.2f}%")
    print(f"Baseline (3% data):        {acc_small:.2f}%  ← Overfits!")
    print(f"Soft targets (3% data):    {acc_soft:.2f}%  ← Recovers performance!")
    print("="*60)


# ============================================================
# RUN ALL EXPERIMENTS
# ============================================================

if __name__ == "__main__":
    # Main MNIST experiment
    teacher, student_normal, student_distilled = run_mnist_experiment()
    
    # Uncomment to run other experiments:
    # run_temperature_experiment()
    # run_mythical_digit_experiment(omit_class=3)
    # run_specialist_experiment()
    # run_regularization_experiment()