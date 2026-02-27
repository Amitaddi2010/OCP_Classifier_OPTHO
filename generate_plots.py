import matplotlib.pyplot as plt
import numpy as np

# Results data
models = ['MobileNetV2', 'VGG16', 'CustomCNN', 'ResNet50']
accuracies = [98.25, 78.95, 70.18, 42.11]

# Plot 1: Model Comparison
plt.figure(figsize=(10, 6))
bars = plt.bar(models, accuracies, color=['#2ecc71', '#3498db', '#f39c12', '#e74c3c'])
plt.ylabel('Test Accuracy (%)', fontsize=12)
plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
plt.ylim(0, 100)
for i, v in enumerate(accuracies):
    plt.text(i, v + 2, f'{v}%', ha='center', fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: model_comparison.png")

# Plot 2: Dataset Distribution
plt.figure(figsize=(8, 6))
classes = ['A. OCP', 'C. Normal Eyes']
counts = [148, 137]
colors = ['#e74c3c', '#2ecc71']
plt.pie(counts, labels=classes, autopct='%1.1f%%', colors=colors, startangle=90)
plt.title('Dataset Distribution', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/dataset_distribution.png', dpi=300, bbox_inches='tight')
print("Saved: dataset_distribution.png")

# Plot 3: CustomCNN Metrics
plt.figure(figsize=(10, 6))
metrics = ['Precision', 'Recall', 'F1-Score']
ocp = [0.67, 0.87, 0.75]
normal = [0.78, 0.52, 0.62]

x = np.arange(len(metrics))
width = 0.35

plt.bar(x - width/2, ocp, width, label='A. OCP', color='#e74c3c')
plt.bar(x + width/2, normal, width, label='C. Normal Eyes', color='#2ecc71')

plt.ylabel('Score', fontsize=12)
plt.title('CustomCNN Performance Metrics', fontsize=14, fontweight='bold')
plt.xticks(x, metrics)
plt.ylim(0, 1)
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('results/customcnn_metrics.png', dpi=300, bbox_inches='tight')
print("Saved: customcnn_metrics.png")

print("\nAll plots saved in 'results/' folder!")
