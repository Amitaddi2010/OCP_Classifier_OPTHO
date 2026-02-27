import os
import glob

for f in glob.glob('models/*.h5') + glob.glob('models/*.keras'):
    os.remove(f)
    print(f"Deleted {f}")

print("\nRun: python train_models.py")
