AgriVision Assignment 1 (Tasks 1–12) — Python code
Student: Vigneshwaran B
Register Number: 192324022

Files:
- AgriVision_Tasks1_12.py: numerical demonstrations (Tasks 2–4), capacity plot (Task 6),
  and optional image-classification CNN training/testing pipeline.
- README.txt: setup, dataset directory structure and cautions.

Install:
pip install numpy matplotlib scikit-learn torch torchvision

Run numerical demonstrations:
python AgriVision_Tasks1_12.py

Run CNN experiment after arranging a PlantVillage subset:
python AgriVision_Tasks1_12.py --data ./PlantVillage3Class --epochs 5

Expected directory:
PlantVillage3Class/
  train/Healthy/...
  train/Early_Blight/...
  train/Late_Blight/...
  val/Healthy/...
  val/Early_Blight/...
  val/Late_Blight/...
  test/Healthy/...
  test/Early_Blight/...
  test/Late_Blight/...

Important:
- Use your own valid dataset source and a stratified split.
- This script does not fabricate experimental metrics.
- Capacity plot values are explicitly illustrative, matching the written answer; replace them with measured errors before submission.
- Task 12 deployment constraints (50 ms, 5 MB) must be tested on the target phone. Desktop measurements do not prove phone compliance.
