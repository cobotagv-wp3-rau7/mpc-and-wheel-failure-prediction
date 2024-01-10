from cobot_ml.inference_utilities import StepByStepPredictor
import torch
import numpy as np

model_file_path = ".\\sample_84_features_model.pt"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_wrapper = StepByStepPredictor(model_file_path, 84, device)

output_data = model_wrapper.step(np.random.rand(1, 10, 84))
print("Finished")
