from cobot_ml.inference_utilities import StepByStepPredictor
import torch
import numpy as np

model_file_path = ".\\sample_84_features_model.pt"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
columns = [f"col_{i}" for i in range(84)]
preprocessing = lambda x: x

model_wrapper = StepByStepPredictor(model_file_path, device=device,
                                    columns=columns,
                                    preprocessing=preprocessing)

output_data = model_wrapper.step(np.random.rand(1, 10, 84))
print("Finished")
