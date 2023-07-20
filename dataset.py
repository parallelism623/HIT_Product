import fiftyone as fo
import fiftyone.zoo as foz

# Download and load the validation split of Open Images V7
dataset = foz.load_zoo_dataset("open-images-v7", split="validation")

session = fo.launch_app(dataset)