# import torch
# from model import data_setup, model_builder, engine, utils
# from pathlib import Path
# from torchvision import transforms
# from torch import nn
# from timeit import default_timer as timer

# device = "cuda" if torch.cuda.is_available() else "cpu"
# # Set random seeds
# torch.manual_seed(42)
# torch.cuda.manual_seed(42)

# # Set number of epochs
# NUM_EPOCHS = 5

# # Setup path to data folder
# data_path = Path("model/data/")
# image_path = data_path / "Tobacco3482_100"

# # Setup train and testing paths
# train_dir = image_path / "Train"
# test_dir = image_path / "Test"

# transform = transforms.Compose(
#     [
#         transforms.Resize((64, 64)),
#         transforms.ToTensor(),
#     ]
# )
# batch_size = 32


# # # Create train/test dataloader and get class names as a list
# train_dataloader, test_dataloader, class_names = data_setup.create_dataloader(
#     train_dir, test_dir, transform, batch_size
# )

# # Check out single image size/shape
# img, label = next(iter(train_dataloader))


# model = model_builder.TinyVGG(
#     input_shape=3, hidden_units=10, output_shape=len(class_names)
# ).to(device)


# # Setup loss function and optimizer
# loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

# # Start the timer
# start_time = timer()
# model_results = engine.train(
#     model, train_dataloader, test_dataloader, optimizer, loss_fn, NUM_EPOCHS, device
# )

# # End the timer and print out how long it took
# end_time = timer()
# print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

# target_dir = "/Users/shreyachauhan/Thesis_Document_Image_Classification/model/models"
# utils.save_model(model=model, target_dir=target_dir, model_name="tinyvgg_model.pth")


# For this notebook to run with updated APIs, we need torch 1.12+ and torchvision 0.13+

# For this notebook to run with updated APIs, we need torch 1.12+ and torchvision 0.13+
# try:
#     import torch
#     import torchvision

#     assert int(torch.__version__.split(".")[1]) >= 12, "torch version should be 1.12+"
#     assert (
#         int(torchvision.__version__.split(".")[1]) >= 13
#     ), "torchvision version should be 0.13+"
#     print(f"torch version: {torch.__version__}")
#     print(f"torchvision version: {torchvision.__version__}")
# except:
#     print(
#         f"[INFO] torch/torchvision versions not as required, installing nightly versions."
#     )
#     import torch
#     import torchvision

#     print(f"torch version: {torch.__version__}")
#     print(f"torchvision version: {torchvision.__version__}")


# # Continue with regular imports
# import matplotlib.pyplot as plt
# import torch
# import torchvision
# import os
# from torch import nn
# from torchvision import transforms

# # Try to get torchinfo, install it if it doesn't work
# try:
#     from torchinfo import summary
# except:
#     print("[INFO] Couldn't find torchinfo... installing it.")
#     from torchinfo import summary

# # Try to import the going_modular directory, download it from GitHub if it doesn't work
# try:
#     from model import data_setup, engine
# except:
#     # Get the going_modular scripts
#     print("[INFO] Couldn't find going_modular scripts... downloading them from GitHub.")


# # Setup device agnostic code
# device = "cuda" if torch.cuda.is_available() else "cpu"


# # Setup path to data folder
# data_path = "/Users/shreyachauhan/Thesis_Document_Image_Classification/model/data"
# image_path = os.path.join(data_path, "Tobacco3482_100")

# # Setup train and testing paths
# train_dir = os.path.join(image_path, "Train")
# test_dir = os.path.join(image_path, "Test")

# weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
# model = torchvision.models.efficientnet_b0(weights=weights).to(device)

# # Print a summary using torchinfo (uncomment for actual output)
# summary(
#     model=model,
#     input_size=(32, 3, 224, 224),  # make sure this is "input_size", not "input_shape"
#     # col_names=["input_size"], # uncomment for smaller output
#     col_names=["input_size", "output_size", "num_params", "trainable"],
#     col_width=20,
#     row_settings=["var_names"],
# )
