# import torch

# print("Torch:", torch.__version__)
# print("CUDA available:", torch.cuda.is_available())

# if torch.cuda.is_available():
#     print("GPU:", torch.cuda.get_device_name(0))

# from turbojpeg import TurboJPEG
# t = TurboJPEG()
# print("works!")

# after install verify
from turbojpeg import TurboJPEG
t = TurboJPEG(r"C:\libjpeg-turbo-gcc64\bin\libturbojpeg.dll")
print("works!")