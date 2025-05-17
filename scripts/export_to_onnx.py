import torch
import torch.nn as nn
from torchvision.models import resnet34

# ✅ Define model class with pretrained ResNet34
class FairFaceResNet34(nn.Module):
    def __init__(self, num_classes=18):
        super(FairFaceResNet34, self).__init__()
        self.base = resnet34(pretrained=False)  # ✅ Use pretrained ImageNet weights
        self.base.fc = nn.Linear(self.base.fc.in_features, num_classes)

    def forward(self, x):
        return self.base(x)

# ✅ Load model and weights
model = FairFaceResNet34()
state_dict = torch.load("fair_face_models/res34_fair_align_multi_7_20190809.pt", map_location="cpu")
model.load_state_dict(state_dict, strict=False)  # Optional: try strict=True if you're sure weights match
model.eval()

# ✅ Export to ONNX
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    "fairface_res34.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=13  # Slightly more modern opset
)

print("✅ Exported pretrained model to fairface_res34.onnx")
