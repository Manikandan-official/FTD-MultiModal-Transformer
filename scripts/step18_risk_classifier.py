import torch
import torch.nn as nn

# ============================================
# DEVICE
# ============================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\nUsing device:", device)

# ============================================
# RISK CLASSIFIER
# ============================================

class RiskClassifier(nn.Module):

    def __init__(
        self,
        input_dim=512,
        hidden_dim=256,
        num_classes=2
    ):
        super().__init__()

        self.classifier = nn.Sequential(

            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(hidden_dim, 64),
            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):

        # x shape:
        # [batch, timepoints, 512]

        # take final timepoint
        x = x[:, -1, :]

        out = self.classifier(x)

        return out

# ============================================
# DUMMY TRANSFORMER OUTPUT
# ============================================

batch_size = 2
timepoints = 4
embedding_dim = 512

dummy_transformer_output = torch.randn(
    batch_size,
    timepoints,
    embedding_dim
).to(device)

# ============================================
# MODEL
# ============================================

model = RiskClassifier().to(device)

# ============================================
# TEST PIPELINE
# ============================================

print("\nTesting Risk Classifier...\n")

output = model(dummy_transformer_output)

print("Input transformer shape:")
print(dummy_transformer_output.shape)

print("\nClassifier output shape:")
print(output.shape)

print("\nDONE")

