# from single_video_processing import process_video
import torch.nn as nn

class VisualProjector(nn.Module):
    def __init__(self, clip_dim=768, hidden_dim=1024, llm_hidden_dim=1536):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(clip_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, llm_hidden_dim)
        )

    def forward(self, x):
        return self.mlp(x.to(self.mlp[0].weight.dtype))
    
# device=torch.device("cpu")
# projector = VisualProjector(
#     clip_dim=768,
#     hidden_dim=1024,
#     llm_hidden_dim=1536,
# ).to(device)

# projected_tokens = projector(process_video())
# print(projected_tokens.shape)