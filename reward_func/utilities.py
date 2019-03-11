import torch
import torch.nn.functional as F

# The reward in explore stage
# a = torch.randn(100, 128)
# b = torch.randn(100, 128)
# output = F.cosine_similarity(a, b)
def cosine_similarity(a, b):
	return F.cosine_similarity(a, b)