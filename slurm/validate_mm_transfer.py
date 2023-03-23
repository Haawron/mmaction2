import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
p_pyth = 'data/weights/timesformer/timesformer_8x32_224_howto100M_mmaction.pth'
pyth = torch.load(p_pyth, map_location=device)

