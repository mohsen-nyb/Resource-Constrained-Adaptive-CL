#Cosine Similarity Function
import torch
from torch import nn
import torch.nn.functional as F

def cosine_similarity_matrix(z, m=1):
  z = F.normalize(z, dim=1)  # Normalize each vector to unit norm
  B = z.shape[0]  # B = 2N
  b = (B + m - 1) // m  # block size (ceil division)
  sim_matrix = torch.zeros(B, B, device=z.device)

  for i in range(m):
      i_start = i * b
      i_end = min((i + 1) * b, B)
      z_i = z[i_start:i_end]  # (b_i, d)

      for j in range(m):
          j_start = j * b
          j_end = min((j + 1) * b, B)
          z_j = z[j_start:j_end]  # (b_j, d)

          # Compute only the needed block
          sim_block = z_i @ z_j.T  # (b_i, b_j)
          sim_matrix[i_start:i_end, j_start:j_end] = sim_block

  return sim_matrix
  #return torch.matmul(z, z.T)  # Similarity matrix: (2N x 2N)


# NT-Xent loss (Normalized Temperature-scaled Cross Entropy Loss)
class NTXentLoss(nn.Module):
  def __init__(self, batch_size, temperature=0.5, m=1):
      super(NTXentLoss, self).__init__()
      self.batch_size = batch_size
      self.temperature = temperature
      self.m = m
      self.mask = self._get_correlated_mask(batch_size)
      self.criterion = nn.CrossEntropyLoss(reduction="sum")

  def _get_correlated_mask(self, batch_size):
      N = 2 * batch_size
      mask = torch.ones((N, N), dtype=bool)
      mask = mask.fill_diagonal_(0)
      for i in range(batch_size):
          mask[i, batch_size + i] = 0
          mask[batch_size + i, i] = 0
      return mask

  def forward(self, zis, zjs):
      N = 2 * self.batch_size
      z = torch.cat([zis, zjs], dim=0)  # (2N, d)
      sim = cosine_similarity_matrix(z, self.m) / self.temperature  # (2N, 2N)
      sim_i_j = torch.diag(sim, self.batch_size)
      sim_j_i = torch.diag(sim, -self.batch_size)
      positives = torch.cat([sim_i_j, sim_j_i], dim=0).unsqueeze(1)  # (2N, 1)

      sim = sim.masked_select(self.mask.to(z.device)).view(N, -1)  # remove self- and positive-pairs
      labels = torch.zeros(N).long().to(z.device)  # positives are always at position 0
      logits = torch.cat([positives, sim], dim=1)  # (2N, 1 + 2N - 2)

      loss = self.criterion(logits, labels)
      return loss / N
