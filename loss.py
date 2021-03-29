from packaging import version
import torch
from torch import nn

class PatchNCELoss(nn.Module):
    def __init__(self, batch, temp=0.07):
        super().__init__()
        self.batch_size = batch
        self.temp = temp
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k):
        batchSize = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(feat_q.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1))
        l_pos = l_pos.view(batchSize, 1)

        # neg logit

        # Should the negatives from the other samples of a minibatch be utilized?
        # In CUT and FastCUT, we found that it's best to only include negatives
        # from the same image. Therefore, we set
        # --nce_includes_all_negatives_from_minibatch as False
        # However, for single-image translation, the minibatch consists of
        # crops from the "same" high-resolution image.
        # Therefore, we will include the negatives from the entire minibatch.
        batch_dim_for_bmm = self.batch_size

        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.temp

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))

        return loss


class JacobianClampingLoss(nn.Module):
    """Module for adding Jacobian Clamping loss.
    See Also:
        https://arxiv.org/abs/1802.08768v2
    """
    def __init__(self, lambda_min=1., lambda_max=20.):
        super(JacobianClampingLoss, self).__init__()
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

    def forward(self, gz, gz_prime, z, z_prime):
        q = (gz - gz_prime).norm() / (z - z_prime).norm()
        l_max = (q.clamp(self.lambda_max, float('inf')) - self.lambda_max) ** 2
        l_min = (q.clamp(float('-inf'), self.lambda_min) - self.lambda_min) ** 2
        l_jc = l_max + l_min

        return l_jc