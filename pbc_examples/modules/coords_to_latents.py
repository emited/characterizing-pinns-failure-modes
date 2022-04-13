from torch import nn

from pbc_examples.modules.features import ExpSineAndLinearFeatures


class PX(nn.Module):
    def __init__(self, emb_dim, out_features, learn_mult=False):
        super(PX, self).__init__()
        self.learn_mult = learn_mult
        self.feat = ExpSineAndLinearFeatures(out_features)
        self.lin_feat = nn.Linear(emb_dim, out_features)
        self.lin_feat_bias = nn.Linear(emb_dim, out_features)
        if self.learn_mult:
            self.lin_u0 = nn.Linear(emb_dim, out_features)

    def forward(self, x, emb):
        # x : (B, X, 1)
        # emb : (B, emb_dim)

        # x_transformed : (B, X, out_features)
        x_transformed = x * self.lin_feat(emb).unsqueeze(-2)\
                        + self.lin_feat_bias(emb).unsqueeze(-2)
        if self.learn_mult:
            m = self.lin_u0(emb).unsqueeze(-2)
        else:
            m = 1
        return self.feat(x_transformed) * m