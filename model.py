import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    def __init__(self, temperature, contrast_mode,
                 base_temperature, device):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.device = device

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        """

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(self.device)
        else:
            mask = mask.float().to(self.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-7)  # fix the nan issue

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class backbone_network(nn.Module):
    def __init__(self, in_dim, hid_dim, dropout=0.1):
        super(backbone_network, self).__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim//2)
        self.fc2 = nn.Linear(hid_dim // 2, hid_dim // 2)
        self.fc3 = nn.Linear(hid_dim//2, hid_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        x = self.act(self.fc2(x))
        x = self.dropout(x)
        x = self.act(self.fc3(x))
        return x


class attn_backbone_network(nn.Module):
    def __init__(self, in_dim, hid_dim, dropout=0.1):
        super(attn_backbone_network, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2, bias=False),
            nn.Mish(),
            nn.Linear(in_dim // 2, in_dim, bias=False),
            nn.Sigmoid())  # 0-1
        self.fc2 = nn.Sequential(
            nn.Linear(in_dim, hid_dim // 2),
            nn.Mish(),
            nn.Dropout(p=dropout),
            nn.Linear(hid_dim // 2, hid_dim // 2),
            nn.Mish(),
            nn.Dropout(p=dropout),
            nn.Linear(hid_dim // 2, hid_dim))
        # self.ln = nn.LayerNorm(in_dim)

    def forward(self, x):
        '''
        :param x: The shape is (batch, inp_dim)
        :return: The encoded features with the shape (batch, inp_dim)
        '''
        x = x * self.fc1(x)  # attention score distribution not that stable: seed 0,1,2,3,4
        x = self.fc2(x)
        return x


class MLPBlock(nn.Module):
    def __init__(self, in_channels, num_patch, dropout=0.1):
        super(MLPBlock, self).__init__()
        self.fc1 = nn.Linear(num_patch, num_patch)
        self.fc2 = nn.Linear(in_channels, in_channels)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):  # x: (B, P, D')
        x = x.transpose(1, 2)  # x: (B, P, D') --> (B, D', P)
        x = self.fc1(x)  # TOKEN mixer: (B, D', P) --> (B, D', P)
        x = self.gelu(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)   # x: (B, D', P) --> (B, P, D')
        x = self.fc2(x)  # CHANNEL mixer: (B, P, D') --> (B, P, D')
        x = self.gelu(x)
        x = self.dropout(x)
        return x   # (B, P, D')


class MLPMixerEncoder(nn.Module):
    def __init__(self, input_dim, patch_size, hidden_channels, num_blocks, dropout=0.1):
        super(MLPMixerEncoder, self).__init__()
        self.input_dim = input_dim
        self.patch_size = patch_size
        self.patch_dim = input_dim // patch_size
        self.embed_dim = hidden_channels
        self.embed = nn.Linear(self.patch_dim, hidden_channels)
        self.mixer_blocks = nn.ModuleList([
            MLPBlock(self.embed_dim, patch_size, dropout=dropout)  # Added dropout parameter
            for _ in range(num_blocks)
        ])
        self.layer_norm = nn.LayerNorm(hidden_channels)
        self.fc_out = nn.Linear(self.embed_dim, self.embed_dim // self.patch_size)

    def forward(self, x):
        # Reshape input into patches: (B, D) --> (B, P, D/P)
        x = x.view(-1, self.patch_size, self.patch_dim)

        # Embed each patch: (B, P, D/P) --> (B, P, D')
        x = self.embed(x)

        # Apply MLP-Mixer blocks: (B, P, D') --> (B, P, D')
        for mixer_block in self.mixer_blocks:
            x = x + mixer_block(self.layer_norm(x))

        x = self.fc_out(x)  # (B, P, D') --> (B, P, D'/P)
        x = x.view(-1, self.embed_dim)  # (B, D')

        return x


class feature_learning_mlp(nn.Module):
    def __init__(self, in_dim, hid_dim, dropout=0.1):
        super(feature_learning_mlp, self).__init__()
        # self.ln = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hid_dim//2)
        self.fc2 = nn.Linear(hid_dim//2, hid_dim)
        self.act = nn.ReLU()
        # self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        # x = self.ln(x)
        x = self.act(self.fc1(x))
        # x = self.dropout(x)
        x = self.fc2(x)
        # return x.norm(p=2, dim=-1)  # [B]
        return x / x.norm(p=2, dim=-1, keepdim=True)  # TODO: try this implementation [B,D]


class downstream_mapping(nn.Module):
    def __init__(self, in_dim, hid_dim, dropout=0.1):
        super(downstream_mapping, self).__init__()
        self.layer_norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hid_dim)
        # self.gelu = nn.GELU()
        # self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hid_dim, 1)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.fc1(x)
        # x = self.gelu(x)  # Added GELU activation
        # x = self.dropout(x)  # Added dropout
        x = self.fc2(x)
        return x


class HybridLT(nn.Module):
    def __init__(self, inp_dim, hid_dim, feat_learn_dim, downstream_dim, base_tau, tau, dropout, patch, block, device):

        super(HybridLT, self).__init__()

        self.emb_dim = hid_dim
        self.feat_emb_dim = feat_learn_dim
        self.down_emb_dim = downstream_dim
        self.tau = tau
        self.base_tau = base_tau

        # self.backbone_encoder = backbone_network(inp_dim, self.emb_dim, dropout)
        # self.backbone_encoder = attn_backbone_network(inp_dim, self.emb_dim, dropout)
        self.backbone_encoder = MLPMixerEncoder(input_dim=inp_dim, patch_size=patch, hidden_channels=self.emb_dim,
                                                dropout=dropout, num_blocks=block)  # 25,3
        self.feat_encoder = feature_learning_mlp(self.emb_dim, self.feat_emb_dim, dropout)
        self.down_encoder = downstream_mapping(self.emb_dim, self.down_emb_dim, dropout=dropout)

        self.sup_loss = SupConLoss(temperature=self.tau, contrast_mode='all',
                                   base_temperature=self.base_tau, device=device)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, feat_x, feat_y, down_x, down_y, alpha):
        feat_x, down_x = self.backbone_encoder(feat_x), self.backbone_encoder(down_x)

        # Contrastive Loss
        feat_x = self.feat_encoder(feat_x).view(feat_x.shape[0], 1, -1)  # (B, 1, D)
        sup_loss = self.sup_loss(feat_x, feat_y)

        # BCE Loss
        down_y_hat = self.down_encoder(down_x).squeeze(-1)
        ce_loss = self.bce_loss(down_y_hat, down_y.type(torch.float32))

        # Overall Loss
        loss = alpha * sup_loss + (1 - alpha) * ce_loss
        # loss = 0 * sup_loss + (1 - 0) * ce_loss
        
        return loss

    def predict(self, x):
        with torch.no_grad():
            x = self.backbone_encoder(x)
            y_hat = self.down_encoder(x).sigmoid().squeeze(-1)

        return y_hat