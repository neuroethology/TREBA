import torch
import torch.nn.functional as F


def compute_label_loss(states, actions, labels, approx_rnn, approx_fc, categorical, loss_weight = 1.0):
    '''Compute attribute consistency loss'''
    assert states.size(0) == actions.size(0)        

    hiddens, _ = approx_rnn(torch.cat([states, actions], dim=-1))
    avg_hiddens = torch.mean(hiddens, dim=0)
    approx_out = approx_fc(avg_hiddens)

    assert approx_out.size() == labels.size()

    if categorical:
        approx_out = F.log_softmax(approx_out, dim=-1)
        return -torch.sum(approx_out*labels)*loss_weight
    else:
        return F.mse_loss(approx_out, labels, reduction='sum')*loss_weight


def compute_decoding_loss(representations, labels, decoding_fc, categorical, loss_weight = 1.0):
    '''Compute attribute decoding loss'''

    approx_out = decoding_fc(representations)

    assert approx_out.size() == labels.size()

    if categorical:
        approx_out = F.log_softmax(approx_out, dim=-1)*loss_weight
        return -torch.sum(approx_out*labels)
    else:
        return F.mse_loss(approx_out, labels, reduction='sum')*loss_weight


# Supervised Contrastive Loss based on:
# https://github.com/HobbitLong/SupContrast/blob/master/losses.py
def compute_contrastive_loss(posterior, aug_posterior, contrastive_fc, 
    labels=None, mask=None, contrast_mode = 'all', temperature = 0.07,
    base_temperature = 0.07, loss_weight = 1.0):
    '''Compute contrastive loss'''

    zis = contrastive_fc(posterior)
    # normalize
    zis = torch.nn.functional.normalize(zis, dim = 1)

    zjs = contrastive_fc(aug_posterior)      
    # normalize      
    zjs = torch.nn.functional.normalize(zjs, dim = 1)        

    features = torch.stack([zis, zjs], dim = 1)

    if labels is not None:
        labels = torch.argmax(labels, dim = -1)

    device = (torch.device('cuda')
              if features.is_cuda
              else torch.device('cpu'))

    features = features.view(features.shape[0], features.shape[1], -1)

    batch_size = features.shape[0]
    if labels is not None and mask is not None:
        raise ValueError('Cannot define both `labels` and `mask`')
    elif labels is None and mask is None:
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)
    elif labels is not None:
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.permute(1,0)).float().to(device)
    else:
        mask = mask.float().to(device)

    contrast_count = features.shape[1]
    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
    if contrast_mode == 'one':
        anchor_feature = features[:, 0]
        anchor_count = 1
    elif contrast_mode == 'all':
        anchor_feature = contrast_feature
        anchor_count = contrast_count
    else:
        raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

    # compute logits
    anchor_dot_contrast = torch.div(
        torch.matmul(anchor_feature, contrast_feature.permute(1,0)),
        temperature)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # tile mask
    mask = mask.repeat(anchor_count, contrast_count)
    # mask-out self-contrast cases
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
        0
    )
    mask = mask * logits_mask

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positives
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

    # loss
    loss = - (temperature / base_temperature) * mean_log_prob_pos
    loss = loss.view(anchor_count, batch_size).mean()

    return loss*loss_weight

