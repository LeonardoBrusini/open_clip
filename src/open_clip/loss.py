from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features

def gather_image_features(
        image_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1
):
    # We gather tensors from all gpus
    if gather_with_grad:
        all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
    else:
        gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
        dist.all_gather(gathered_image_features, image_features)
        if not local_loss:
            # ensure grads for local rank when all_* features don't have a gradient
            gathered_image_features[rank] = image_features
        all_image_features = torch.cat(gathered_image_features, dim=0)
    return all_image_features

def concat_all_gather(tensor: torch.Tensor) -> torch.Tensor:
    """
    Performs all_gather operation on the provided tensor across all processes.
    """
    world_size = dist.get_world_size()
    tensors_gather = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensors_gather, tensor)
    return torch.cat(tensors_gather, dim=0)

class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale, logit_bias=None):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features,
                text_features,
                local_loss=self.local_loss,
                gather_with_grad=self.gather_with_grad,
                rank=self.rank,
                world_size=self.world_size,
                use_horovod=self.use_horovod,
            )

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T # for MultiCLIP: e.g. if m=4, then logits_per_image is 16x4
                logits_per_text = logit_scale * text_features @ all_image_features.T # for MultiCLIP: e.g. if m=4, then logits_per_text is 4x16
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        if logit_bias is not None:
            logits_per_image += logit_bias
            logits_per_text += logit_bias

        return logits_per_image, logits_per_text

    def forward(
            self,
            image_features,
            text_features,
            logit_scale,
            logit_bias=None,
            output_dict=False,
    ):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss


class HNClipLoss(ClipLoss):
    
    #Hard Negative Weighted Contrastive Loss from https://arxiv.org/abs/2301.02280
    
    def __init__(self, alpha=1.0, beta=0.25, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta
        print(f"Using HNClipLoss (Implementation similar to DiHT formula) with alpha={self.alpha} and beta={self.beta}")

    def forward(
        self,
        image_features,
        text_features,
        logit_scale,
        logit_bias=None,
        output_dict=False,
    ):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(
            image_features,
            text_features,
            logit_scale,
            logit_bias=logit_bias,
        )
        n_anchors, n_global = logits_per_image.shape
        labels = self.get_ground_truth(device, n_anchors)  # length n_anchors
        rows = torch.arange(n_anchors, device=device)
        mask = torch.ones_like(logits_per_image, dtype=torch.bool)
        mask[rows, labels] = False  # mask out the positive logits

        # =====================
        # Image-to-text loss
        # =====================
        exp_sim_i2t = torch.exp(logits_per_image)
        weights_i2t = torch.exp(self.beta * logits_per_image)
        weights_i2t = weights_i2t * mask  # zero out the positive logits

        weights_i2t = (n_anchors - 1) * weights_i2t / weights_i2t.sum(dim=1, keepdim=True)
        numerators = exp_sim_i2t[rows, labels]
        denominators = self.alpha * numerators + (exp_sim_i2t * weights_i2t).sum(dim=1)
        loss_i2t = -torch.log(numerators / denominators).mean()

        # =====================
        # Text-to-image loss
        # =====================
        exp_sim_t2i = torch.exp(logits_per_text)
        weights_t2i = torch.exp(self.beta * logits_per_text)
        weights_t2i = weights_t2i * mask  # zero out the positive logits
    
        weights_t2i = (n_anchors - 1) * weights_t2i / weights_t2i.sum(dim=1, keepdim=True)
        numerators = exp_sim_t2i[rows, labels]
        denominators = self.alpha * numerators + (exp_sim_t2i * weights_t2i).sum(dim=1)
        loss_t2i = -torch.log(numerators / denominators).mean()
        
        total_loss = (loss_i2t + loss_t2i) / 2
        return {"hn_weighted_contrastive_loss": total_loss} if output_dict else total_loss

def compute_cross_entropy(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    log_q = F.log_softmax(q, dim=-1)
    return -(p * log_q).sum(dim=-1).mean()

class MultiCLIPLoss(ClipLoss):
    def __init__(
        self,
        m: int,
        local_loss=False,
        gather_with_grad=False,
        cache_labels=False,
        rank=0,
        world_size=1,
        use_horovod=False,
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod,
        )
        self.m = m

    def get_ground_truth(self, device: torch.device, num_logits: int):
        """
        Returns:
          - labels_t2i: soft labels for text-to-image, shape [n, n*m*world_size]
          - labels_i2t: one-hot labels for image-to-text, shape [n*m, n*world_size]

        `num_logits` is the number of captions (n).
        """
        # Check cache
        if self.prev_num_logits != num_logits or device not in self.labels:
            n = num_logits
            m = self.m
            ws = self.world_size
            r  = self.rank

            # Caption indices
            cap_idx = torch.arange(n, device=device) + r * n                  # [n]
            all_cap_idx = concat_all_gather(cap_idx)                         # [n*world_size]

            # Image indices
            num_img = n * m
            img_idx = torch.arange(num_img, device=device) + r * num_img     # [n*m]
            all_img_idx = concat_all_gather(img_idx)                         # [n*m*world_size]

            # Text-to-Image soft labels
            img_group = all_img_idx // m                                      # [n*m*world_size]
            mask_t2i = cap_idx.unsqueeze(1) == img_group.unsqueeze(0)         # [n, n*m*world_size]
            labels_t2i = mask_t2i.float().div(mask_t2i.sum(1, keepdim=True).clamp(min=1.0))

            # Image-to-Text one-hot labels
            img_to_cap = img_idx // m                                         # [n*m]
            mask_i2t = img_to_cap.unsqueeze(1) == all_cap_idx.unsqueeze(0)     # [n*m, n*world_size]
            labels_i2t = mask_i2t.float()

            # Cache
            if self.cache_labels:
                self.labels[device] = (labels_t2i, labels_i2t)
                self.prev_num_logits = num_logits
            else:
                return (labels_t2i, labels_i2t)
        # Return from cache or newly computed
        return self.labels[device]

    def forward(
            self,
            image_features: torch.Tensor,
            text_features: torch.Tensor,
            logit_scale: torch.Tensor,
            logit_bias: torch.Tensor = None,
            output_dict: bool = False
    ):
        logits_per_image, logits_per_text = self.get_logits(
            image_features, text_features, logit_scale, logit_bias
        )
        #print(f"Logits per image:\n{logits_per_image.shape}")
        #print(f"Logits per text:\n{logits_per_text.shape}")

        # Determine batch size for ground-truth
        gt_t2i, gt_i2t = self.get_ground_truth(image_features.device, logits_per_text.shape[0])
        #print(f"Ground truth t2i at rank {self.rank} with shape {gt_t2i.shape}:\n{gt_t2i.cpu().numpy()}")
        #print(f"Ground truth i2t at rank {self.rank} with shape {gt_i2t.shape}:\n{gt_i2t.cpu().numpy()}")

        loss_i2t = compute_cross_entropy(gt_i2t, logits_per_image)
        loss_t2i = compute_cross_entropy(gt_t2i, logits_per_text)
        total_loss = (loss_i2t + loss_t2i) / 2
        
        if output_dict:
            return {"mp_contrast_loss": total_loss}
        return total_loss

class StableRepPlusLoss(MultiCLIPLoss):
    def __init__(
        self,
        m: int = 4,
        local_loss=False,
        gather_with_grad=False,
        cache_labels=False,
        rank=0,
        world_size=1,
        use_horovod=False,
    ):
        super().__init__(
            m=m,
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod
        )
        self.stablerep_prev_num_logits = 0
        self.stablerep_labels = {}

    def get_stablerep_logits(self, image_features, logit_scale=10, logit_bias=None):
        if self.world_size > 1:
            all_image_features = gather_image_features(
                image_features,
                local_loss=self.local_loss,
                gather_with_grad=self.gather_with_grad,
                rank=self.rank,
                world_size=self.world_size
            )

            # StableRep originally uses a FIXED temperature of 0.1 and devides the logits, which is equivalent to multiplying by 10
            # Thus here we use a logit_scale of 10 to match the original StableRep implementation

            if self.local_loss:
                logits = logit_scale * image_features @ all_image_features.T
            else:
                logits = logit_scale * all_image_features @ all_image_features.T
        else:
            logits = logit_scale * image_features @ image_features.T

        if logit_bias is not None:
            logits += logit_bias

        return logits

    def get_stablerep_ground_truth(self, device: torch.device, num_logits: int) -> torch.Tensor:
        if self.stablerep_prev_num_logits != num_logits or device not in self.stablerep_labels:
            local_idx = torch.arange(num_logits, device=device)
            if self.local_loss and self.world_size > 1:
                local_idx += self.rank * num_logits
            all_idx = concat_all_gather(local_idx)
            group_local = local_idx // self.m
            group_all = all_idx // self.m
            mask = group_local.unsqueeze(1) == group_all.unsqueeze(0)
            # self-mask is used to exclude self-comparisons
            if self.local_loss:
                # self-mask is used to exclude self-comparisons in the global matrix
                logits_range = torch.arange(num_logits, device=device)
                global_idx = self.rank * num_logits + logits_range
                mask[logits_range, global_idx] = False
            else:
                mask.fill_diagonal_(0)
            labels = mask.float().div(mask.sum(1, keepdim=True).clamp(min=1.0))
            if self.cache_labels:
                self.stablerep_labels[device] = labels
                self.stablerep_prev_num_logits = num_logits
            return labels
        else:
            return self.stablerep_labels[device]

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        logit_scale: torch.Tensor,
        logit_bias: torch.Tensor = None,
        image_embeddings: Optional[torch.Tensor] = None,
        output_dict: bool = False
    ):
        multiclip_loss = super().forward(
            image_features=image_features,
            text_features=text_features,
            logit_scale=logit_scale,
            logit_bias=logit_bias,
            output_dict=False
        )

        #print(f"Using provided image embeddings for StableRep loss at rank {self.rank}.")
        logits = self.get_stablerep_logits(
            image_features=image_embeddings,
            logit_bias=logit_bias
        ) # note: logit_scale is not used here, as StableRep uses a fixed temperature of 0.1

        dtype = logits.dtype
        min_val = -torch.finfo(dtype).max  # most negative representable value
        safe_val = min_val / 10            # avoid edge-of-range numerical instability

        if self.local_loss:
            # Explicitly mask out each local sample's self‐comparison in the global matrix
            B = logits.size(0)
            b_arange = torch.arange(B, device=image_features.device)
            global_idx = self.rank * B + b_arange
            logits[b_arange, global_idx] = safe_val
        else:
            logits.fill_diagonal_(safe_val)  # Avoid self-comparisons in StableRep
        gt = self.get_stablerep_ground_truth(image_features.device, logits.shape[0])
        #print(f"StableRep ground truth p at rank {self.rank}:\n{gt.cpu().numpy()}")
        stablerep_loss = compute_cross_entropy(gt, logits)

        if output_dict:
            return {
                "mp_contrast_loss": multiclip_loss,
                "stable_rep_loss": stablerep_loss
            }
        return multiclip_loss, stablerep_loss

class TripletCLIPLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels
    
    # negclip_loss and tripletclip_loss from original TripletCLIP implementation: https://github.com/tripletclip/TripletCLIP
    # adapted to local loss computation
    def negclip_loss(self, img_embs, text_embs, neg_text_embs, all_img_embs, all_text_embs, all_neg_text_embs, logit_scale):
        # Normalize embeddings
        batch_size = img_embs.shape[0]
        labels = self.get_ground_truth(img_embs.device, batch_size)

        img_text_similarity = logit_scale * img_embs @ all_text_embs.t()
        text_img_similarity = logit_scale * text_embs @ all_img_embs.t()
        img_negtext_similarity = logit_scale * img_embs @ all_neg_text_embs.t()

        '''preds_i2t = torch.cat((img_text_similarity, img_negtext_similarity), dim=-1).argmax(
            dim=-1
        )
        preds_t2i = img_text_similarity.t().argmax(dim=-1)
        acc_i2t = (preds_i2t == labels).float().mean().item()
        acc_t2i = (preds_t2i == labels).float().mean().item()
        accuracy = (acc_i2t + acc_t2i) / 2'''

        loss = (
            F.cross_entropy(
                torch.cat([img_text_similarity, img_negtext_similarity], dim=-1), labels
            )
            + F.cross_entropy(text_img_similarity, labels)
        ).div(2)
        return loss#, accuracy
    
    def tripletclip_loss(self, img_embs, text_embs, neg_img_embs, neg_text_embs, all_img_embs, all_text_embs, all_neg_img_embs, all_neg_text_embs, logit_scale):
        # cross-type image negatives (base<-->hn) is NOT used in TripletCLIP, that's why i think it is not as good ad the usual clip loss
        #loss_1, accuracy1 = self.negclip_loss(img_embs, text_embs, neg_text_embs, all_img_embs, all_text_embs, all_neg_text_embs, logit_scale)
        #loss_2, accuracy2 = self.negclip_loss(neg_img_embs, neg_text_embs, text_embs, all_neg_img_embs, all_neg_text_embs, all_text_embs, logit_scale)
        loss_1 = self.negclip_loss(img_embs, text_embs, neg_text_embs, all_img_embs, all_text_embs, all_neg_text_embs, logit_scale)
        loss_2 = self.negclip_loss(neg_img_embs, neg_text_embs, text_embs, all_neg_img_embs, all_neg_text_embs, all_text_embs, logit_scale)

        loss = loss_1 + loss_2
        #accuracy = (accuracy1 + accuracy2) / 2
        return loss#, accuracy

    def forward(
            self,
            image_features,
            text_features,
            logit_scale,
            logit_bias=None,
            output_dict=False,
    ):

        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod
            )
        else:
            all_image_features, all_text_features = image_features, text_features

        img_embs, neg_img_embs = image_features[0::2], image_features[1::2]
        text_embs, neg_text_embs = text_features[0::2], text_features[1::2]
        all_img_embs, all_neg_img_embs = all_image_features[0::2], all_image_features[1::2]
        all_text_embs, all_neg_text_embs = all_text_features[0::2], all_text_features[1::2]

        # FINO A QUI OK
        loss = self.tripletclip_loss(
            img_embs, text_embs,
            neg_img_embs, neg_text_embs,
            all_img_embs, all_text_embs,
            all_neg_img_embs, all_neg_text_embs,
            logit_scale
        )

        if output_dict:
            return {"tripletclip_loss": loss}
        return loss

class CoCaLoss(ClipLoss):
    def __init__(
            self,
            caption_loss_weight,
            clip_loss_weight,
            pad_id=0,  # pad_token for open_clip custom tokenizer
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod
        )

        self.clip_loss_weight = clip_loss_weight
        self.caption_loss_weight = caption_loss_weight
        self.caption_loss = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, image_features, text_features, logits, labels, logit_scale, output_dict=False):
        if self.clip_loss_weight:
            clip_loss = super().forward(image_features, text_features, logit_scale)
            clip_loss = self.clip_loss_weight * clip_loss
        else:
            clip_loss = torch.tensor(0, device=logits.device)

        caption_loss = self.caption_loss(
            logits.permute(0, 2, 1),
            labels,
        )
        caption_loss = caption_loss * self.caption_loss_weight

        if output_dict:
            return {"contrastive_loss": clip_loss, "caption_loss": caption_loss}

        return clip_loss, caption_loss


class DistillClipLoss(ClipLoss):

    def dist_loss(self, teacher_logits, student_logits):
        return -(teacher_logits.softmax(dim=1) * student_logits.log_softmax(dim=1)).sum(dim=1).mean(dim=0)

    def forward(
            self,
            image_features,
            text_features,
            logit_scale,
            dist_image_features,
            dist_text_features,
            dist_logit_scale,
            output_dict=False,
    ):
        logits_per_image, logits_per_text = \
            self.get_logits(image_features, text_features, logit_scale)

        dist_logits_per_image, dist_logits_per_text = \
            self.get_logits(dist_image_features, dist_text_features, dist_logit_scale)

        labels = self.get_ground_truth(image_features.device, logits_per_image.shape[0])

        contrastive_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        distill_loss = (
            self.dist_loss(dist_logits_per_image, logits_per_image) +
            self.dist_loss(dist_logits_per_text, logits_per_text)
        ) / 2

        if output_dict:
            return {"contrastive_loss": contrastive_loss, "distill_loss": distill_loss}

        return contrastive_loss, distill_loss


def neighbour_exchange(from_rank, to_rank, tensor, group=None):
    tensor_recv = torch.zeros_like(tensor)
    send_op = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor,
        to_rank,
        group=group,
    )
    recv_op = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_recv,
        from_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()
    return tensor_recv


def neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    tensor_from_left = torch.zeros_like(tensor_to_right)
    tensor_from_right = torch.zeros_like(tensor_to_left)
    send_op_left = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_left,
        left_rank,
        group=group,
    )
    send_op_right = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_right,
        right_rank,
        group=group,
    )
    recv_op_left = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_left,
        left_rank,
        group=group,
    )
    recv_op_right = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_right,
        right_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op_right, send_op_left, recv_op_right, recv_op_left])
    for req in reqs:
        req.wait()
    return tensor_from_right, tensor_from_left


class NeighbourExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, from_rank, to_rank, group, tensor):
        ctx.group = group
        ctx.from_rank = from_rank
        ctx.to_rank = to_rank
        return neighbour_exchange(from_rank, to_rank, tensor, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, None) + (NeighbourExchange.apply(ctx.to_rank, ctx.from_rank, ctx.group, grad_output),)


def neighbour_exchange_with_grad(from_rank, to_rank, tensor, group=None):
    return NeighbourExchange.apply(from_rank, to_rank, group, tensor)


class NeighbourExchangeBidir(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left_rank, right_rank, group, tensor_to_left, tensor_to_right):
        ctx.group = group
        ctx.left_rank = left_rank
        ctx.right_rank = right_rank
        return neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=group)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None, None, None) + \
            NeighbourExchangeBidir.apply(ctx.right_rank, ctx.left_rank, ctx.group, *grad_outputs)


def neighbour_exchange_bidir_with_grad(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    return NeighbourExchangeBidir.apply(left_rank, right_rank, group, tensor_to_left, tensor_to_right)


class SigLipLoss(nn.Module):
    """ Sigmoid Loss for Language Image Pre-Training (SigLIP) - https://arxiv.org/abs/2303.15343

    @article{zhai2023sigmoid,
      title={Sigmoid loss for language image pre-training},
      author={Zhai, Xiaohua and Mustafa, Basil and Kolesnikov, Alexander and Beyer, Lucas},
      journal={arXiv preprint arXiv:2303.15343},
      year={2023}
    }
    """
    def __init__(
            self,
            cache_labels: bool = False,
            rank: int = 0,
            world_size: int = 1,
            dist_impl: Optional[str] = None,
    ):
        super().__init__()
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.dist_impl = dist_impl or 'bidir'  # default to bidir exchange for now, this will likely change
        assert self.dist_impl in ('bidir', 'shift', 'reduce', 'gather')

        # cache state FIXME cache not currently used, worthwhile?
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, dtype, num_logits, negative_only=False) -> torch.Tensor:
        labels = -torch.ones((num_logits, num_logits), device=device, dtype=dtype)
        if not negative_only:
            labels = 2 * torch.eye(num_logits, device=device, dtype=dtype) + labels
        return labels

    def get_logits(self, image_features, text_features, logit_scale, logit_bias=None):
        logits = logit_scale * image_features @ text_features.T
        if logit_bias is not None:
            logits += logit_bias
        return logits

    def _loss(self, image_features, text_features, logit_scale, logit_bias=None, negative_only=False):
        logits = self.get_logits(image_features, text_features, logit_scale, logit_bias)
        labels = self.get_ground_truth(
            image_features.device,
            image_features.dtype,
            image_features.shape[0],
            negative_only=negative_only,
        )
        loss = -F.logsigmoid(labels * logits).sum() / image_features.shape[0]
        return loss

    def forward(self, image_features, text_features, logit_scale, logit_bias, output_dict=False):
        loss = self._loss(image_features, text_features, logit_scale, logit_bias)

        if self.world_size > 1:
            if self.dist_impl == 'bidir':
                right_rank = (self.rank + 1) % self.world_size
                left_rank = (self.rank - 1 + self.world_size) % self.world_size
                text_features_to_right = text_features_to_left = text_features
                num_bidir, remainder = divmod(self.world_size - 1, 2)
                for i in range(num_bidir):
                    text_features_recv = neighbour_exchange_bidir_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_left,
                        text_features_to_right,
                    )
                    for f in text_features_recv:
                        loss += self._loss(
                            image_features,
                            f,
                            logit_scale,
                            logit_bias,
                            negative_only=True,
                        )
                    text_features_to_left, text_features_to_right = text_features_recv

                if remainder:
                    text_features_recv = neighbour_exchange_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_right
                    )
                    loss += self._loss(
                        image_features,
                        text_features_recv,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            elif self.dist_impl == "shift":
                right_rank = (self.rank + 1) % self.world_size
                left_rank = (self.rank - 1 + self.world_size) % self.world_size
                text_features_to_right = text_features
                for i in range(self.world_size - 1):
                    text_features_from_left = neighbour_exchange_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_right,
                    )
                    loss += self._loss(
                        image_features,
                        text_features_from_left,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
                    text_features_to_right = text_features_from_left
            elif self.dist_impl == "reduce":
                for i in range(self.world_size):
                    text_from_other = torch.distributed.nn.all_reduce(
                        text_features * (self.rank == i),
                        torch.distributed.ReduceOp.SUM,
                    )
                    loss += float(i != self.rank) * self._loss(
                        image_features,
                        text_from_other,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            elif self.dist_impl == "gather":
                all_text = torch.distributed.nn.all_gather(text_features)
                for i in range(self.world_size):
                    loss += float(i != self.rank) * self._loss(
                        image_features,
                        all_text[i],
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            else:
                assert False

        return {"contrastive_loss": loss} if output_dict else loss


'''class MockHNClipLoss(ClipLoss):
    def __init__(self, alpha=1.0, beta=0.25, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta

    def forward(
        self,
        image_features,
        text_features,
        global_image_features,
        global_text_features,
        logit_scale,
        logit_bias=None,
        output_dict=False,
    ):
        device = image_features.device

        logits_per_image = logit_scale * image_features @ global_text_features.T
        logits_per_text = logit_scale * text_features @ global_image_features.T
        n_anchors, n_global = logits_per_image.shape
        labels = self.get_ground_truth(device, n_anchors)  # length n_anchors
        print(f"LABELS at rank {self.rank} with shape {labels.shape}:\n{labels.cpu().numpy()}")
        rows = torch.arange(n_anchors, device=device)
        mask = torch.ones_like(logits_per_image, dtype=torch.bool)
        mask[rows, labels] = False  # mask out the positive logits

        # =====================
        # Image-to-text loss
        # =====================
        exp_sim_i2t = torch.exp(logits_per_image)
        print(f"Exp sim i2t at rank {self.rank} with shape {exp_sim_i2t.shape}:\n{exp_sim_i2t.cpu().numpy()}")
        weights_i2t = torch.exp(self.beta * logits_per_image)
        weights_i2t = weights_i2t * mask  # zero out the positive logits

        weights_i2t = (n_anchors - 1) * weights_i2t / weights_i2t.sum(dim=1, keepdim=True)
        print(f"Weights i2t at rank {self.rank} with shape {weights_i2t.shape}:\n{weights_i2t.cpu().numpy()}")
        numerators = exp_sim_i2t[rows, labels]
        print(f"Numerators i2t at rank {self.rank} with shape {numerators.shape}:\n{numerators.cpu().numpy()}")
        denominators = self.alpha * numerators + (exp_sim_i2t * weights_i2t).sum(dim=1)
        print(f"Denominators i2t at rank {self.rank} with shape {denominators.shape}:\n{denominators.cpu().numpy()}")
        loss_i2t = -torch.log(numerators / denominators).mean()

        # =====================
        # Text-to-image loss
        # =====================
        exp_sim_t2i = torch.exp(logits_per_text)
        weights_t2i = torch.exp(self.beta * logits_per_text)
        weights_t2i = weights_t2i * mask  # zero out the positive logits
    
        weights_t2i = (n_anchors - 1) * weights_t2i / weights_t2i.sum(dim=1, keepdim=True)
        numerators = exp_sim_t2i[rows, labels]
        denominators = self.alpha * numerators + (exp_sim_t2i * weights_t2i).sum(dim=1)
        loss_t2i = -torch.log(numerators / denominators).mean()
        
        total_loss = (loss_i2t + loss_t2i) / 2
        return {"hn_weighted_contrastive_loss": total_loss} if output_dict else total_loss

if __name__ == "__main__":
    torch.manual_seed(0)
    n = 16  # total samples
    d = 128
    world_size = 4
    logit_scale = torch.tensor(1.0)

    img_feats = torch.randn(n, d)
    txt_feats = torch.randn(n, d)

    # simulate splitting across 4 gpus
    local_bs = n // world_size
    for rank in range(world_size):
        loss_fn = MockHNClipLoss(local_loss=True, rank=rank, world_size=world_size)
        print(f"\n=== Simulated GPU {rank} ===")
        local_img = img_feats[rank * local_bs : (rank + 1) * local_bs]
        local_txt = txt_feats[rank * local_bs : (rank + 1) * local_bs]

        loss = loss_fn(local_img, local_txt, img_feats, txt_feats, logit_scale, output_dict=False)
        print("loss:", loss.item())'''