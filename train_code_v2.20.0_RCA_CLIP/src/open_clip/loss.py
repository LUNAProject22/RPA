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


def gather_features_v3(
        image_features,
        text_features,
        neg_image_features,
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
            all_neg_image_features = hvd.allgather(neg_image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_neg_image_features = hvd.allgather(neg_image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_neg_image_features = list(all_neg_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_neg_image_features[rank] = neg_image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_neg_image_features = torch.cat(gathered_neg_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_neg_image_features = torch.cat(torch.distributed.nn.all_gather(neg_image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_neg_image_features = [torch.zeros_like(neg_image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_neg_image_features, neg_image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_neg_image_features[rank] = neg_image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_neg_image_features = torch.cat(gathered_neg_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features, all_neg_image_features


class Max_ClipLoss(nn.Module):

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

    def get_logits(self, text_features, image_features, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                b, n, d = image_features.shape
                b_all, n, d = all_image_features.shape

                b, d = text_features.shape
                b_all, d = all_text_features.shape

                #print(image_features.shape, all_image_features.shape)
                #print(text_features.shape, all_text_features.shape)

                # b x b_all
                logits_per_image = logit_scale * image_features @ all_text_features.T   # [b, n, b_all]
                logits_per_image = logits_per_image.max(dim=1)[0]                # [b, b_all]

                # b_all x b
                logits_per_text = logit_scale * all_image_features @ text_features.T   # [b_all, n, b]
                logits_per_text = logits_per_text.max(dim=1)[0]                 # [b_all, b]
                logits_per_text = logits_per_text.T                             # [b, b_all]

                #logits_per_image = logit_scale * image_features @ all_text_features.T
                #logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T
        
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale, output_dict=False):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss


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

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T
        
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale, output_dict=False):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss


class ClipLoss_NegImg(nn.Module):

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

    def get_logits(self, image_features, text_features, neg_image_features, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features, all_neg_image_features = gather_features_v3(
                image_features, text_features, neg_image_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ torch.cat((all_image_features, all_neg_image_features), dim=0).T
                #logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logit_scale * all_text_features @ torch.cat((all_image_features, all_neg_image_features), dim=0).T
                #logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ torch.cat((image_features, neg_image_features), dim=0).T
            #logits_per_text = logit_scale * text_features @ image_features.T
        
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, neg_image_features, logit_scale, output_dict=False):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, neg_image_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss


#class Max_ClipLoss(nn.Module):
#
#    def __init__(
#            self,
#            local_loss=False,
#            gather_with_grad=False,
#            cache_labels=False,
#            rank=0,
#            world_size=1,
#            use_horovod=False,
#    ):
#        super().__init__()
#        self.local_loss = local_loss
#        self.gather_with_grad = gather_with_grad
#        self.cache_labels = cache_labels
#        self.rank = rank
#        self.world_size = world_size
#        self.use_horovod = use_horovod
#
#        # cache state
#        self.prev_num_logits = 0
#        self.labels = {}
#
#    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
#        # calculated ground-truth and cache if enabled
#        if self.prev_num_logits != num_logits or device not in self.labels:
#            labels = torch.arange(num_logits, device=device, dtype=torch.long)
#            if self.world_size > 1 and self.local_loss:
#                labels = labels + num_logits * self.rank
#            if self.cache_labels:
#                self.labels[device] = labels
#                self.prev_num_logits = num_logits
#        else:
#            labels = self.labels[device]
#        return labels
#
#    def get_logits(self, features_A, features_B, logit_scale):
#        # features_A: [batch_size, feature_dim]
#        # features_B: [batch_size, num_sentences, feature_dim]
#        # logit_scale: scalar
#        
#        b, num_sentences, num_words = features_B.shape
#        #features_B = features_B.reshape(b*num_sentences, num_words)
#
#        if self.world_size > 1:
#            all_features_A, all_features_B = gather_features(
#                features_A, features_B,
#                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)
#
#            if self.local_loss:
#                b, d   = features_A.shape
#                b_all, d = all_features_A.shape
#
#                b, n, d   = features_B.shape
#                b_all, n, d = all_features_B.shape
#
#                # b x b_all
#                logits_per_image = logit_scale * features_B @ all_features_A.T   # [b, n, b_all]
#                #logits_per_image = logits_per_image.max(dim=1)[0]                # [b, b_all]
#                logits_per_image  = logits_per_image.squeeze(1)
#
#                # b x b_b
#                logits_per_text  = logit_scale * all_features_B @ features_A.T   # [b_all, n, b]
#                logits_per_text  = logits_per_text.squeeze(1)
#                #logits_per_text  = logits_per_text.max(dim=1)[0]                 # [b_all, b]
#                logits_per_text   = logits_per_text.T                             # [b, b_all]
#                
#                #logits_per_image = logit_scale * features_A @ all_features_B.T   # [b, bn]
#                #ba, bb = logits_per_image.shape
#                #logits_per_image = logits_per_image.reshape(ba, -1, num_sentences) # [b, b, num_sentences]
#                #logits_per_image = logits_per_image.max(dim=2)[0]                # [b, b]
#
#                #logits_per_text = logit_scale * features_B @ all_features_A.T  # [bn, b]
#                #bb, ba = logits_per_text.shape
#                #logits_per_text = logits_per_text.reshape(-1, num_sentences, ba) # [b, num_sentences, b]
#                #logits_per_text = logits_per_text.max(dim=1)[0]                # [b, b]
#            else:
#                # To Hack
#                logits_per_image = logit_scale * all_features_A @ all_features_B.T
#                logits_per_text = logits_per_image.T
#        else:
#            # To Hack
#            logits_per_image = logit_scale * features_A @ features_B.T
#            logits_per_text = logit_scale * features_B @ features_A.T
#        
#        return logits_per_image, logits_per_text
#
#    def forward(self, features_A, features_B, logit_scale, output_dict=False):
#        device = features_A.device
#        logits_per_image, logits_per_text = self.get_logits(features_A, features_B, logit_scale)
#
#        labels = self.get_ground_truth(device, logits_per_image.shape[0])
#
#        total_loss = (
#            F.cross_entropy(logits_per_image, labels) +
#            F.cross_entropy(logits_per_text, labels)
#        ) / 2
#
#        return {"contrastive_loss": total_loss} if output_dict else total_loss
    

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
        
        clip_loss = 0
        
        if self.clip_loss_weight:
            clip_loss = super().forward(image_features, text_features, logit_scale)
            clip_loss = self.clip_loss_weight * clip_loss

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
