# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .transformer import build_transformer
# ========== DN-DETR ADDITION: Import denoising components ==========
from .dn_components import prepare_for_dn, dn_post_process
# ===================================================================


class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False, use_dn=False):  # DN-DETR: Added use_dn parameter
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            use_dn: True if DN-DETR denoising training should be used. [DN-DETR ADDITION]
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.num_classes = num_classes  # DN-DETR: Store num_classes for denoising
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss
        # ========== DN-DETR ADDITION START =============
        self.use_dn = use_dn

        # DN-DETR: Label encoding layer for converting class labels to embeddings
        if use_dn:
            self.label_enc = nn.Embedding(num_classes, hidden_dim)
            self.bbox_pos_enc = MLP(4, hidden_dim, hidden_dim, 3)
        # ========== DN-DETR ADDITION END ============

    def forward(self, samples: NestedTensor, targets=None, dn_args=None):  # DN-DETR: Added targets and dn_args parameters
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None

        # ========== DN-DETR ADDITION START: Prepare denoising queries ===============
        input_query_label, input_query_bbox, attn_mask, dn_meta = None, None, None, None
        if self.use_dn and dn_args is not None:
            if targets is not None:
                dn_args['targets'] = targets
                # Prepare denoising queries by adding noise to ground truth
                input_query_label, input_query_bbox, attn_mask, dn_meta = prepare_for_dn(
                    dn_args, self.training, self.num_queries, self.num_classes,
                    self.transformer.d_model, self.label_enc
                )

        # Combine denoising queries (if any) with regular matching queries
        # Denoising queries come first, matching queries come second
        if input_query_label is not None and input_query_bbox is not None:
            bs = src.shape[0]
            matching_queries = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)  # [num_queries, batch_size, hidden_dim]

            num_dn = input_query_bbox.shape[0]
            bbox_pos_embed = self.bbox_pos_enc(input_query_bbox.flatten(0, 1)).view(num_dn, bs, -1)  # [num_dn, batch_size, hidden_dim]

            query_embed = torch.cat([bbox_pos_embed, matching_queries], dim=0)  # [num_dn + num_queries, batch_size, hidden_dim]

            matching_tgt_init = torch.zeros_like(matching_queries)
            tgt_init = torch.cat([input_query_label, matching_tgt_init], dim=0)  # [num_dn + num_queries, batch_size, hidden_dim]
        else:
            query_embed = self.query_embed.weight
            tgt_init = None
        # ========== DN-DETR ADDITION END ==========

        # Pass attention mask to transformer (DN-DETR prevents denoising queries from seeing each other)
        hs = self.transformer(self.input_proj(src), mask, query_embed, pos[-1], attn_mask=attn_mask, tgt_init=tgt_init)[0]  # DN-DETR: Added attn_mask and tgt_init

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        # ========== DN-DETR ADDITION START: Handle denoising metadata =============
        # Store DN metadata for loss computation
        if dn_meta is not None:
            out['dn_meta'] = dn_meta

        # Remove denoising queries during inference (only use matching queries for predictions)
        if not self.training and dn_meta is not None:
            out = dn_post_process(out, dn_meta)
        # ========== DN-DETR ADDITION END ==========

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, use_dn=False):  # DN-DETR: Added use_dn parameter
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            use_dn: whether to use DN-DETR denoising losses [DN-DETR ADDITION]
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.use_dn = use_dn  # DN-DETR: Store use_dn flag
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        # AMP: Cast empty_weight to match src_logits dtype (float16 with AMP, float32 without)
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight.to(src_logits.dtype))
        losses = {'loss_ce': loss_ce}

        if log:
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # ========== DN-DETR ADDITION START: Handle denoising losses ==============
        # DN-DETR splits outputs into two parts:
        # 1. Denoising part (first num_dn queries) - uses direct one-to-one matching with GT
        # 2. Matching part (remaining queries) - uses Hungarian matching like original DETR
        dn_meta = outputs.get('dn_meta', None)
        if dn_meta is not None and dn_meta['num_dn'] > 0:
            # Split outputs into denoising and matching parts
            num_dn = dn_meta['num_dn']
            dn_outputs = {
                'pred_logits': outputs['pred_logits'][:, :num_dn, :],  # Denoising queries
                'pred_boxes': outputs['pred_boxes'][:, :num_dn, :]
            }
            matching_outputs = {
                'pred_logits': outputs['pred_logits'][:, num_dn:, :],  # Matching queries (original DETR)
                'pred_boxes': outputs['pred_boxes'][:, num_dn:, :]
            }

            # Compute denoising losses (one-to-one assignment, no matching needed)
            dn_losses = self.compute_dn_loss(dn_outputs, targets, dn_meta)

            # Compute matching losses
            outputs_without_aux = {k: v for k, v in matching_outputs.items()}
        else:
            # No DN-DETR, use standard DETR
            outputs_without_aux = {k: v for k, v in outputs.items() if k not in ['aux_outputs', 'dn_meta']}
            dn_losses = {}
        # ========== DN-DETR ADDITION END =============

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses (standard DETR losses for matching queries)
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs_without_aux, targets, indices, num_boxes))

        # ========== DN-DETR ADDITION: Add denoising losses ==========
        losses.update(dn_losses)
        # ============================================================

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                # ========== DN-DETR ADDITION START: Handle auxiliary denoising outputs ==========
                # Split auxiliary outputs if DN is used
                if dn_meta is not None and dn_meta['num_dn'] > 0:
                    num_dn = dn_meta['num_dn']

                    # Compute DN losses for this auxiliary layer
                    aux_dn_outputs = {
                        'pred_logits': aux_outputs['pred_logits'][:, :num_dn, :],
                        'pred_boxes': aux_outputs['pred_boxes'][:, :num_dn, :]
                    }
                    aux_dn_losses = self.compute_dn_loss(aux_dn_outputs, targets, dn_meta)
                    # Add suffix to distinguish from final layer DN losses
                    aux_dn_losses = {k + f'_{i}': v for k, v in aux_dn_losses.items()}
                    losses.update(aux_dn_losses)

                    # Compute matching losses on matching part
                    aux_matching = {
                        'pred_logits': aux_outputs['pred_logits'][:, num_dn:, :],
                        'pred_boxes': aux_outputs['pred_boxes'][:, num_dn:, :]
                    }
                    indices = self.matcher(aux_matching, targets)
                else:
                    # Standard DETR
                    aux_matching = aux_outputs
                    indices = self.matcher(aux_outputs, targets)
                # ========== DN-DETR ADDITION END ==========

                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_matching, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    # ========== DN-DETR ADDITION: New method for computing denoising losses ==============
    def compute_dn_loss(self, dn_outputs, targets, dn_meta):
        losses = {}
        num_dn = dn_meta['num_dn']
        known_num = dn_meta['known_num']  # Max objects in batch
        num_dn_groups = dn_meta['num_dn_groups']

        # Prepare ground truth
        batch_size = len(targets)
        device = dn_outputs['pred_logits'].device

        # Build ground truth labels and boxes for denoising queries
        # Need to pad each sample to known_num and repeat for num_dn_groups
        gt_labels = torch.zeros(batch_size, num_dn, dtype=torch.long, device=device)
        gt_boxes = torch.zeros(batch_size, num_dn, 4, device=device)
        valid_mask = torch.zeros(batch_size, num_dn, dtype=torch.bool, device=device)

        for i, t in enumerate(targets):
            labels = t['labels']
            boxes = t['boxes']
            num_gt = labels.shape[0]

            # Repeat GT for each denoising group
            for g in range(num_dn_groups):
                start_idx = g * known_num
                end_idx = start_idx + num_gt
                gt_labels[i, start_idx:end_idx] = labels
                gt_boxes[i, start_idx:end_idx] = boxes
                valid_mask[i, start_idx:end_idx] = True  # Mark valid positions

        # Flatten for loss computation
        pred_logits_dn = dn_outputs['pred_logits'] 
        pred_boxes_dn = dn_outputs['pred_boxes']  

        # Flatten
        pred_logits_dn = pred_logits_dn.flatten(0, 1)  
        pred_boxes_dn = pred_boxes_dn.flatten(0, 1)
        gt_labels = gt_labels.flatten()  
        gt_boxes = gt_boxes.flatten(0, 1) 
        valid_mask = valid_mask.flatten() 

        # Compute losses only on valid positions
        if valid_mask.sum() > 0:
            # Classification loss
            loss_ce_dn = F.cross_entropy(pred_logits_dn[valid_mask], gt_labels[valid_mask])
            losses['loss_ce_dn'] = loss_ce_dn

            # Box regression loss
            loss_bbox_dn = F.l1_loss(pred_boxes_dn[valid_mask], gt_boxes[valid_mask])
            losses['loss_bbox_dn'] = loss_bbox_dn

            # GIoU loss
            from util import box_ops
            valid_pred_boxes = pred_boxes_dn[valid_mask]
            valid_gt_boxes = gt_boxes[valid_mask]
            loss_giou_dn = 1 - torch.diag(box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(valid_pred_boxes),
                box_ops.box_cxcywh_to_xyxy(valid_gt_boxes)
            )).mean()
            losses['loss_giou_dn'] = loss_giou_dn
        else:
            # No valid GT, set losses to zero
            losses['loss_ce_dn'] = torch.tensor(0.0, device=device)
            losses['loss_bbox_dn'] = torch.tensor(0.0, device=device)
            losses['loss_giou_dn'] = torch.tensor(0.0, device=device)

        return losses
    # ========== DN-DETR ADDITION END ========================


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    num_classes = 20 if args.dataset_file != 'coco' else 91
    if args.dataset_file == "coco_panoptic":
        # for panoptic, we just add a num_classes that is large enough to hold
        # max_obj_id + 1, but the exact value doesn't really matter
        num_classes = 250
    if args.dataset_file == 'custom':
        # "You should always use num_classes = max_id + 1 where max_id is the highest class ID that you have in your dataset."
        # Reference: https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
        num_classes = 2
    num_classes_specified_at_run_time = args.num_classes
    if num_classes_specified_at_run_time is not None:
        # Override the value hard-coded in this file with the value specified at run-time
        num_classes = num_classes_specified_at_run_time
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    # ========== DN-DETR ADDITION: Check if DN-DETR is enabled ==========
    use_dn = getattr(args, 'use_dn', False)
    # ===================================================================

    model = DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        use_dn=use_dn,  # DN-DETR: Pass use_dn flag to model
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef

    # ========== DN-DETR ADDITION: Add loss weights for denoising ==========
    if use_dn:
        weight_dict['loss_ce_dn'] = 1  # Denoising classification loss
        weight_dict['loss_bbox_dn'] = args.bbox_loss_coef  # Denoising bbox loss
        weight_dict['loss_giou_dn'] = args.giou_loss_coef  # Denoising GIoU loss
    # =====================================================================

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses, use_dn=use_dn)  # DN-DETR: Pass use_dn to criterion
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
