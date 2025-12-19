import torch
import torch.nn.functional as F
from util.misc import inverse_sigmoid


def prepare_for_dn(dn_args, training, num_queries, num_classes, hidden_dim, label_enc):
    if not training or dn_args is None:
        return None, None, None, None

    targets = dn_args['targets']
    num_dn_groups = dn_args['num_dn_groups']
    label_noise_ratio = dn_args['label_noise_ratio']
    box_noise_scale = dn_args['box_noise_scale']

    if num_dn_groups == 0:
        return None, None, None, None

    batch_size = len(targets)

    # Prepare ground truth
    known_labels_list = []
    known_bboxs_list = []
    known_num_list = []

    for t in targets:
        known_labels_list.append(t['labels'])
        known_bboxs_list.append(t['boxes'])
        known_num_list.append(t['boxes'].shape[0])

    known_num = max(known_num_list)

    if known_num == 0:
        return None, None, None, None

    # Prepare for denoising
    unmask_bbox = torch.zeros(batch_size, known_num, 4).cuda()
    unmask_label = torch.zeros(batch_size, known_num, dtype=torch.long).cuda()
    valid_mask_original = torch.zeros(batch_size, known_num, dtype=torch.bool).cuda()

    for i, (labels, bboxs) in enumerate(zip(known_labels_list, known_bboxs_list)):
        num = labels.shape[0]
        unmask_bbox[i, :num] = bboxs
        unmask_label[i, :num] = labels
        valid_mask_original[i, :num] = True  # Mark valid boxes

    # Create attention mask
    num_dn = num_dn_groups * known_num

    # Add noise to labels
    labels = unmask_label.repeat(1, num_dn_groups).view(-1)
    boxes = unmask_bbox.repeat(1, num_dn_groups, 1)
    valid_mask = valid_mask_original.repeat(1, num_dn_groups)
    batch_idx = torch.arange(batch_size).unsqueeze(1).repeat(1, num_dn).view(-1).cuda()

    # Label noise - only apply to valid boxes
    if label_noise_ratio > 0:
        prob = torch.rand_like(labels.float())
        chosen_indice = prob < label_noise_ratio
        new_label = torch.randint_like(labels, 0, num_classes)
        labels = torch.where(chosen_indice, new_label, labels)

    # Box noise - only apply to valid boxes
    if box_noise_scale > 0:
        diff = torch.zeros_like(boxes)
        diff[:, :, :2] = boxes[:, :, 2:] / 2
        diff[:, :, 2:] = boxes[:, :, 2:]

        # Generate noise for all boxes
        noise = torch.mul((torch.rand_like(boxes) * 2 - 1.0), diff) * box_noise_scale
        # Only apply noise to valid boxes using the mask
        boxes = torch.where(valid_mask.unsqueeze(-1), boxes + noise, boxes)
        boxes = boxes.clamp(min=0.0, max=1.0)

    # Create query embeddings
    input_query_label = label_enc(labels)
    input_query_label = input_query_label.view(batch_size, num_dn, -1).transpose(0, 1)
    input_query_bbox = inverse_sigmoid(boxes) 
    input_query_bbox = input_query_bbox.transpose(0, 1)

    attn_mask = create_dn_attention_mask(num_dn, num_queries, known_num, batch_size)

    # Store metadata
    dn_meta = {
        'num_dn_groups': num_dn_groups,
        'num_group_queries': known_num,
        'num_dn': num_dn,
        'known_num': known_num
    }

    return input_query_label, input_query_bbox, attn_mask, dn_meta


def create_dn_attention_mask(num_dn, num_queries, num_group_queries, batch_size):
    # If no denoising queries, return None
    if num_dn == 0:
        return None

    total_queries = num_dn + num_queries

    # P = number of denoising groups
    # M = number of queries per group (num_group_queries)
    P = num_dn // num_group_queries if num_group_queries > 0 else 0
    M = num_group_queries

    # Create mask
    # Initialize all to False (can attend)
    attn_mask = torch.zeros(total_queries, total_queries, dtype=torch.bool)

    # Apply masking rules for denoising part (first num_dn queries)
    for i in range(total_queries):
        for j in range(total_queries):
            # Rule 1: Different denoising groups cannot see each other
            if j < num_dn and i < num_dn:
                if (i // M) != (j // M):
                    attn_mask[i, j] = True  # Block

            # Rule 2: Matching part cannot see denoising part
            if j < num_dn and i >= num_dn:
                attn_mask[i, j] = True  # Block

    return attn_mask.cuda()


def dn_post_process(outputs, dn_meta):
    #Remove denoising queries from outputs during inference

    if dn_meta is None or dn_meta['num_dn'] == 0:
        return outputs

    num_dn = dn_meta['num_dn']

    # Remove denoising part from outputs
    outputs['pred_logits'] = outputs['pred_logits'][:, num_dn:]
    outputs['pred_boxes'] = outputs['pred_boxes'][:, num_dn:]

    if 'aux_outputs' in outputs:
        for i in range(len(outputs['aux_outputs'])):
            outputs['aux_outputs'][i]['pred_logits'] = outputs['aux_outputs'][i]['pred_logits'][:, num_dn:]
            outputs['aux_outputs'][i]['pred_boxes'] = outputs['aux_outputs'][i]['pred_boxes'][:, num_dn:]

    return outputs
