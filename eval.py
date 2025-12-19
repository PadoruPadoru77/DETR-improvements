from util.plot_utils import plot_logs
from pathlib import Path
import sys
import os
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image
import time
import argparse

# Ensure we import from local models directory, not torch hub cache
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import build_model


# standard PyTorch mean-std input image normalization
transform = T.Compose([T.Resize(800), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def filter_bboxes_from_outputs(outputs, threshold=0.7):
    # keep only predictions with confidence above threshold
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > threshold

    probas_to_keep = probas[keep]

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)

    return probas_to_keep, bboxes_scaled


# PASCAL class
num_classes = 21
finetuned_classes = [
    'N/A', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
]

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# show training log
# log_directory = [Path('outputs/')]
# fields_of_interest = ('loss', 'mAP')
# plot_logs(log_directory, fields_of_interest)

# show training results
# Create args object with training configuration using argparse.Namespace
args = argparse.Namespace(
    # Model parameters
    num_classes=21,
    num_queries=50,  # Match the training
    aux_loss=True,

    # Backbone
    backbone='resnet50',
    dilation=False,
    position_embedding='sine',
    lr_backbone=1e-5,

    # Transformer
    enc_layers=6,
    dec_layers=4,  # Match training
    dim_feedforward=2048,
    hidden_dim=256,
    dropout=0.1,
    nheads=8,
    pre_norm=False,

    # DN-DETR parameters
    use_dn=True,  # Enable DN-DETR
    num_dn_groups=5,
    label_noise_ratio=0.2,
    box_noise_scale=0.4,

    # Loss coefficients
    bbox_loss_coef=5,
    giou_loss_coef=2,
    eos_coef=0.1,

    # Matcher coefficients
    set_cost_class=1,
    set_cost_bbox=5,
    set_cost_giou=2,

    # Segmentation
    mask_loss_coef=1,
    dice_loss_coef=1,

    # Other
    masks=False,
    device='cpu',
    dataset_file='pascal',  # Changed from 'custom' to avoid override issues
    frozen_weights=None
)

# Build model using DN-DETR implementation
model, criterion, postprocessors = build_model(args)

# Load checkpoint
checkpoint = torch.load('outputs/checkpoint.pth', map_location='cpu', weights_only=False)

# Handle class_embed size mismatch
checkpoint_state = checkpoint['model']
model_state = model.state_dict()

if 'class_embed.weight' in checkpoint_state and 'class_embed.weight' in model_state:
    ckpt_classes = checkpoint_state['class_embed.weight'].shape[0]
    model_classes = model_state['class_embed.weight'].shape[0]

    if ckpt_classes != model_classes:
        print(f"Warning: Checkpoint has {ckpt_classes} class outputs, model has {model_classes}")
        print(f"Resizing class_embed to match checkpoint...")
        # Keep first N classes or pad with checkpoint values
        if ckpt_classes > model_classes:
            # Checkpoint has more - take first N
            checkpoint_state['class_embed.weight'] = checkpoint_state['class_embed.weight'][:model_classes]
            checkpoint_state['class_embed.bias'] = checkpoint_state['class_embed.bias'][:model_classes]
        else:
            # Model has more - pad with checkpoint
            padded_weight = model_state['class_embed.weight'].clone()
            padded_bias = model_state['class_embed.bias'].clone()
            padded_weight[:ckpt_classes] = checkpoint_state['class_embed.weight']
            padded_bias[:ckpt_classes] = checkpoint_state['class_embed.bias']
            checkpoint_state['class_embed.weight'] = padded_weight
            checkpoint_state['class_embed.bias'] = padded_bias

model.load_state_dict(checkpoint_state, strict=False)
model.eval()


def plot_finetuned_results(pil_img, prob=None, boxes=None, threshold=None):
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    if prob is not None and boxes is not None:
        for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=c, linewidth=3))
            cl = p.argmax()
            text = f'{finetuned_classes[cl]}: {p[cl]:0.2f}'
            ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    title = "The detection threshold is set to " + str(threshold)
    plt.title(title)
    plt.show()


def run_worflow(my_image, my_model):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(my_image).unsqueeze(0)

    # propagate through the model
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()
    outputs = my_model(img)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.time()

    # Calculate and print inference time
    inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
    print(f"Inference time: {inference_time:.2f} ms ({inference_time/1000:.4f} seconds)")

    for threshold in [0.9, 0.7, 0.5]:
        probas_to_keep, bboxes_scaled = filter_bboxes_from_outputs(outputs, threshold=threshold)
        plot_finetuned_results(my_image, probas_to_keep, bboxes_scaled, threshold)


# run eval code
img_name = '/content/data/custom/trainval/2007_000032.jpg'
im = Image.open(img_name)
run_worflow(im, model)

