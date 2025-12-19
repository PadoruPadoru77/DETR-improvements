"""
Inspect checkpoint file to see what's inside
"""
import torch
import sys

checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "detr-r50_no-class-head.pth"

print(f"Loading checkpoint: {checkpoint_path}\n")

try:
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    print("=" * 70)
    print("CHECKPOINT CONTENTS")
    print("=" * 70)

    # Check top-level keys
    if isinstance(ckpt, dict):
        print(f"\nTop-level keys: {list(ckpt.keys())}\n")

        # If 'model' key exists, it's a training checkpoint
        if 'model' in ckpt:
            model_dict = ckpt['model']
            print("This is a training checkpoint with 'model' state dict")
            if 'epoch' in ckpt:
                print(f"Saved at epoch: {ckpt['epoch']}")
            if 'args' in ckpt:
                print(f"Training args available: Yes")
        else:
            # Might be just state_dict directly
            model_dict = ckpt
            print("This appears to be a direct state_dict")
    else:
        model_dict = ckpt
        print("Checkpoint is not a dict - unusual format")

    print(f"\nTotal parameters in checkpoint: {len(model_dict)}")

    # Analyze what's in the model
    print("\n" + "=" * 70)
    print("PARAMETER BREAKDOWN")
    print("=" * 70)

    backbone_params = [k for k in model_dict.keys() if 'backbone' in k]
    transformer_params = [k for k in model_dict.keys() if 'transformer' in k]
    class_embed_params = [k for k in model_dict.keys() if 'class_embed' in k]
    bbox_embed_params = [k for k in model_dict.keys() if 'bbox_embed' in k]
    input_proj_params = [k for k in model_dict.keys() if 'input_proj' in k]
    query_embed_params = [k for k in model_dict.keys() if 'query_embed' in k]

    print(f"\nBackbone parameters: {len(backbone_params)}")
    print(f"Transformer parameters: {len(transformer_params)}")
    print(f"Input projection parameters: {len(input_proj_params)}")
    print(f"Query embedding parameters: {len(query_embed_params)}")
    print(f"Class embedding parameters: {len(class_embed_params)}")
    print(f"BBox embedding parameters: {len(bbox_embed_params)}")

    # Check backbone architecture
    print("\n" + "=" * 70)
    print("BACKBONE ARCHITECTURE")
    print("=" * 70)

    # Check input_proj to determine backbone output channels
    if 'input_proj.weight' in model_dict:
        input_proj_shape = model_dict['input_proj.weight'].shape
        print(f"\ninput_proj.weight shape: {input_proj_shape}")
        print(f"  → Backbone output channels: {input_proj_shape[1]}")

        if input_proj_shape[1] == 2048:
            print("  → This is from ResNet-50 or ResNet-101")
        elif input_proj_shape[1] == 512:
            print("  → This is from ResNet-18 or ResNet-34")

    # Sample some backbone parameters
    print("\nFirst 15 backbone parameters:")
    for i, k in enumerate(backbone_params[:15]):
        print(f"  {k}: {model_dict[k].shape}")

    # Check if class head exists
    print("\n" + "=" * 70)
    print("CLASSIFICATION HEAD")
    print("=" * 70)

    if len(class_embed_params) > 0:
        print(f"\nClass embedding parameters found: {class_embed_params}")
        for k in class_embed_params:
            print(f"  {k}: {model_dict[k].shape}")

        # Extract number of classes
        if 'class_embed.weight' in model_dict:
            num_classes = model_dict['class_embed.weight'].shape[0] - 1  # -1 for no-object class
            print(f"\n  → Number of classes: {num_classes}")
    else:
        print("\nNo class embedding found - class head was removed!")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_params = sum(p.numel() for p in model_dict.values())
    print(f"\nTotal parameter count: {total_params:,}")

    # Determine what this checkpoint is
    print("\nThis checkpoint contains:")
    if len(backbone_params) > 0:
        print("  ✓ Backbone (ResNet)")
    if len(transformer_params) > 0:
        print("  ✓ Transformer (encoder + decoder)")
    if len(input_proj_params) > 0:
        print("  ✓ Input projection layer")
    if len(query_embed_params) > 0:
        print("  ✓ Query embeddings")
    if len(bbox_embed_params) > 0:
        print("  ✓ BBox prediction head")
    if len(class_embed_params) > 0:
        print("  ✓ Class prediction head")
    else:
        print("  ✗ Class prediction head (removed)")

except FileNotFoundError:
    print(f"Error: File '{checkpoint_path}' not found!")
except Exception as e:
    print(f"Error loading checkpoint: {e}")
