import argparse
import torch
from models.detr import build


def count_parameters(model, verbose=False):
    #Count total and trainable parameters in a model.
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 60)
    print("Model Parameter Count")
    print("=" * 60)
    print(f"Total parameters:      {total_params:,}")
    print(f"Trainable parameters:  {trainable_params:,}")
    print(f"Non-trainable params:  {total_params - trainable_params:,}")
    print("=" * 60)

    if verbose:
        print("\nDetailed breakdown by module:")
        print("-" * 60)
        for name, module in model.named_children():
            module_params = sum(p.numel() for p in module.parameters())
            print(f"{name:20s}: {module_params:,} parameters")
        print("-" * 60)

    return total_params, trainable_params


def main():
    parser = argparse.ArgumentParser(description='Count parameters in DETR checkpoint')
    parser.add_argument('--checkpoint', type=str, default='outputs/checkpoint.pth',
                        help='Path to checkpoint file')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed parameter breakdown by module')
    args = parser.parse_args()

    print(f"\nLoading checkpoint from: {args.checkpoint}")

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

    # Get saved args from checkpoint
    if 'args' not in checkpoint:
        print("Warning: 'args' not found in checkpoint. Using default values.")
        # You may need to manually set these if args are not in checkpoint
        from main import get_args_parser
        saved_args = get_args_parser().parse_args([])
    else:
        saved_args = checkpoint['args']

    print(f"Model configuration:")
    print(f"  - Backbone: {saved_args.backbone}")
    print(f"  - Num queries: {saved_args.num_queries}")
    print(f"  - Hidden dim: {saved_args.hidden_dim}")
    print(f"  - Encoder layers: {saved_args.enc_layers}")
    print(f"  - Decoder layers: {saved_args.dec_layers}")
    if hasattr(saved_args, 'use_dn'):
        print(f"  - DN-DETR enabled: {saved_args.use_dn}")

    # Build model
    model, _, _ = build(saved_args)

    # Load state dict
    model.load_state_dict(checkpoint['model'], strict=False)

    # Count parameters
    print()
    total, trainable = count_parameters(model, verbose=args.verbose)

    # Print epoch info if available
    if 'epoch' in checkpoint:
        print(f"\nCheckpoint from epoch: {checkpoint['epoch']}")

    return total, trainable


if __name__ == '__main__':
    main()
