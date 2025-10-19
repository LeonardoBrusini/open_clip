"""
This script tests the StableRepPlusLoss class from loss.py in a simulated
multi-GPU environment. It does not require actual GPUs to run.

Key simulation aspects:
- Simulates a world size of 4 GPUs.
- Mocks the torch.distributed communication functions (`gather_features` and
  `concat_all_gather`) to simulate tensor gathering across devices.
- Initializes StableRepPlusLoss with `local_loss=True`.
- Iterates through each simulated GPU (rank), calculating the loss on its
  local data slice.
- Prints detailed information at each step, including the shapes and values
  of local features, gathered features, logits, ground truth tensors, and losses.
"""
import torch
import loss as loss_module

def run_test():
    """
    Tests the StableRepPlusLoss class in a simulated 4-GPU environment
    with local_loss=True.
    """
    # Set torch print options to prevent truncation and wrapping
    torch.set_printoptions(profile="full", linewidth=2000)

    # --- Simulation Parameters ---
    world_size = 4
    local_batch_size = 4  # Number of captions per GPU
    m = 4                 # Number of images per caption
    feature_dim = 128     # Dimensionality of features
    total_captions = local_batch_size * world_size
    total_images = total_captions * m
    device = torch.device("cpu")

    print("--- Test Setup ---")
    print(f"Simulating {world_size} GPUs.")
    print(f"Local batch size (captions): {local_batch_size}")
    print(f"Images per caption (m): {m}")
    print(f"Total captions: {total_captions}")
    print(f"Total images: {total_images}")
    print(f"Feature dimension: {feature_dim}")
    print("-" * 20)

    # --- Mock Data ---
    # Create global tensors that represent the data across all simulated GPUs.
    torch.manual_seed(42)
    all_image_features = torch.randn(total_images, feature_dim, device=device)
    all_text_features = torch.randn(total_captions, feature_dim, device=device)
    # For this test, StableRep's image_embeddings can be the same as image_features.
    all_image_embeddings = all_image_features
    logit_scale = torch.tensor(1.0, device=device)

    print("\n--- Global (Gathered) Tensors ---")
    print(f"all_image_features shape: {all_image_features.shape}")
    print(f"all_text_features shape: {all_text_features.shape}")
    print(f"all_image_embeddings shape: {all_image_embeddings.shape}")


    # --- Monkey-patching torch.distributed functions ---
    # We replace the distributed communication functions with mocks that simulate
    # the gathering of tensors across the 4 GPUs.

    original_gather_features = loss_module.gather_features
    original_concat_all_gather = loss_module.concat_all_gather

    def mock_gather_features(
        image_features, text_features=None, local_loss=False,
        gather_with_grad=False, rank=0, world_size=1, use_horovod=False
    ):
        # This mock simulates the gathering of features from all GPUs.
        # It returns the pre-defined global tensors.
        if text_features is not None:
            return all_image_features, all_text_features
        return all_image_features, None

    def mock_concat_all_gather(tensor: torch.Tensor) -> torch.Tensor:
        # This mock simulates gathering index tensors. It reconstructs the
        # global index tensor based on the shape of the local input tensor.
        if tensor.dim() == 0: # handle single value tensors if any
            tensor = tensor.unsqueeze(0)

        # Used in MultiCLIPLoss.get_ground_truth for caption indices
        if tensor.shape[0] == local_batch_size:
            return torch.arange(total_captions, device=tensor.device)

        # Used in MultiCLIPLoss.get_ground_truth for image indices
        # and in StableRepPlusLoss.get_stablerep_ground_truth for local indices
        if tensor.shape[0] == local_batch_size * m:
            # Reconstruct the full gathered list of indices from all ranks
            all_indices = []
            for r in range(world_size):
                # This logic mirrors the creation of `local_idx` in the loss function
                start_index = r * (local_batch_size * m)
                rank_indices = torch.arange(local_batch_size * m, device=tensor.device) + start_index
                all_indices.append(rank_indices)
            return torch.cat(all_indices)

        raise ValueError(f"Unexpected tensor shape in mock_concat_all_gather: {tensor.shape}")

    loss_module.gather_features = mock_gather_features
    loss_module.concat_all_gather = mock_concat_all_gather

    # --- Per-GPU Simulation Loop ---
    total_multiclip_loss = 0
    total_stablerep_loss = 0

    for rank in range(world_size):
        print(f"\n{'='*25} GPU {rank} {'='*25}")

        loss_fn = loss_module.StableRepPlusLoss(
            m=m,
            local_loss=True,
            gather_with_grad=True,
            cache_labels=False,  # Disable cache for test clarity
            rank=rank,
            world_size=world_size,
        )

        # --- Get local data slice for this rank ---
        caption_start, caption_end = rank * local_batch_size, (rank + 1) * local_batch_size
        image_start, image_end = rank * local_batch_size * m, (rank + 1) * local_batch_size * m

        local_image_features = all_image_features[image_start:image_end]
        local_text_features = all_text_features[caption_start:caption_end]
        local_image_embeddings = all_image_embeddings[image_start:image_end]

        print("--- Local Features ---")
        print(f"local_image_features shape: {local_image_features.shape}")
        print(f"local_text_features shape: {local_text_features.shape}")
        print(f"local_image_embeddings shape: {local_image_embeddings.shape}")

        # --- Inspect Intermediate Values ---

        # 1. MultiCLIP part
        print("\n--- MultiCLIP Loss Internals ---")
        mc_logits_per_image, mc_logits_per_text = loss_fn.get_logits(
            local_image_features, local_text_features, logit_scale
        )
        print(f"Logits per image shape: {mc_logits_per_image.shape}")
        print(f"Logits per text shape: {mc_logits_per_text.shape}")

        gt_t2i, gt_i2t = loss_fn.get_ground_truth(device, local_text_features.shape[0])
        print(f"Ground truth T2I shape: {gt_t2i.shape}")
        print(f"Ground truth T2I values:\n{gt_t2i}")
        print(f"Ground truth I2T shape: {gt_i2t.shape}")
        print(f"Ground truth I2T values:\n{gt_i2t}")

        # 2. StableRep part
        print("\n--- StableRep Loss Internals ---")
        sr_logits = loss_fn.get_stablerep_logits(
            image_features=local_image_embeddings
        )
        print(f"StableRep logits shape: {sr_logits.shape}")

        sr_gt = loss_fn.get_stablerep_ground_truth(device, sr_logits.shape[0])
        print(f"StableRep ground truth shape: {sr_gt.shape}")
        print(f"StableRep ground truth values:\n{sr_gt}")

        # --- Calculate and Print Losses ---
        print("\n--- Loss Calculation ---")
        multiclip_loss, stablerep_loss = loss_fn(
            image_features=local_image_features,
            text_features=local_text_features,
            logit_scale=logit_scale,
            image_embeddings=local_image_embeddings,
        )
        print(f"Intermediate MultiCLIP Loss: {multiclip_loss.item():.4f}")
        print(f"Intermediate StableRep Loss: {stablerep_loss.item():.4f}")

        total_loss = multiclip_loss + stablerep_loss
        print(f"Total Loss on GPU {rank}: {total_loss.item():.4f}")

        total_multiclip_loss += multiclip_loss.item()
        total_stablerep_loss += stablerep_loss.item()

    # --- Final Aggregation ---
    print(f"\n{'='*60}")
    print("--- Final Aggregated Results ---")
    avg_multiclip_loss = total_multiclip_loss / world_size
    avg_stablerep_loss = total_stablerep_loss / world_size
    avg_total_loss = avg_multiclip_loss + avg_stablerep_loss
    print(f"Average MultiCLIP Loss across GPUs: {avg_multiclip_loss:.4f}")
    print(f"Average StableRep Loss across GPUs: {avg_stablerep_loss:.4f}")
    print(f"Average Total Loss across GPUs: {avg_total_loss:.4f}")

    # --- Restore original functions ---
    loss_module.gather_features = original_gather_features
    loss_module.concat_all_gather = original_concat_all_gather
    print("\nTest finished. Original functions restored.")


if __name__ == "__main__":
    run_test()
