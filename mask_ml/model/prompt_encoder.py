# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Modified: added support for generic joint-embedding prompt tokens
# via the new `joint_tokens` argument in PromptEncoder.forward.

import numpy as np
import torch
from torch import nn
from typing import Any, Optional, Tuple, Type

from mask_ml.model.common_layers import LayerNorm2d  # keep your original import


class PositionEmbeddingRandom(nn.Module):
    """Random Fourier-feature positional encoding."""
    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None):
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        coords = 2 * coords - 1                       # [0,1] -> [-1,1]
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(0) - 0.5
        x_embed = grid.cumsum(1) - 0.5
        y_embed, x_embed = y_embed / h, x_embed / w
        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)                    # C × H × W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        coords = coords_input.clone()
        coords[..., 0] /= image_size[1]
        coords[..., 1] /= image_size[0]
        return self._pe_encoding(coords.float())      # B × N × C


# ──────────────────────────────────────────────────────────────────────────
# Prompt Encoder (now with joint_tokens)
# ──────────────────────────────────────────────────────────────────────────
class PromptEncoder(nn.Module):
    """
    Encodes five kinds of prompts:
       • points   (coords + FG/BG labels)
       • boxes
       • masks    (dense)
       • joint_tokens  ← NEW  (e.g. CLIP image/text embedding projected to 256-D)
       • internal padding “no-mask / not-a-point” tokens
    Output:
       sparse B × N × C     and     dense B × C × H′ × W′
    """
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size

        # positional-encoding module
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        # point / box type tokens
        self.num_point_embeddings = 4                # FG, BG, corner1, corner2
        self.point_embeddings = nn.ModuleList(
            [nn.Embedding(1, embed_dim) for _ in range(self.num_point_embeddings)]
        )
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        # mask down-scaling conv tower
        self.mask_input_size = (4 * image_embedding_size[0], 4 * image_embedding_size[1])
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, 2, 2),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, 2, 2),
            LayerNorm2d(mask_in_chans),
            activation(),
            nn.Conv2d(mask_in_chans, embed_dim, 1),
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    # ───────────────────── helper methods ────────────────────
    def get_dense_pe(self) -> torch.Tensor:
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(self, pts, lbl, pad) -> torch.Tensor:
        pts = pts + 0.5
        if pad:
            pad_pt = torch.zeros((pts.shape[0], 1, 2), device=pts.device)
            pad_lb = -torch.ones((lbl.shape[0], 1), device=lbl.device)
            pts, lbl = torch.cat([pts, pad_pt], 1), torch.cat([lbl, pad_lb], 1)
        emb = self.pe_layer.forward_with_coords(pts, self.input_image_size)
        emb[lbl == -1] += self.not_a_point_embed.weight
        emb[lbl == 0] += self.point_embeddings[0].weight
        emb[lbl == 1] += self.point_embeddings[1].weight
        return emb

    def _embed_boxes(self, boxes) -> torch.Tensor:
        boxes = boxes + 0.5
        coords = boxes.reshape(-1, 2, 2)
        emb = self.pe_layer.forward_with_coords(coords, self.input_image_size)
        emb[:, 0] += self.point_embeddings[2].weight
        emb[:, 1] += self.point_embeddings[3].weight
        return emb

    def _embed_masks(self, m) -> torch.Tensor:
        return self.mask_downscaling(m)

    # ────────────────────── new public forward ───────────────────────────
    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
        *,
        joint_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args
        ----
        points : (coords, labels) or None
        boxes  : Tensor or None
        masks  : Tensor or None
        joint_tokens : Tensor [B,1,embed_dim] or None  ← NEW
                       Output of make_joint_prompt_tokens or analogous func.

        Returns
        -------
        sparse_embeddings : B × N × embed_dim
        dense_embeddings  : B × embed_dim × H′ × W′
        """
        # determine batch
        def _get_batch_size(
            points: Optional[Tuple[torch.Tensor, torch.Tensor]],
            boxes:  Optional[torch.Tensor],
            masks:  Optional[torch.Tensor],
            joint_tokens: Optional[torch.Tensor],
        ) -> int:
            """Infer batch size from whichever prompt tensor is present."""
            if points is not None:               # points = (coords, labels)
                return points[0].shape[0]        # coords shape → (B, N, 2)

            if boxes is not None:                # boxes shape → (B, 4)
                return boxes.shape[0]

            if masks is not None:                # masks shape → (B, 1, H, W)
                return masks.shape[0]

            if joint_tokens is not None:         # joint_tokens shape → (B, 1, C)
                return joint_tokens.shape[0]

            # fall-back when no prompts are given
            return 1
        bs = _get_batch_size(points, boxes, masks, joint_tokens)

        dev = self.point_embeddings[0].weight.device
        sparse = torch.empty((bs, 0, self.embed_dim), device=dev)

        # (0) optional joint token first
        if joint_tokens is not None:
            assert joint_tokens.shape == (bs, 1, self.embed_dim)
            sparse = torch.cat([joint_tokens.to(dev), sparse], dim=1)

        # (1) points
        if points is not None:
            coords, labels = points
            p_emb = self._embed_points(coords, labels, pad=(boxes is None))
            sparse = torch.cat([sparse, p_emb], dim=1)

        # (2) boxes
        if boxes is not None:
            b_emb = self._embed_boxes(boxes)
            sparse = torch.cat([sparse, b_emb], dim=1)

        # (3) dense mask embedding
        if masks is not None:
            dense = self._embed_masks(masks)
        else:
            dense = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        return sparse, dense
