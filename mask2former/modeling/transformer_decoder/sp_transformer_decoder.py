# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by [Your Name] from mask2former_transformer_decoder.py
import logging
import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F
import numpy as np

from detectron2.config import configurable
from detectron2.layers import Conv2d

from .position_encoding import PositionEmbeddingSine
from .maskformer_transformer_decoder import TRANSFORMER_DECODER_REGISTRY
from .mask2former_transformer_decoder import (
    SelfAttentionLayer,
    CrossAttentionLayer, 
    FFNLayer,
    MLP,
    _get_activation_fn
)

# Import superpixel functionality
import sys
import os
sys.path.append('/home/likai/code/Segment/Mask2Former')
from super_pixel.superpixel import SuperpixelExtractor


@TRANSFORMER_DECODER_REGISTRY.register()
class SuperPixelQueryTransformerDecoder(nn.Module):
    """
    Transformer decoder with SuperPixel-enhanced learnable queries.
    Generates superpixels internally and combines with learnable queries.
    """

    _version = 2

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "static_query" in k:
                    newk = k.replace("static_query", "query_feat")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    @configurable
    def __init__(
        self,
        in_channels,
        mask_classification=True,
        *,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,
        # SuperPixel-specific parameters
        superpixel_fusion_dim: int = 128,
        superpixel_feature_type: str = "id_embedding",
        fusion_strategy: str = "concat_mlp",
        # SuperPixel generation parameters
        superpixel_algorithm: str = "slic",
        superpixel_n_segments: int = 100,
        superpixel_compactness: int = 10,
        superpixel_sigma: int = 1,
        enable_superpixel: bool = True,
    ):
        """
        Args:
            superpixel_fusion_dim: dimension for superpixel feature processing
            superpixel_feature_type: type of superpixel features to extract
            fusion_strategy: how to fuse superpixel features with learnable queries
            superpixel_algorithm: algorithm for superpixel generation
            superpixel_n_segments: number of superpixel segments
            superpixel_compactness: compactness parameter for SLIC
            superpixel_sigma: sigma parameter for SLIC
            enable_superpixel: whether to enable superpixel enhancement
        """
        super().__init__()

        self.mask_classification = mask_classification
        self.nheads = nheads  # 添加这行
        
        # transformer self-attention layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(dec_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.num_layers = dec_layers

        self.num_queries = num_queries
        
        # Original learnable query features and embeddings
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # SuperPixel components
        self.enable_superpixel = enable_superpixel
        self.superpixel_fusion_dim = superpixel_fusion_dim
        self.superpixel_feature_type = superpixel_feature_type
        self.fusion_strategy = fusion_strategy
        
        # SuperPixel generation parameters
        self.superpixel_algorithm = superpixel_algorithm
        self.superpixel_params = {
            "n_segments": superpixel_n_segments,
            "compactness": superpixel_compactness,
            "sigma": superpixel_sigma,
            "start_label": 0,
            "min_size_factor": 0.5,
            "max_num_iter": 10,
            "enforce_connectivity": True,
        }
        
        # Initialize superpixel extractor
        if self.enable_superpixel:
            self.superpixel_extractor = SuperpixelExtractor(self.superpixel_algorithm)
            self._build_superpixel_fusion_modules(hidden_dim)

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # Position encoding
        self.pe_layer = PositionEmbeddingSine(hidden_dim // 2, normalize=True)

        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

    def _build_superpixel_fusion_modules(self, hidden_dim):
        """Build superpixel feature extraction and fusion modules"""
        
        # Superpixel feature extraction
        if self.superpixel_feature_type == "id_embedding":
            # Simple ID embedding
            self.superpixel_mlp = nn.Sequential(
                nn.Linear(1, self.superpixel_fusion_dim),
                nn.ReLU(),
                nn.Linear(self.superpixel_fusion_dim, self.superpixel_fusion_dim),
                nn.LayerNorm(self.superpixel_fusion_dim)
            )
        elif self.superpixel_feature_type == "spatial":
            # Spatial features (centroid, area, etc.)
            self.superpixel_mlp = nn.Sequential(
                nn.Linear(4, self.superpixel_fusion_dim),  # [cx, cy, area, aspect_ratio]
                nn.ReLU(),
                nn.Linear(self.superpixel_fusion_dim, self.superpixel_fusion_dim),
                nn.LayerNorm(self.superpixel_fusion_dim)
            )
        
        # Query-SuperPixel fusion
        if self.fusion_strategy == "concat_mlp":
            self.query_fusion = nn.Sequential(
                nn.Linear(hidden_dim + self.superpixel_fusion_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim)
            )
        elif self.fusion_strategy == "attention":
            self.fusion_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )
            self.sp_proj = nn.Linear(self.superpixel_fusion_dim, hidden_dim)
        elif self.fusion_strategy == "add":
            self.sp_proj = nn.Linear(self.superpixel_fusion_dim, hidden_dim)

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification
        
        ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD

        assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
        ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
        ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
        ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ
        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM

        # SuperPixel-specific configs
        ret["superpixel_fusion_dim"] = cfg.MODEL.MASK_FORMER.get("SUPERPIXEL_FUSION_DIM", 128)
        ret["superpixel_feature_type"] = cfg.MODEL.MASK_FORMER.get("SUPERPIXEL_FEATURE_TYPE", "id_embedding")
        ret["fusion_strategy"] = cfg.MODEL.MASK_FORMER.get("FUSION_STRATEGY", "concat_mlp")
        ret["superpixel_algorithm"] = cfg.MODEL.MASK_FORMER.get("SUPERPIXEL_ALGORITHM", "slic")
        ret["superpixel_n_segments"] = cfg.MODEL.MASK_FORMER.get("SUPERPIXEL_N_SEGMENTS", 100)
        ret["superpixel_compactness"] = cfg.MODEL.MASK_FORMER.get("SUPERPIXEL_COMPACTNESS", 10)
        ret["superpixel_sigma"] = cfg.MODEL.MASK_FORMER.get("SUPERPIXEL_SIGMA", 1)
        ret["enable_superpixel"] = cfg.MODEL.MASK_FORMER.get("ENABLE_SUPERPIXEL", True)

        return ret

    def _generate_superpixels(self, images):
        """Generate superpixels from input images"""
        # Convert tensor to numpy for superpixel generation
        # images: [B, C, H, W] tensor
        batch_superpixels = []
        
        for i in range(images.shape[0]):
            # Convert single image: [C, H, W] -> [H, W, C]
            img = images[i].permute(1, 2, 0).cpu().numpy()
            
            # Normalize to [0, 1] if needed
            if img.max() > 1.0:
                img = img / 255.0
            
            # Convert to tensor format for superpixel extractor: [1, C, H, W]
            img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
            
            # Generate superpixels
            try:
                _, _, _, assigned_masks = self.superpixel_extractor(img_tensor, self.superpixel_params)
                superpixel_labels = assigned_masks[0].numpy().astype(np.int32)
                batch_superpixels.append(superpixel_labels)
            except Exception as e:
                print(f"Superpixel generation failed: {e}")
                # Fallback: create dummy superpixels
                H, W = img.shape[:2]
                dummy_labels = np.zeros((H, W), dtype=np.int32)
                batch_superpixels.append(dummy_labels)
        
        # Convert back to tensor
        superpixels_tensor = torch.from_numpy(np.stack(batch_superpixels, axis=0)).to(images.device)
        return superpixels_tensor

    def forward(self, x, mask_features, mask=None, images=None):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        base_query_feat = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        
        # Generate and fuse superpixel information with learnable queries
        if self.enable_superpixel and images is not None:
            # Generate superpixels from input images
            superpixels = self._generate_superpixels(images)
            output = self._fuse_superpixel_query(base_query_feat, superpixels, size_list[0])
        else:
            output = base_query_feat

        predictions_class = []
        predictions_mask = []

        # prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[0])
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )
            
            # FFN
            output = self.transformer_ffn_layers[i](output)

            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        assert len(predictions_class) == self.num_layers + 1

        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask
            )
        }
        return out

    def _fuse_superpixel_query(self, base_query, superpixels, target_size):
        """Fuse superpixel information with learnable queries"""
        bs = base_query.shape[1]
        
        # Resize superpixels to target size
        superpixels_resized = F.interpolate(
            superpixels.float().unsqueeze(1), 
            size=target_size, 
            mode='nearest'
        ).squeeze(1)  # [bs, H, W]
        
        # Extract superpixel features for each query
        enhanced_queries = []
        for i in range(self.num_queries):
            # Extract superpixel features for query i
            sp_features = self._extract_superpixel_features(superpixels_resized, i)  # [bs, feature_dim]
            
            # Process through superpixel MLP
            sp_embed = self.superpixel_mlp(sp_features)  # [bs, superpixel_fusion_dim]
            
            # Fuse with original query
            query_i = base_query[i]  # [bs, hidden_dim]
            enhanced_query = self._fuse_features(query_i, sp_embed)  # [bs, hidden_dim]
            
            enhanced_queries.append(enhanced_query)
        
        return torch.stack(enhanced_queries, dim=0)  # [num_queries, bs, hidden_dim]

    def _extract_superpixel_features(self, superpixels, query_idx):
        """Extract superpixel features for a specific query"""
        bs, H, W = superpixels.shape
        
        if self.superpixel_feature_type == "id_embedding":
            # Simple strategy: use query index to select superpixel
            max_sp_id = superpixels.max().item()
            if max_sp_id > 0:
                target_sp_id = query_idx % (max_sp_id + 1)
                # Normalize superpixel ID
                sp_features = torch.full((bs, 1), target_sp_id / max_sp_id, 
                                       device=superpixels.device, dtype=torch.float32)
            else:
                sp_features = torch.zeros((bs, 1), device=superpixels.device, dtype=torch.float32)
                
        elif self.superpixel_feature_type == "spatial":
            # Extract spatial features (centroid, area, etc.)
            sp_features = []
            for b in range(bs):
                max_sp_id = superpixels[b].max().item()
                if max_sp_id > 0:
                    target_sp_id = query_idx % (max_sp_id + 1)
                    mask = (superpixels[b] == target_sp_id).float()
                    
                    if mask.sum() > 0:
                        # Calculate spatial features
                        y_coords, x_coords = torch.meshgrid(
                            torch.arange(H, device=superpixels.device),
                            torch.arange(W, device=superpixels.device),
                            indexing='ij'
                        )
                        
                        # Centroid
                        cx = (x_coords * mask).sum() / mask.sum() / W  # normalized
                        cy = (y_coords * mask).sum() / mask.sum() / H  # normalized
                        
                        # Area (normalized)
                        area = mask.sum() / (H * W)
                        
                        # Aspect ratio approximation
                        x_span = (x_coords * mask).max() - (x_coords * mask).min() + 1
                        y_span = (y_coords * mask).max() - (y_coords * mask).min() + 1
                        aspect_ratio = x_span / (y_span + 1e-6)
                        
                        features = torch.tensor([cx, cy, area, aspect_ratio], device=superpixels.device)
                    else:
                        features = torch.zeros(4, device=superpixels.device)
                else:
                    features = torch.zeros(4, device=superpixels.device)
                
                sp_features.append(features)
            
            sp_features = torch.stack(sp_features, dim=0)  # [bs, 4]
        
        return sp_features

    def _fuse_features(self, query_feat, sp_feat):
        """Fuse query features with superpixel features"""
        if self.fusion_strategy == "concat_mlp":
            # Concatenate and pass through MLP
            fused_features = torch.cat([query_feat, sp_feat], dim=-1)
            return self.query_fusion(fused_features)
            
        elif self.fusion_strategy == "add":
            # Project superpixel features and add
            sp_proj = self.sp_proj(sp_feat)
            return query_feat + sp_proj
            
        elif self.fusion_strategy == "attention":
            # Use attention to fuse features
            sp_proj = self.sp_proj(sp_feat).unsqueeze(1)  # [bs, 1, hidden_dim]
            query_expanded = query_feat.unsqueeze(1)  # [bs, 1, hidden_dim]
            
            fused, _ = self.fusion_attention(query_expanded, sp_proj, sp_proj)
            return fused.squeeze(1)  # [bs, hidden_dim]
        
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        """Forward prediction heads"""
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.nheads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]
