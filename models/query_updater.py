# Copyright (c) Ruopeng Gao. All Rights Reserved.
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List
from .utils import pos_to_pos_embed, logits_to_scores
from torch.utils.checkpoint import checkpoint

from .ffn import FFN
from .mlp import MLP
from structures.track_instances import TrackInstances
from utils.utils import inverse_sigmoid
from utils.box_ops import box_cxcywh_to_xyxy, box_iou_union


class QueryUpdater(nn.Module):
    def __init__(self, hidden_dim: int, ffn_dim: int,
                 tp_drop_ratio: float, fp_insert_ratio: float,
                 dropout: float,
                 use_checkpoint: bool, use_dab: bool,
                 update_threshold: float, 
                 max_keyframes: int = 30, keyframe_sim_threshold: float = 0.9,
                 keyframe_conf_threshold: float = 0.5,
                 visualize: bool = False,
                 attention_temperature: float = 1.0):  # 注意力温度系数
        super(QueryUpdater, self).__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.tp_drop_ratio = tp_drop_ratio
        self.fp_insert_ratio = fp_insert_ratio
        self.dropout = dropout

        self.use_checkpoint = use_checkpoint
        self.use_dab = use_dab
        self.visualize = visualize

        self.update_threshold = update_threshold
        
        # 创新点：多样性驱动的动态关键帧库参数
        self.max_keyframes = max_keyframes
        self.keyframe_sim_threshold = keyframe_sim_threshold
        self.keyframe_conf_threshold = keyframe_conf_threshold
        
        # 注意力融合参数
        self.attention_temperature = attention_temperature

        self.confidence_weight_net = nn.Sequential(
            MLP(input_dim=self.hidden_dim, hidden_dim=self.hidden_dim, output_dim=self.hidden_dim, num_layers=2),
            nn.Sigmoid()
        )
        self.short_memory_fusion = MLP(input_dim=2*self.hidden_dim, hidden_dim=2*self.hidden_dim,
                                       output_dim=self.hidden_dim, num_layers=2)
        self.memory_attn = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=8, batch_first=True)
        self.memory_dropout = nn.Dropout(self.dropout)
        self.memory_norm = nn.LayerNorm(self.hidden_dim)
        self.memory_ffn = FFN(d_model=self.hidden_dim, d_ffn=self.ffn_dim, dropout=self.dropout)
        self.query_feat_dropout = nn.Dropout(self.dropout)
        self.query_feat_norm = nn.LayerNorm(self.hidden_dim)
        self.query_feat_ffn = FFN(d_model=self.hidden_dim, d_ffn=self.ffn_dim, dropout=self.dropout)
        self.query_pos_head = MLP(
            input_dim=self.hidden_dim*2,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            num_layers=2
        )

        if self.use_dab is False:
            self.linear_pos1 = nn.Linear(256, 256)
            self.linear_pos2 = nn.Linear(256, 256)
            self.norm_pos = nn.LayerNorm(256)
            self.activation = nn.ReLU(inplace=True)

        # 创新点：通道级解耦门控网络 (Non-Linear Bottleneck)
        # 用于生成特征突变度 G_t
        self.channel_gate = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim // 2), 
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim),     
            nn.Sigmoid()                                          
        )
        
        # 卡尔曼滤波器实例（由外层 MeMOTR 注入）
        self.kalman_filter = None

        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self,
                previous_tracks: List[TrackInstances],
                new_tracks: List[TrackInstances],
                unmatched_dets: List[TrackInstances] | None,
                no_augment: bool = False):
        tracks = self.select_active_tracks(previous_tracks, new_tracks, unmatched_dets, no_augment=no_augment)
        tracks = self.update_tracks_embedding(tracks=tracks)
        return tracks

    def _fuse_keyframes(self, keyframes, query=None):
        """
        关键帧融合：使用注意力机制进行加权融合
        
        Args:
            keyframes: list of tensors, 每个元素形状为 (C,)
            query: 当前查询特征，形状为 (C,)，用于计算注意力权重
        
        Returns:
            fused_feature: 融合后的特征，形状为 (C,)
        """
        if len(keyframes) == 0:
            return None
        
        keyframes_tensor = torch.stack(keyframes)  # (N, C)
        
        # 如果没有提供查询，默认用最新帧作为查询
        if query is None:
            query = keyframes_tensor[-1]
        
        # 1. 计算余弦相似度
        similarities = F.cosine_similarity(keyframes_tensor, query, dim=-1)  # (N,)
        
        # 2. Softmax 归一化（带温度系数）
        weights = F.softmax(similarities / self.attention_temperature, dim=0)  # (N,)
        
        # 3. 加权融合
        fused = torch.sum(keyframes_tensor * weights.unsqueeze(-1), dim=0)  # (C,)
        
        return fused

    def _update_keyframes(self, keyframes, new_embed, conf_score):
        """多样性驱动的动态关键帧库准入机制"""
        # 漏斗一：置信度截断（拒收严重遮挡或模糊的特征）
        if conf_score < self.keyframe_conf_threshold:
            return keyframes

        if len(keyframes) == 0:
            return [new_embed.detach().clone()]

        # 向量化余弦相似度计算
        kf_tensor = torch.stack(keyframes)
        sims = F.cosine_similarity(new_embed.unsqueeze(0), kf_tensor, dim=-1)

        # 漏斗二：冗余排斥逻辑 (Extremum-based Pruning)
        # 只要库里已有的 任意一帧 与当前帧高度相似，就判定为冗余，直接拒收
        if torch.max(sims) > self.keyframe_sim_threshold:
            return keyframes

        # 准许入库
        keyframes.append(new_embed.detach().clone())
        
        # 漏斗三：维持动态平衡
        if len(keyframes) > self.max_keyframes:
            keyframes = keyframes[-self.max_keyframes:]
        
        return keyframes

    def update_tracks_embedding(self, tracks: List[TrackInstances]):
        for b in range(len(tracks)):
            scores = torch.max(logits_to_scores(logits=tracks[b].logits), dim=1).values
            is_pos = scores > self.update_threshold
            
            if self.use_dab:
                tracks[b].ref_pts[is_pos] = inverse_sigmoid(tracks[b][is_pos].boxes.detach().clone())
            else:
                tracks[b].ref_pts[is_pos] = inverse_sigmoid(tracks[b][is_pos].boxes.detach().clone())

            query_pos = pos_to_pos_embed(tracks[b].ref_pts.sigmoid(), num_pos_feats=self.hidden_dim//2)
            output_embed = tracks[b].output_embed
            last_output_embed = tracks[b].last_output
            long_memory = tracks[b].long_memory.detach()

            confidence_weight = self.confidence_weight_net(output_embed)
            short_memory = self.short_memory_fusion(
                torch.cat((confidence_weight * output_embed, last_output_embed), dim=-1)
            )

            query_pos = self.query_pos_head(query_pos)
            q = short_memory + query_pos
            k = long_memory + query_pos
            tgt = output_embed
            tgt2 = self.memory_attn(q[None, :], k[None, :], tgt[None, :])[0][0, :]
            tgt = tgt + self.memory_dropout(tgt2)
            tgt = self.memory_norm(tgt)
            tgt = self.memory_ffn(tgt)
            query_feat = long_memory + self.query_feat_dropout(tgt)
            query_feat = self.query_feat_norm(query_feat)
            query_feat = self.query_feat_ffn(query_feat)

            # --- 【极致性能优化：Batched 动态门控与平滑】 ---
            fused_keyframes_list = []
            
            for i in range(len(tracks[b])):
                if hasattr(tracks[b], 'keyframes'):
                    tracks[b].keyframes[i] = self._update_keyframes(
                        tracks[b].keyframes[i], 
                        output_embed[i], 
                        scores[i]
                    )
                else:
                    tracks[b].keyframes = [[output_embed[i].detach().clone()]]
                
                # 使用注意力融合，传入当前帧特征作为查询
                keyframe_fused = self._fuse_keyframes(tracks[b].keyframes[i], query=output_embed[i])
                fused_keyframes_list.append(keyframe_fused if keyframe_fused is not None else output_embed[i].detach().clone())

            # 矩阵化梭哈：提取特征突变度 G_t
            fused_keyframes_batch = torch.stack(fused_keyframes_list)
            memory_input = torch.cat((fused_keyframes_batch, output_embed), dim=-1)
            G_t = self.channel_gate(memory_input)
            
            # 张量魔法：一次性完成所有目标 Hadamard 细粒度记忆平滑计算
            long_memory = (1.0 - G_t) * fused_keyframes_batch + G_t * output_embed
            # --- 优化结束 ---

            tracks[b].long_memory = tracks[b].long_memory * ~is_pos.reshape((is_pos.shape[0], 1)) + \
                                    long_memory * is_pos.reshape((is_pos.shape[0], 1))

            # --- 【KPRR 时序交互闭环】 ---
            for i in range(len(tracks[b])):
                if tracks[b].ids[i] >= 0:
                    track_id = tracks[b].ids[i].item()
                    
                    if self.kalman_filter is not None:
                        # 1. (KPRR 阶段3): 用当前观测坐标 + 特征突变 G_t，更新卡尔曼的后验状态
                        if is_pos[i]:
                            measurement = tracks[b].boxes[i, :4]
                            self.kalman_filter.update(track_id, measurement, G_t[i])
                        
                        # 2. (KPRR 阶段1): 预测下一帧！将卡尔曼的物理预测直接注入到 ref_pts 中
                        if is_pos[i]:
                            kf_ref_pts, _ = self.kalman_filter.predict(track_id)
                            if kf_ref_pts is not None:
                                tracks[b].ref_pts[i] = kf_ref_pts.detach()
                            else:
                                tracks[b].ref_pts[i] = inverse_sigmoid(tracks[b].boxes[i].detach().clone())
                    else:
                        # 原版兜底逻辑
                        if is_pos[i]:
                            tracks[b].ref_pts[i] = inverse_sigmoid(tracks[b].boxes[i].detach().clone())

            tracks[b].last_output = tracks[b].last_output * ~is_pos.reshape((is_pos.shape[0], 1)) + \
                                    output_embed * is_pos.reshape((is_pos.shape[0], 1))

            if self.use_dab:
                tracks[b].query_embed[is_pos] = query_feat[is_pos]
            else:
                tracks[b].query_embed[:, self.hidden_dim:][is_pos] = query_feat[is_pos]
                new_query_pos = self.linear_pos2(self.activation(self.linear_pos1(output_embed)))
                query_pos = tracks[b].query_embed[:, :self.hidden_dim]
                query_pos = query_pos + new_query_pos
                query_pos = self.norm_pos(query_pos)
                tracks[b].query_embed[:, :self.hidden_dim][is_pos] = query_pos[is_pos]

        return tracks

    def select_active_tracks(self, previous_tracks: List[TrackInstances],
                             new_tracks: List[TrackInstances],
                             unmatched_dets: List[TrackInstances],
                             no_augment: bool = False):
        tracks = []
        if self.training:
            for b in range(len(new_tracks)):
                new_tracks[b].last_output = new_tracks[b].output_embed
                if self.use_dab:
                    new_tracks[b].long_memory = new_tracks[b].query_embed
                else:
                    new_tracks[b].long_memory = new_tracks[b].query_embed[:, self.hidden_dim:]
                unmatched_dets[b].last_output = unmatched_dets[b].output_embed
                if self.use_dab:
                    unmatched_dets[b].long_memory = unmatched_dets[b].query_embed
                else:
                    unmatched_dets[b].long_memory = unmatched_dets[b].query_embed[:, self.hidden_dim:]
                
                new_tracks[b].keyframes = [[emb.detach().clone()] for emb in new_tracks[b].output_embed]
                unmatched_dets[b].keyframes = [[emb.detach().clone()] for emb in unmatched_dets[b].output_embed]

                if self.tp_drop_ratio == 0.0 and self.fp_insert_ratio == 0.0:
                    active_tracks = TrackInstances.cat_tracked_instances(previous_tracks[b], new_tracks[b])
                
                tracks.append(active_tracks)
            
            return tracks


def build(config: dict):
    """构建 QueryUpdater"""
    return QueryUpdater(
        hidden_dim=config["HIDDEN_DIM"],
        ffn_dim=config["FFN_DIM"],
        tp_drop_ratio=config.get("TP_DROP_RATIO", 0.0),
        fp_insert_ratio=config.get("FP_INSERT_RATIO", 0.0),
        dropout=config["DROPOUT"],
        use_checkpoint=config["USE_CHECKPOINT"],
        use_dab=config["USE_DAB"],
        update_threshold=config.get("UPDATE_THRESHOLD", 0.5),
        max_keyframes=config.get("MAX_KEYFRAMES", 30),
        keyframe_sim_threshold=config.get("KEYFRAME_SIM_THRESHOLD", 0.9),
        keyframe_conf_threshold=config.get("KEYFRAME_CONF_THRESHOLD", 0.5),
        visualize=config["VISUALIZE"],
        attention_temperature=config.get("ATTENTION_TEMPERATURE", 1.0)
    )