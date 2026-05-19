# Copyright (c) Ruopeng Gao. All Rights Reserved.
import torch
import torch.nn as nn
from typing import Optional, Tuple


class KalmanFilterWithUncertaintyModulation(nn.Module):
    """
    KPRR: Kalman-Prompted Residual Refinement
    内置了基于特征突变 (G_t) 的动态协方差调制机制。
    """
    
    def __init__(self, hidden_dim: int, Q_base: Optional[torch.Tensor] = None, R_base: Optional[torch.Tensor] = None,
                 beta: float = 2.0, gamma: float = 2.0):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        
        self.Q_base = nn.Parameter(Q_base if Q_base is not None else torch.ones(4) * 0.1)
        self.R_base = nn.Parameter(R_base if R_base is not None else torch.ones(4) * 0.5)
        
        # 创新点3：高维语义 -> 不确定性系数
        self.uncertainty_net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 8),
            nn.Sigmoid()
        )
        
        self.register_buffer('F', torch.eye(6))
        self.F[0, 4] = 1.0
        self.F[1, 5] = 1.0
        
        self.register_buffer('H', torch.zeros(4, 6))
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = self.H[3, 3] = 1.0
        
        self.track_states = {}
        
    def predict(self, track_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """KPRR 阶段1：动力学空间提示 (输出潜空间坐标)"""
        if track_id not in self.track_states:
            return None, None
        
        state, covariance = self.track_states[track_id]
        X_KF = self.F @ state
        bbox = self.H @ X_KF
        
        bbox_clamped = torch.clamp(bbox, min=1e-5, max=1-1e-5)
        P_ref = torch.log(bbox_clamped / (1 - bbox_clamped))  # Inverse Sigmoid
        return P_ref, X_KF
    
    def update(self, track_id: int, measurement: torch.Tensor, G_t: torch.Tensor) -> torch.Tensor:
        """KPRR 阶段3：语义驱动的不确定性调制"""
        if track_id not in self.track_states:
            state = torch.zeros(6, device=measurement.device)
            state[:4] = measurement
            covariance = torch.eye(6, device=measurement.device)
            self.track_states[track_id] = (state, covariance)
            return state
        
        state, covariance = self.track_states[track_id]
        
        # 确保 G_t 与 measurement 在同一设备
        if G_t.device != measurement.device:
            G_t = G_t.to(measurement.device)
        
        # 提取动态缩放因子
        modulation = self.uncertainty_net(G_t.unsqueeze(0).unsqueeze(-1))
        w_Q, w_R = modulation[:, :4], modulation[:, 4:]
        
        Q_t = torch.exp(self.beta * w_Q) * self.Q_base.unsqueeze(0).to(measurement.device)
        R_t = torch.exp(self.gamma * w_R) * self.R_base.unsqueeze(0).to(measurement.device)
        
        state_pred = self.F @ state
        covariance_pred = self.F @ covariance @ self.F.T
        covariance_pred[:4, :4] += torch.diag(Q_t.squeeze())
        
        innovation = measurement - self.H @ state_pred
        S = self.H @ covariance_pred @ self.H.T + torch.diag(R_t.squeeze())
        K = covariance_pred @ self.H.T @ torch.inverse(S)
        
        state_updated = state_pred + K @ innovation
        covariance_updated = (torch.eye(6, device=state.device) - K @ self.H) @ covariance_pred
        
        self.track_states[track_id] = (state_updated, covariance_updated)
        return state_updated
    
    def get_predicted_ref_pts(self, track_id: int) -> Optional[torch.Tensor]:
        """获取预测参考点，供 Deformable Attention 使用"""
        P_ref, _ = self.predict(track_id)
        return P_ref
    
    def reset_track(self, track_id: int):
        """重置单个目标状态"""
        if track_id in self.track_states:
            del self.track_states[track_id]
    
    def reset_all(self):
        """重置所有目标状态"""
        self.track_states.clear()
    
    def has_track(self, track_id: int) -> bool:
        """检查目标是否存在"""
        return track_id in self.track_states


def build_kalman_filter(config: dict) -> KalmanFilterWithUncertaintyModulation:
    """构建带不确定性调制的卡尔曼滤波器"""
    Q_base = config.get("KALMAN_Q_BASE", None)
    R_base = config.get("KALMAN_R_BASE", None)
    
    # 将列表转换为 Tensor
    if Q_base is not None and isinstance(Q_base, list):
        Q_base = torch.tensor(Q_base, dtype=torch.float32)
    if R_base is not None and isinstance(R_base, list):
        R_base = torch.tensor(R_base, dtype=torch.float32)
    
    return KalmanFilterWithUncertaintyModulation(
        hidden_dim=config["HIDDEN_DIM"],
        Q_base=Q_base,
        R_base=R_base,
        beta=config.get("KALMAN_BETA", 2.0),
        gamma=config.get("KALMAN_GAMMA", 2.0)
    )
