# ===============================================
# ESVM-Omega v2.0 - Full MVP Implementation
# Extended from ESVM-Pro
# Author: ZUREON Lab
# ===============================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeStabilityLayer(nn.Module):
    """
    時系列安定性層（TSL）
    - 入力: [T, 4] （H1〜H4の系列）
    - Conv1d で軽く時間平滑化
    """
    def __init__(self, kernel_size: int = 5):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(4, 4, kernel_size=kernel_size,
                              padding=padding, bias=False)
        with torch.no_grad():
            self.conv.weight.fill_(1.0 / kernel_size)

    def forward(self, H_stack: torch.Tensor) -> torch.Tensor:
        # H_stack: [T, 4] -> [1, 4, T] -> Conv1d -> [T, 4]
        H = H_stack.transpose(0, 1).unsqueeze(0)   # [1,4,T]
        H_smooth = self.conv(H)                    # [1,4,T]
        return H_smooth.squeeze(0).transpose(0, 1) # [T,4]


class GaussianSmoothing1D(nn.Module):
    """
    1次元ガウシアン平滑（弱微分用）
    - 入力: [L] → 出力: [L]
    """
    def __init__(self, kernel_size: int = 7, sigma: float = 1.5):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size,
                              padding=padding, bias=False)

        # ガウシアンカーネルで初期化
        x = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel = kernel / kernel.sum()
        with torch.no_grad():
            self.conv.weight.copy_(kernel.view(1, 1, -1))

        for p in self.parameters():
            p.requires_grad = False  # 固定カーネル

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [L] → [1,1,L] → conv → [L]
        x_ = x.view(1, 1, -1)
        y = self.conv(x_)
        return y.view(-1)


class InconsistencyIndex(nn.Module):
    """
    不一致指数 II
    II = |H1-H2| + |H1-H3| + |H2-H3| + |H4-(H1+H2+H3)/3|
    """
    def __init__(self):
        super().__init__()

    def forward(self, H1, H2, H3, H4):
        avg_phys = (H1 + H2 + H3) / 3.0
        ii = (H1 - H2).abs() \
           + (H1 - H3).abs() \
           + (H2 - H3).abs() \
           + (H4 - avg_phys).abs()
        return ii.mean()


class SubjectiveTrustLayer(nn.Module):
    """
    主観信頼度 w(t)
    w = exp( - (diff^2) / (2 * sigma^2) )
    diff: 主観 vs 生理平均の差
    """
    def __init__(self, sigma: float = 0.25):
        super().__init__()
        self.sigma2 = sigma ** 2

    def forward(self, H4, H_phys_mean):
        diff = H4 - H_phys_mean
        w = torch.exp(- (diff ** 2) / (2 * self.sigma2))
        w = torch.clamp(w, 0.0, 1.0)
        return w.mean()


class FutureDriftPredictor(nn.Module):
    """
    Future Drift Predictor (FDP)
    - 直近の軌道 + 最終変化率 + 安定性 → 30秒後の負荷予測
    """
    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(6, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, traj_tail: torch.Tensor,
                last_dH_dt: torch.Tensor,
                stability: torch.Tensor):
        last_state = traj_tail[-1]           # [4]
        x = torch.cat([
            last_state.view(-1),             # 4
            last_dH_dt.view(-1),             # 1
            stability.view(-1)               # 1
        ], dim=0)                            # [6]
        h = torch.tanh(self.fc1(x))
        pred = torch.sigmoid(self.fc2(h))    # 0〜1
        return pred.squeeze()


class StateMachine5(nn.Module):
    """
    5状態分類機構
    0: Calm
    1: Normal
    2: Latent Stress
    3: Active Stress
    4: Critical
    """
    def __init__(self):
        super().__init__()
        self.th_calm = 0.20
        self.th_normal = 0.35
        self.th_latent = 0.55
        self.th_active = 0.75
        self.ii_latent_boost = 0.15

    def forward(self, H_final, II, stability):
        load = H_final
        adj_load = load + torch.clamp(II, 0, 1) * self.ii_latent_boost

        load_val = float(adj_load.item())
        ii_val = float(II.item())
        stab_val = float(stability.item())

        if load_val < self.th_calm:
            state = 0  # Calm
        elif load_val < self.th_normal:
            state = 1  # Normal
        elif load_val < self.th_latent:
            state = 2  # Latent Stress
        elif load_val < self.th_active:
            state = 3  # Active Stress
        else:
            if ii_val > 0.5 and stab_val < 0.3:
                state = 4
            else:
                state = 3

        latent_score = adj_load
        return state, latent_score


class HypothesisHead(nn.Module):
    """
    各仮説 H1〜H3 を出すための小さなヘッド
    入力: [T, F] → 出力: [T]（0〜1）
    """
    def __init__(self, in_dim: int, hidden_dim: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.fc1(z))
        out = torch.sigmoid(self.fc2(h)).squeeze(-1)  # [T]
        return out


class ESVM_Omega(nn.Module):
    """
    ESVM-Ω 完全統合版
    - ESVM-Pro の 8層構造を維持しつつ、
      ・非対称価値（損失回避）
      ・R(t) に基づくズレ D(t)
      ・主観時間密度 ρ(t)
      ・ガウシアン弱微分メタ認知コスト
      ・感情ダイナミクス ODE（E_dyn）
      ・ESVM-Ω 出力 V_omega
      を組み込んだ完全体。
    """
    def __init__(self,
                 physio_dim=4,
                 action_dim=4,
                 spectrum_dim=4,
                 window_sec=30,
                 fs=30,
                 future_sec=30):
        super().__init__()
        self.window = window_sec * fs
        self.future = future_sec * fs

        # 個人ベースライン
        self.register_buffer('B_phys', torch.zeros(physio_dim))
        self.register_buffer('MAD_phys', torch.ones(physio_dim))
        self.register_buffer('B_act', torch.zeros(action_dim))
        self.register_buffer('MAD_act', torch.ones(action_dim))
        self.register_buffer('B_spec', torch.zeros(spectrum_dim))
        self.register_buffer('MAD_spec', torch.ones(spectrum_dim))
        self.alpha = 0.001  # ベースライン追従速度

        # R(t) のスカラー版本（統合負荷の基準点）
        self.register_buffer('R_scalar', torch.zeros(1))

        # メタ認知コストの内部状態（減衰メモリ）
        self.register_buffer('C_m_state', torch.zeros(1))

        # 感情ダイナミクスの内部状態
        self.register_buffer('E_dyn', torch.zeros(1))

        # 仮説ヘッド
        self.h_phys = HypothesisHead(physio_dim)
        self.h_act = HypothesisHead(action_dim)
        self.h_spec = HypothesisHead(spectrum_dim)

        # 8層構造コンポーネント
        self.tsl = TimeStabilityLayer()
        self.ii = InconsistencyIndex()
        self.w_t = SubjectiveTrustLayer()
        self.fdp = FutureDriftPredictor()
        self.state_machine = StateMachine5()
        self.gauss = GaussianSmoothing1D(kernel_size=7, sigma=1.5)

        # ESVM-Ω ハイパーパラメータ（理論由来）
        # 非対称価値（プロスペクト的）
        self.a_plus = 1.0
        self.a_minus = 2.0
        self.p_plus = 1.0
        self.p_minus = 1.0
        self.b_vel = 1.0  # 速度重み

        # 主観時間
        self.kappa_time = 1.0
        self.gamma_time = 1.0

        # 感情帯域モデル
        self.alpha_band = 1.0
        self.B_max = 0.6
        self.beta_band = 2.0

        # メタ認知コスト
        self.lambda_meta = 1.0
        self.meta_decay = 0.99  # 忘却率（0.99 → 長期メモリ）

        # 感情ODE
        self.phi1 = 1.0
        self.phi2 = 0.5
        self.phi3 = 0.2
        self.phi4 = 0.8
        self.dt = 1.0 / fs  # 1ステップあたりの時間幅（秒）

    # ===== ベースライン更新 & ロバストz =====
    def update_baseline_phys(self, x_raw: torch.Tensor):
        self.B_phys = (1 - self.alpha) * self.B_phys + self.alpha * x_raw
        self.MAD_phys = (1 - self.alpha) * self.MAD_phys + self.alpha * (x_raw - self.B_phys).abs()

    def update_baseline_act(self, x_raw: torch.Tensor):
        self.B_act = (1 - self.alpha) * self.B_act + self.alpha * x_raw
        self.MAD_act = (1 - self.alpha) * self.MAD_act + self.alpha * (x_raw - self.B_act).abs()

    def update_baseline_spec(self, x_raw: torch.Tensor):
        self.B_spec = (1 - self.alpha) * self.B_spec + self.alpha * x_raw
        self.MAD_spec = (1 - self.alpha) * self.MAD_spec + self.alpha * (x_raw - self.B_spec).abs()

    def robust_z_phys(self, x: torch.Tensor):
        return (x - self.B_phys) / (1.4826 * self.MAD_phys + 1e-8)

    def robust_z_act(self, x: torch.Tensor):
        return (x - self.B_act) / (1.4826 * self.MAD_act + 1e-8)

    def robust_z_spec(self, x: torch.Tensor):
        return (x - self.B_spec) / (1.4826 * self.MAD_spec + 1e-8)

    # ===== main forward =====
    def forward(self, physio, action, spectrum, subjective):
        """
        physio    : [T, Fp]
        action    : [T, Fa]
        spectrum  : [T, Fs]
        subjective: [T] または [1] （0〜100）
        """
        T = physio.size(0)
        device = physio.device

        # 1) ロバストz変換
        z_p = self.robust_z_phys(physio)     # [T, Fp]
        z_a = self.robust_z_act(action)      # [T, Fa]
        z_s = self.robust_z_spec(spectrum)   # [T, Fs]

        # 2) 多仮説推定 H1〜H3
        H1 = self.h_phys(z_p)                # [T]
        H2 = self.h_act(z_a)                 # [T]
        H3 = self.h_spec(z_s)                # [T]

        # 主観 H4（0〜1）
        if subjective.numel() == 1:
            H4 = subjective.view(1).expand(T).to(device) / 100.0
        else:
            H4 = subjective.view(T).to(device) / 100.0

        # 3) 時系列層 TSL
        H_stack = torch.stack([H1, H2, H3, H4], dim=-1)  # [T,4]
        traj = self.tsl(H_stack)                         # [T,4]

        # 4) 変化率・安定性
        if T > 2:
            dH_dt = traj[1:] - traj[:-1]                # [T-1,4]
            dH_dt_abs = dH_dt.abs().mean(-1)            # [T-1]
            dH_dt_last = dH_dt_abs[-1]                  # ()
            d2H_dt2 = dH_dt[1:] - dH_dt[:-1]            # [T-2,4]
            _ = d2H_dt2.abs().mean(-1)
        else:
            dH_dt_last = torch.tensor(0.0, device=device)
            dH_dt_abs = torch.zeros(max(T-1, 1), device=device)

        stability = traj.std(0).mean()                  # ()

        # 5) 不一致指数 II
        II = self.ii(H1, H2, H3, H4)                    # ()

        # 6) 主観信頼度 w
        w = self.w_t(H4, (H1 + H2 + H3) / 3.0)          # ()

        # 7) 統合推定値（0〜1）
        H_final_series = w * H4 + (1 - w) * (H1 + H2 + H3) / 3.0  # [T]
        H_final = H_final_series.mean()                           # ()

        # 8) 未来予測 FDP
        tail_len = min(T, int(self.future))
        traj_tail = traj[-tail_len:]
        pred_30s = self.fdp(traj_tail, dH_dt_last, stability)

        # ======== ここから ESVM-Ω 理論パート ========

        # Deviation D(t) = H_final_series - R(t)（スカラー版）
        with torch.no_grad():
            # R_scalar を H_final にゆっくり追従させる（基準点）
            self.R_scalar.mul_(1.0 - self.alpha)
            self.R_scalar.add_(self.alpha * H_final.detach())

        D_series = H_final_series - self.R_scalar.to(device)  # [T]
        absD = D_series.abs()

        # 主観時間密度 ρ(t) = 1 + kappa |d|D|/dt|^γ の最後の値
        if T > 1:
            d_absD = absD[1:] - absD[:-1]   # [T-1]
            d_absD_last = d_absD[-1]
        else:
            d_absD_last = torch.tensor(0.0, device=device)

        rho_last = 1.0 + self.kappa_time * (d_absD_last.abs() ** self.gamma_time)

        # 非対称価値 V_dev^Ω
        D_last = D_series[-1]
        D_plus = torch.clamp(D_last, min=0.0)
        D_minus = torch.clamp(D_last, max=0.0)  # <=0

        f_plus = self.a_plus * (D_plus ** self.p_plus)
        f_minus = - self.a_minus * ((-D_minus) ** self.p_minus)

        if T > 1:
            vel = dH_dt_abs[-1]
        else:
            vel = torch.tensor(0.0, device=device)
        g_vel = torch.clamp(self.b_vel * vel, min=0.0)

        V_dev = (f_plus + f_minus) * g_vel  # スカラー

        # 感情帯域モデル（H4→Eとして利用、0〜1→-1〜1にマッピングも可能）
        E_series = (H4 - 0.5) * 2.0         # [-1,1] 仮定
        E_last = E_series[-1].abs()
        band_term = torch.clamp(E_last - self.B_max, min=0.0) ** self.beta_band
        V_band = self.alpha_band * band_term * rho_last

        # メタ認知コスト（Gaussian弱微分＋減衰メモリ）
        if T > 3:
            Ddiff = D_series[1:] - D_series[:-1]       # [T-1]
            Ddiff_smooth = self.gauss(Ddiff)           # [T-1]
            curvature = Ddiff_smooth[1:] - Ddiff_smooth[:-1]  # [T-2]
            curv_mean = curvature.abs().mean()
        else:
            curv_mean = torch.tensor(0.0, device=device)

        with torch.no_grad():
            self.C_m_state.mul_(self.meta_decay)
            self.C_m_state.add_((1.0 - self.meta_decay) * curv_mean.detach())

        V_meta = self.lambda_meta * self.C_m_state.to(device).squeeze()

        # ESVM-Ω の総合出力
        V_omega = V_dev + V_band + V_meta

        # 感情ダイナミクス ODE (内部状態 E_dyn)
        with torch.no_grad():
            absD_last = absD[-1]
            dE = (self.phi1 * V_omega.detach()
                  + self.phi2 * absD_last.detach()
                  + self.phi3 * rho_last.detach()
                  - self.phi4 * self.E_dyn)
            self.E_dyn.add_(self.dt * dE)

        # 9) 5状態分類（従来どおり統合負荷ベース）
        state, latent_score = self.state_machine(H_final, II, stability)

        return {
            'state': state,                    # 0:Calm ~ 4:Critical（int）
            'latent_stress': latent_score,     # Tensor 0〜1
            'inconsistency': II,               # Tensor
            'trust_subjective': w,             # Tensor
            'predict_30s': pred_30s,           # Tensor 0〜1
            'final_load': H_final,             # Tensor 0〜1
            'stability': stability,            # Tensor
            'V_omega': V_omega,                # ESVM-Ω の価値出力（スカラー）
            'E_dyn': self.E_dyn.to(device)     # 内部感情状態（動的）
        }
