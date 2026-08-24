"""The architectures under test.

H = "slow" / high-level HRM state, L = "fast" / low-level HRM state.
mixer: "mha" = HRM-Text gated multi-head softmax attention,
       "kda" = Kimi Delta Attention (linear attention).
ffn:   "dense" = SwiGLU, "moe" = Kimi Stable LatentMoE (8 experts, top-2, 1 shared).
"""
from lab.model import Variant

_V = [
    # --- baselines (non-recurrent, 4 layers) -------------------------------
    Variant("base_mha_dense", "plain", mixer_h="mha", ffn_h="dense", n_layers=4,
            note="Plain 4-layer transformer, HRM-Text blocks (softmax MHA + SwiGLU)."),
    Variant("base_kda_dense", "plain", mixer_h="kda", ffn_h="dense", n_layers=4,
            note="Plain 4-layer transformer with Kimi linear attention instead of MHA."),
    Variant("base_mha_moe", "plain", mixer_h="mha", ffn_h="moe", n_layers=4,
            note="Plain 4-layer transformer, MHA + Kimi Stable LatentMoE."),

    # --- 1. HRM + MoE MHA --------------------------------------------------
    Variant("hrm_mha_moe", "hrm", mixer_h="mha", ffn_h="moe", mixer_l="mha", ffn_l="moe",
            H_cycles=2, L_cycles=2,
            note="#1 HRM, both levels = MHA + MoE."),

    # --- 2. HRM + Kimi linear ---------------------------------------------
    Variant("hrm_kda_dense", "hrm", mixer_h="kda", ffn_h="dense", mixer_l="kda", ffn_l="dense",
            H_cycles=2, L_cycles=2,
            note="#2 HRM, both levels = Kimi Delta Attention + SwiGLU."),

    # --- 3. looped HRM Kimi-linear with MoE MHA across the loops -----------
    Variant("hrm_loop5_kda_mhamoe", "hrm", mixer_h="mha", ffn_h="moe", mixer_l="kda", ffn_l="dense",
            H_cycles=5, L_cycles=2, bp_min_steps=2, bp_max_steps=5,
            note="#3 5 outer loops: fast(L)=Kimi linear, slow(H)=MHA+MoE across loops."),

    # --- extra controls ----------------------------------------------------
    Variant("hrm_mha_dense", "hrm", mixer_h="mha", ffn_h="dense", mixer_l="mha", ffn_l="dense",
            H_cycles=2, L_cycles=2,
            note="#4a HRM control: MHA + dense, isolates the MoE contribution."),
    Variant("hrm_loop5_kda_mha", "hrm", mixer_h="mha", ffn_h="dense", mixer_l="kda", ffn_l="dense",
            H_cycles=5, L_cycles=2, bp_min_steps=2, bp_max_steps=5,
            note="#5 5 outer loops: fast(L)=Kimi linear, slow(H)=MHA, both dense "
                 "-- the dense twin of #3, isolating MoE."),
    Variant("hrm_loop5_kda_dense", "hrm", mixer_h="kda", ffn_h="dense", mixer_l="kda", ffn_l="dense",
            H_cycles=5, L_cycles=2, bp_min_steps=2, bp_max_steps=5,
            note="#4b All-linear 5-loop HRM: isolates the loop count from the mixer choice."),

    # --- which state wants which mixer? (2 cycles, matched to the peers above) --
    Variant("hrm_hybrid_kda_mha", "hrm", mixer_h="mha", ffn_h="dense", mixer_l="kda", ffn_l="dense",
            H_cycles=2, L_cycles=2,
            note="Hybrid at 2 cycles: fast(L)=Kimi linear, slow(H)=MHA. Direct peer of "
                 "hrm_mha_dense and hrm_kda_dense -- isolates the hybrid from the loop count."),
    Variant("hrm_hybrid_mha_kda", "hrm", mixer_h="kda", ffn_h="dense", mixer_l="mha", ffn_l="dense",
            H_cycles=2, L_cycles=2,
            note="Mirror control: fast(L)=MHA, slow(H)=Kimi linear. Tests whether the "
                 "assignment of mixer to state matters or only the mixture."),

    # --- trained loop-count ladder for the hybrid (fast=KDA, slow=MHA) ------
    # Same architecture, only H_cycles changes: 1 (no outer recurrence at all,
    # the degenerate control) -> 2 -> 3 -> 5. hrm_hybrid_kda_mha is the H=2 rung
    # and hrm_loop5_kda_mha the H=5 rung.
    Variant("hrm_loop1_kda_mha", "hrm", mixer_h="mha", ffn_h="dense", mixer_l="kda", ffn_l="dense",
            H_cycles=1, L_cycles=2, bp_min_steps=2, bp_max_steps=3,
            note="Loop ladder H=1: a single HRM pass, i.e. no outer recurrence. Control for "
                 "whether the looping does anything at all."),
    Variant("hrm_loop3_kda_mha", "hrm", mixer_h="mha", ffn_h="dense", mixer_l="kda", ffn_l="dense",
            H_cycles=3, L_cycles=2, bp_min_steps=2, bp_max_steps=4,
            note="Loop ladder H=3: fast(L)=Kimi linear, slow(H)=MHA."),

    # --- confound controls: the short causal conv ---------------------------
    # KDA carries a kernel-4 causal conv on q/k/v that HRM-Text's MHA does not.
    # These two isolate it: take it away from KDA, and give it to MHA.
    Variant("base_kda_noconv", "plain", mixer_h="kda", ffn_h="dense", n_layers=4,
            cfg_overrides={"kda_conv_kernel": 1},
            note="Kimi linear attention with its short conv disabled (kernel 1)."),
    Variant("base_mha_conv", "plain", mixer_h="mha", ffn_h="dense", n_layers=4,
            cfg_overrides={"mha_conv_kernel": 4},
            note="Softmax MHA given KDA's kernel-4 causal conv on q/k/v."),

    # --- depth-matched HRM (upstream `half_layers: True` convention) --------
    # 2 blocks per level vs the 4-layer baseline: same unique depth/params,
    # HRM still spends more compute through its cycles.
    Variant("hrm_mha_dense_d4", "hrm", mixer_h="mha", ffn_h="dense", mixer_l="mha", ffn_l="dense",
            h_layers=2, l_layers=2, H_cycles=2, L_cycles=2,
            note="Depth-matched HRM (2 blocks per level) vs the 4-layer MHA baseline."),
    Variant("hrm_kda_dense_d4", "hrm", mixer_h="kda", ffn_h="dense", mixer_l="kda", ffn_l="dense",
            h_layers=2, l_layers=2, H_cycles=2, L_cycles=2,
            note="Depth-matched HRM with Kimi linear attention (2 blocks per level)."),
]

VARIANTS = {v.name: v for v in _V}
