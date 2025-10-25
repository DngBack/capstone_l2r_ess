"""
GSE Worst-group with Exponentiated-Gradient (EG) outer loop.
This implements the EG-outer algorithm for worst-group selective prediction.
"""
import torch
import numpy as np

@torch.no_grad()
def compute_raw_margin_with_beta(eta, alpha, mu, beta, class_to_group):
    """Compute raw margin with beta weighting: (α*β)_g(y) * η̃_y - ((α*β)_g(y) - μ_g(y)) * Σ η̃_y'"""
    cg = class_to_group.to(eta.device)
    ab = (alpha * beta).to(eta.device)             # [K]
    score = (ab[cg] * eta).max(dim=1).values
    coeff = ab[cg] - mu[cg]
    thr = (coeff.unsqueeze(0) * eta).sum(dim=1)
    return score - thr

def accepted_pred_with_beta(eta, alpha, mu, beta, thr, class_to_group):
    """Accept samples and make predictions using beta-weighted margins."""
    raw = compute_raw_margin_with_beta(eta, alpha, mu, beta, class_to_group)
    accepted = (raw >= thr)
    preds = ((alpha*beta)[class_to_group] * eta).argmax(dim=1)
    return accepted, preds, raw - thr

def inner_cost_sensitive_plugin_with_per_group_thresholds(eta_S1, y_S1, eta_S2, y_S2, class_to_group, K,
                                beta, lambda_grid, M=8, alpha_steps=4,
                                target_cov_by_group=None, gamma=0.25, use_conditional_alpha=False,
                                class_weights=None):
    """
    Inner plugin optimization using per-group thresholds t_k fitted on correct predictions.
    
    Args:
        eta_S1, y_S1: S1 split data
        eta_S2, y_S2: S2 split data  
        class_to_group: class to group mapping
        K: number of groups
        beta: [K] group weights from EG outer loop
        lambda_grid: lambda values to search over
        M: number of plugin iterations
        alpha_steps: fixed-point steps for alpha
        target_cov_by_group: [K] target coverage per group
        gamma: EMA factor for alpha updates
        use_conditional_alpha: use conditional acceptance for alpha updates
        class_weights: [C] optional class weights for reweighting (CRITICAL for balanced data!)
    
    Returns:
        best_alpha, best_mu, best_t_group, best_score
    """
    device = eta_S1.device
    alpha = torch.ones(K, device=device)
    best = {"score": float("inf"), "lambda_idx": None}
    mus = []
    lambda_grid = list(lambda_grid)  # Ensure it's mutable
    
    # Default per-group coverage targets
    if target_cov_by_group is None:
        target_cov_by_group = [0.55, 0.45] if K == 2 else [0.58] * K
    
    for lam in lambda_grid:
        if K==2:
            # ✅ FIX: Correct mu convention for worst-group optimization
            # μ_head should be NEGATIVE (lower threshold → accept more head samples)
            # μ_tail should be POSITIVE (higher threshold → selective on tail)
            mus.append(torch.tensor([-lam/2.0, lam/2.0], device=device))
        else: 
            raise NotImplementedError("Provide mu grid for K>2")

    for m in range(M):
        best_lambda_idx = None
        # Track alpha changes for debugging
        alpha_before = alpha.clone()
        
        for i, (lam, mu) in enumerate(zip(lambda_grid, mus)):
            a_cur = alpha.clone()
            t_group_cur = None
            
            # Import functions
            from src.train.gse_balanced_plugin import update_alpha_fixed_point_blend
            from src.train.per_group_threshold import fit_group_thresholds_from_raw
            
            for _ in range(alpha_steps):
                raw_S1 = compute_raw_margin_with_beta(eta_S1, a_cur, mu, beta, class_to_group)
                
                # ✅ FIX: Fit thresholds on ALL samples (not just correct predictions!)
                # Selective training fits on all samples for better coverage control
                y_groups_S1 = class_to_group[y_S1.cpu()]  # Ground-truth groups from labels
                
                # Fit per-group thresholds on ALL samples (not filtered by correctness)
                t_group_cur = fit_group_thresholds_from_raw(
                    raw_S1.cpu(),           # ALL margins (not just correct)
                    y_groups_S1,           # Ground-truth groups
                    target_cov_by_group,
                    K=K
                )
                t_group_cur = torch.tensor(t_group_cur, device=device)
                
                # ✅ Alpha update: use conditional acceptance with per-group thresholds
                # Compute acceptance based on current thresholds
                y_groups_full_S1 = class_to_group[y_S1]  # [N] on device
                thresholds_per_sample = t_group_cur[y_groups_full_S1]  # [N]
                accepted = (raw_S1 > thresholds_per_sample)  # [N] boolean mask
                
                # Conditional acceptance per group: r̂_k = #{acc ∧ y∈G_k} / #{y∈G_k}
                alpha_hat = torch.zeros(K, dtype=torch.float32, device=device)
                for k in range(K):
                    group_mask = (y_groups_full_S1 == k)
                    if group_mask.sum() > 0:
                        accept_rate = accepted[group_mask].float().mean()
                        alpha_hat[k] = accept_rate
                    else:
                        alpha_hat[k] = 1.0 / K  # Fallback
                
                # EMA update with projection
                a_new = (1 - gamma) * a_cur + gamma * alpha_hat
                # Project: clamp min, then normalize geomean=1
                alpha_min = 0.75
                alpha_max = 1.40
                a_new = a_new.clamp_min(alpha_min)
                log_alpha = a_new.log()
                a_new = torch.exp(log_alpha - log_alpha.mean())
                a_cur = a_new.clamp(min=alpha_min, max=alpha_max)

            # Evaluate on S2 using same per-group thresholds
            from src.train.gse_balanced_plugin import worst_error_on_S_with_per_group_thresholds
            w_err, gerrs = worst_error_on_S_with_per_group_thresholds(
                eta_S2, y_S2, a_cur, mu, t_group_cur, class_to_group, K, class_weights=class_weights
            )
            
            if w_err < best["score"]:
                best.update(dict(score=w_err, alpha=a_cur.clone(), mu=mu.clone(), t_group=t_group_cur.clone()))
                best_lambda_idx = i
                
        # Adaptive lambda grid expansion when best hits boundary
        if best_lambda_idx is not None and best_lambda_idx in [0, len(lambda_grid)-1]:
            step = lambda_grid[1] - lambda_grid[0] if len(lambda_grid) > 1 else 0.25
            if best_lambda_idx == 0:
                new_min = lambda_grid[0] - 4*step
                lambda_grid = np.linspace(new_min, lambda_grid[-1], len(lambda_grid)+4).tolist()
            else:
                new_max = lambda_grid[-1] + 4*step
                lambda_grid = np.linspace(lambda_grid[0], new_max, len(lambda_grid)+4).tolist()
            
            # Update mus for new lambda grid
            mus = []
            for lam in lambda_grid:
                if K==2:
                    # Use same convention as above
                    mus.append(torch.tensor([-lam/2.0, lam/2.0], device=device))
            
            print(f"↔️ Expanded lambda_grid to [{lambda_grid[0]:.2f}, {lambda_grid[-1]:.2f}] ({len(lambda_grid)} pts)")
                
        alpha = 0.5*alpha + 0.5*best["alpha"]
        
        # Debug: print alpha evolution
        alpha_change = (alpha - alpha_before).abs().max().item()
        print(f"  [Inner {m+1}/{M}] α={[f'{a:.4f}' for a in alpha.cpu().tolist()]} (Δmax={alpha_change:.4f}), best_score={best['score']:.4f}")
    
    return best["alpha"], best["mu"], best["t_group"], best["score"]

def worst_group_eg_outer(eta_S1, y_S1, eta_S2, y_S2, class_to_group, K,
                         T=30, xi=0.2, lambda_grid=None, beta_floor=0.05, 
                         beta_momentum=0.25, patience=6, class_weights=None, **inner_kwargs):
    """
    Improved Worst-group EG-outer algorithm with anti-collapse and smooth updates.
    
    Args:
        eta_S1, y_S1: S1 split data
        eta_S2, y_S2: S2 split data
        class_to_group: class to group mapping
        K: number of groups
        T: number of EG outer iterations
        xi: EG step size (reduced for stability)
        lambda_grid: lambda values for inner optimization
        beta_floor: minimum beta value to prevent collapse
        beta_momentum: EMA factor for beta updates
        patience: early stopping patience
        class_weights: [C] optional class weights for reweighting balanced data
        **inner_kwargs: additional arguments for inner optimization
    
    Returns:
        alpha_star, mu_star, t_star, beta_star, history
    """
    device = eta_S1.device
    if lambda_grid is None:
        lambda_grid = np.linspace(-1.5, 1.5, 31).tolist()
    
    # Initialize uniform beta
    beta = torch.full((K,), 1.0/K, device=device)
    best = {"score": float("inf"), "alpha": None, "mu": None, "t": None, "beta": beta.clone()}
    history = []
    no_improve = 0

    print(f"Starting improved EG-outer with T={T}, xi={xi}, beta_floor={beta_floor}, momentum={beta_momentum}")
    print(f"Lambda grid: [{lambda_grid[0]:.2f}, {lambda_grid[-1]:.2f}] ({len(lambda_grid)} points)")
    print(f"Reweighting: {'✅ ENABLED' if class_weights is not None else '❌ DISABLED (may fail on balanced data!)'}")

    for t in range(T):
        print(f"\n=== EG Iteration {t+1}/{T} ===")
        print(f"β={[f'{b:.4f}' for b in beta.detach().cpu().tolist()]}")
        
        # Inner optimization with current beta - use per-group version
        a_t, m_t, thr_group_t, _ = inner_cost_sensitive_plugin_with_per_group_thresholds(
            eta_S1, y_S1, eta_S2, y_S2, class_to_group, K, beta,
            lambda_grid=lambda_grid, class_weights=class_weights, **inner_kwargs
        )
        
        print(f"Learned: α={[f'{a:.4f}' for a in a_t.cpu().tolist()]}, μ={[f'{m:.4f}' for m in m_t.cpu().tolist()]}, t_group={[f'{thr:.4f}' for thr in thr_group_t.cpu().tolist()]}")
        
        # Compute per-group errors on S2 using per-group thresholds
        from src.train.gse_balanced_plugin import worst_error_on_S_with_per_group_thresholds
        w_err, gerrs = worst_error_on_S_with_per_group_thresholds(
            eta_S2, y_S2, a_t, m_t, thr_group_t, class_to_group, K, class_weights=class_weights
        )
        
        # ① Centering errors for relative comparison
        e = torch.tensor(gerrs, device=device)
        e_centered = e - e.mean()
        
        # ② EG update with beta floor to prevent collapse
        beta_new = beta * torch.exp(xi * e_centered)
        beta_new = beta_new + beta_floor / K  # ③ Floor to prevent collapse
        beta_new = beta_new / beta_new.sum()  # Normalize
        
        # ④ EMA/momentum for smooth updates
        beta = (1 - beta_momentum) * beta + beta_momentum * beta_new
        beta = beta / beta.sum()  # Ensure normalization
        
        # ⑤ Early stopping based on worst error improvement
        if w_err + 1e-6 < best["score"]:
            best.update({
                "score": w_err, 
                "alpha": a_t.clone(), 
                "mu": m_t.clone(), 
                "t_group": thr_group_t.clone(),  # Store per-group thresholds
                "beta": beta.clone()
            })
            no_improve = 0
            print(f"  ✅ NEW BEST! Worst={w_err:.4f}, Group errors: {[f'{g:.4f}' for g in gerrs]}")
        else:
            no_improve += 1
            print(f"  Worst={w_err:.4f}, Group errors: {[f'{g:.4f}' for g in gerrs]} (no improve: {no_improve})")
            
            if no_improve >= patience:
                print(f"⏹ Early stop EG at iter {t+1}, best worst={best['score']:.4f}")
                break
                
        history.append({
            "iteration": t+1,
            "beta": beta.detach().cpu().tolist(), 
            "gerrs": [float(x) for x in gerrs],
            "worst_error": float(w_err),
            "centered_errors": e_centered.detach().cpu().tolist()
        })

    print("✅ EG-outer optimization complete")
    return best["alpha"], best["mu"], best["t_group"], best["beta"].detach().cpu(), history