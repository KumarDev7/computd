import functools
import jax
import jax.numpy as jnp
from flax import nnx


class MultiAspectRetrieval(nnx.Module):
    """
    Multi-aspect sigmoid-gated retrieval over the vector pool.

    Phase 1 (use_sigmoid=False, forced_idx provided):
      - Use externally supplied forced_idx with uniform α = 1/k_max.
      - Every vector gets gradient over the course of phase 1 (rotating batches).
      - Similarity scores are still computed for aux losses and W_K gradient.

    Phase 2+ (use_sigmoid=True):
      - Normal similarity → sigmoid gate → top-k → normalized α.
      - Now retrieval has real content to discriminate between.
    """

    def __init__(self, D: int, d_A: int, S: int, d_k: int, N: int, rngs: nnx.Rngs,
                 wk_sharding=None):
        self.S = S
        self.d_k = d_k
        self.N = N

        self.W_Q = nnx.Param(
            (jax.random.normal(rngs.params(), (S, d_k, d_A)) * (d_A ** -0.5)).astype(jnp.bfloat16)
        )
        if wk_sharding is not None:
            # Create W_K directly sharded to avoid OOM on one device.
            # (S, d_k, D) sharded as P(None, None, 'tp') — D split across chips.
            wk_shape = (S, d_k, D)
            def _wk_callback(idx):
                n_dev = wk_sharding.mesh.size
                d_local = D // n_dev
                dev_idx = idx[2].start // d_local
                k = jax.random.fold_in(rngs.params(), dev_idx + 100)
                local_shape = (S, d_k, idx[2].stop - idx[2].start)
                return jax.random.normal(k, local_shape, dtype=jnp.bfloat16) * (D ** -0.5)
            W_K_val = jax.make_array_from_callback(wk_shape, wk_sharding, _wk_callback)
        else:
            W_K_val = (jax.random.normal(rngs.params(), (S, d_k, D)) * (D ** -0.5)).astype(jnp.bfloat16)
        self.W_K = nnx.Param(W_K_val)
        self.aspect_logits = nnx.Param(jnp.zeros(S))
        self.tau = nnx.Param(jnp.zeros(S))

    def __call__(
        self,
        z: jax.Array,              # (batch, d_A)
        vectors: jax.Array,        # (N, D)
        k_max: int,
        T: float = 1.0,
        lambda_sharp: float = 1.0,
        use_sigmoid: bool = True,
        forced_idx: jax.Array | None = None,  # (batch, k_max) — phase 1 only
    ):
        batch = z.shape[0]

        # Always compute full similarities — needed for aux losses and W_K gradients
        keys    = jnp.einsum('skd,nd->snk', self.W_K.value, vectors)
        keys    = keys / (jnp.linalg.norm(keys, axis=-1, keepdims=True) + 1e-8)
        queries = jnp.einsum('skd,bd->sbk', self.W_Q.value, z)
        queries = queries / (jnp.linalg.norm(queries, axis=-1, keepdims=True) + 1e-8)

        aspect_sims = jnp.einsum('sbk,snk->sbn', queries, keys)  # (S, batch, N)
        w    = jax.nn.softmax(self.aspect_logits.value)           # (S,)
        sims = jnp.einsum('s,sbn->bn', w, aspect_sims)           # (batch, N)

        if use_sigmoid:
            tau      = jnp.dot(w, self.tau.value)
            gate     = jax.nn.sigmoid(lambda_sharp * (sims - tau))
            alpha_raw = gate * jnp.exp(sims / T)
            top_vals, top_idx = jax.lax.top_k(alpha_raw, k_max)
            alpha = top_vals / (jnp.sum(top_vals, axis=-1, keepdims=True) + 1e-8)
            return alpha, top_idx, sims, alpha_raw
        else:
            alpha_raw = jnp.exp(sims / T)
            if forced_idx is not None:
                # Phase-1 warmup: use externally supplied rotation indices, uniform α
                idx   = forced_idx
                alpha = jnp.full((batch, k_max), 1.0 / k_max)
            else:
                # Phase-1 eval / fallback: plain softmax top-k (no sigmoid gate)
                top_vals, idx = jax.lax.top_k(alpha_raw, k_max)
                alpha = top_vals / (jnp.sum(top_vals, axis=-1, keepdims=True) + 1e-8)
            return alpha, idx, sims, alpha_raw

    def soft_forward(
        self,
        z: jax.Array,        # (batch, seq, d_A)
        vectors: jax.Array,  # (N, D)
        T: float = 1.0,
        lambda_sharp: float = 1.0,
        use_sigmoid: bool = True,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """
        Soft retrieval: alpha over ALL N vectors — no top-k, no gather, pure GEMMs.
        TPU-optimal: every op is a dense matmul the MXU can saturate.

        Returns:
          alpha:         (batch, seq, N) — per-position soft weights over all N
          sims_seq:      (batch, N)     — seq-mean sims  (for aux losses)
          alpha_raw_seq: (batch, N)     — seq-mean raw scores
        """
        # Keys for all N vectors: (S, N, d_k)
        keys = jnp.einsum('skd,nd->snk', self.W_K.value, vectors)
        keys = keys / (jnp.linalg.norm(keys, axis=-1, keepdims=True) + 1e-8)

        # Per-position queries: (S, batch, seq, d_k)
        queries = jnp.einsum('skd,btd->sbtk', self.W_Q.value, z)
        queries = queries / (jnp.linalg.norm(queries, axis=-1, keepdims=True) + 1e-8)

        # Per-position similarities to all N: (S, batch, seq, N)
        sims = jnp.einsum('sbtk,snk->sbtn', queries, keys)
        w      = jax.nn.softmax(self.aspect_logits.value)
        sims_w = jnp.einsum('s,sbtn->btn', w, sims)  # (batch, seq, N)

        if use_sigmoid:
            tau       = jnp.dot(w, self.tau.value)
            gate      = jax.nn.sigmoid(lambda_sharp * (sims_w - tau))
            alpha_raw = gate * jnp.exp(sims_w / T)
        else:
            alpha_raw = jnp.exp(sims_w / T)

        alpha = alpha_raw / (alpha_raw.sum(axis=-1, keepdims=True) + 1e-8)

        # Collapse seq dim for aux-loss compatibility (diversity_loss, entropy_loss)
        return alpha, sims_w.mean(axis=1), alpha_raw.mean(axis=1)

    def hybrid_forward(
        self,
        z: jax.Array,        # (batch, seq, d_A)
        vectors: jax.Array,  # (N, D)
        k_max: int,
        T: float = 1.0,
        lambda_sharp: float = 1.0,
        use_sigmoid: bool = True,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """
        Hybrid retrieval: compute ALL N dot products (GEMM-saturated, like soft)
        but keep only top-k for assembly (memory-efficient, like hard).

        TPU-optimal: the full similarity GEMM saturates the MXU, then top_k
        immediately discards the (B, seq, N) intermediate — only (B, seq, k_max)
        is materialized for assembly and backprop.

        Returns:
          alpha:         (batch, seq, k_max) — top-k weights (tiny, same as hard)
          top_idx:       (batch, seq, k_max) — indices of top-k vectors
          sims_seq:      (batch, N)          — seq-mean sims (for aux losses)
          alpha_raw_seq: (batch, N)          — seq-mean raw scores (for aux losses)
        """
        # Keys for all N vectors: (S, N, d_k)
        keys = jnp.einsum('skd,nd->snk', self.W_K.value, vectors)
        keys = keys / (jnp.linalg.norm(keys, axis=-1, keepdims=True) + 1e-8)

        # Per-position queries: (S, batch, seq, d_k)
        queries = jnp.einsum('skd,btd->sbtk', self.W_Q.value, z)
        queries = queries / (jnp.linalg.norm(queries, axis=-1, keepdims=True) + 1e-8)

        # Per-position similarities to all N: (S, batch, seq, N)
        sims = jnp.einsum('sbtk,snk->sbtn', queries, keys)
        w      = jax.nn.softmax(self.aspect_logits.value)
        sims_w = jnp.einsum('s,sbtn->btn', w, sims)  # (batch, seq, N)

        if use_sigmoid:
            tau       = jnp.dot(w, self.tau.value)
            gate      = jax.nn.sigmoid(lambda_sharp * (sims_w - tau))
            alpha_raw = gate * jnp.exp(sims_w / T)
        else:
            alpha_raw = jnp.exp(sims_w / T)

        # Top-k: full GEMM computed, but only keep k_max per position
        top_vals, top_idx = jax.lax.top_k(alpha_raw, k_max)  # (batch, seq, k_max)

        # Normalize only over top-k (not over all N — saves the massive softmax)
        alpha = top_vals / (jnp.sum(top_vals, axis=-1, keepdims=True) + 1e-8)

        # Collapse seq dim for aux-loss compatibility
        return alpha, top_idx, sims_w.mean(axis=1), alpha_raw.mean(axis=1)

    def per_position_alpha(
        self,
        z: jax.Array,              # (batch, seq, d_A)
        selected_vecs: jax.Array,  # (batch, k_max, D) — already-gathered k vectors
    ) -> jax.Array:                # (batch, seq, k_max)
        """
        Per-position weights against the k pre-selected vectors.
        Cost: O(batch × seq × k × d_k) — no N-wide similarity scan.
        Called after sequence-level retrieval determines which k vectors to use.
        """
        # Keys for only the k selected vectors: (S, batch, k_max, d_k)
        keys = jnp.einsum('sdc,bkc->sbkd', self.W_K.value, selected_vecs)
        keys = keys / (jnp.linalg.norm(keys, axis=-1, keepdims=True) + 1e-8)
        # Per-position queries: (S, batch, seq, d_k)
        queries = jnp.einsum('sdc,btc->sbtd', self.W_Q.value, z)
        queries = queries / (jnp.linalg.norm(queries, axis=-1, keepdims=True) + 1e-8)
        # Similarities: (S, batch, seq, k_max) — no N dim!
        sims = jnp.einsum('sbtd,sbkd->sbtk', queries, keys)
        w    = jax.nn.softmax(self.aspect_logits.value)
        return jax.nn.softmax(jnp.einsum('s,sbtk->btk', w, sims), axis=-1)

    def pallas_hybrid_forward(
        self,
        z: jax.Array,          # (batch, seq, d_A)
        vectors: jax.Array,    # (N, D)  — full pool (GSPMD handles per-chip view)
        k_max: int,
        T: float = 1.0,
        lambda_sharp: float = 1.0,
        use_sigmoid: bool = True,
        tp_axis: str | None = 'tp',   # mesh axis name for shard_map; None = single-chip
        mesh=None,                    # jax.sharding.Mesh — required when tp_axis is not None
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """
        Pallas-backed hybrid retrieval with multi-chip support.

        Each chip runs the fused GEMM+gate+exp Pallas kernel over its local
        N_local pool shard.  Global top-k is obtained via all_gather of the
        per-chip top-k candidates followed by a final JAX top_k.

        Memory profile (7B, 8 chips, BT=8192, N=32768)
        ------------------------------------------------
        Per-chip fused scores : (8192, 4096) float32 = 128 MB
        Full (B,T,N)          : never assembled on any single chip  ✓

        Returns same 4-tuple as hybrid_forward:
            alpha     (batch, seq, k_max)   — normalised weights
            top_idx   (batch, seq, k_max)   — global pool indices
            sims_seq  (batch, N_local)      — seq-mean sims (aux losses, local)
            alpha_raw_seq (batch, N_local)  — seq-mean raw scores (aux losses, local)
        """
        from src.kernels.tiled_topk import tiled_topk_fused, tiled_topk_fallback

        B, seq, d_A = z.shape
        N_local = vectors.shape[0]

        # ------------------------------------------------------------------
        # 1. Project pool vectors to key space: (n_aspects, N_local, d_k)
        # ------------------------------------------------------------------
        keys_s = jnp.einsum('skd,nd->snk', self.W_K.value, vectors)  # (n_asp, N_local, d_k)
        keys_s = keys_s / (jnp.linalg.norm(keys_s, axis=-1, keepdims=True) + 1e-8)

        # ------------------------------------------------------------------
        # 2. Project query tokens: (n_aspects, batch, seq, d_k)
        # ------------------------------------------------------------------
        queries_s = jnp.einsum('skd,btd->sbtk', self.W_Q.value, z)   # (n_asp, B, seq, d_k)
        queries_s = queries_s / (jnp.linalg.norm(queries_s, axis=-1, keepdims=True) + 1e-8)

        # ------------------------------------------------------------------
        # 3. Aspect weights
        # ------------------------------------------------------------------
        w   = jax.nn.softmax(self.aspect_logits.value)   # (n_asp,)
        tau = jnp.dot(w, self.tau.value) if use_sigmoid else jnp.array(0.0)

        # ------------------------------------------------------------------
        # 4. Collapse aspects into a single d_k projection via weighted mean.
        #    keys_w:    (N_local, d_k)   — aspect-averaged keys
        #    queries_w: (B*seq, d_k)     — aspect-averaged queries, flattened
        # ------------------------------------------------------------------
        keys_w    = jnp.einsum('s,snk->nk', w, keys_s)               # (N_local, d_k)
        queries_w = jnp.einsum('s,sbtk->btk', w, queries_s)          # (B, seq, d_k)
        queries_flat = queries_w.reshape(B * seq, -1)                 # (BT, d_k)

        # ------------------------------------------------------------------
        # 5+6. Cross-chip top-k merge via shard_map.
        #
        #  Uses the differentiable JAX fallback for the similarity computation
        #  (Pallas kernel lacks reverse-mode autodiff).  The key memory win is
        #  the 8-way TP sharding of the pool + top-k (only k_max kept), not the
        #  Pallas fusion.  The Pallas kernel is available for inference-only use.
        #
        #  With tp_axis set: shard_map so each chip runs on its local pool shard,
        #  then all_gather + merge (axis_index requires shard_map context).
        #
        #  Without tp_axis: single-chip, no communication.
        # ------------------------------------------------------------------
        BT = B * seq

        if tp_axis is not None and mesh is not None:
            import functools
            from jax.experimental.shard_map import shard_map
            from jax.sharding import PartitionSpec as P

            def _chip_topk(queries_flat, keys_w_local):
                N_local_l = keys_w_local.shape[0]
                # Use differentiable JAX path (not Pallas — no VJP support)
                tv, li = tiled_topk_fallback(
                    queries_flat, keys_w_local, k_max,
                    lambda_sharp, tau, T, bool(use_sigmoid),
                )
                chip_id    = jax.lax.axis_index(tp_axis)
                global_idx = li + chip_id * N_local_l           # (BT, k_max)

                cand_v = jax.lax.all_gather(tv,         tp_axis, axis=0, tiled=False)
                cand_i = jax.lax.all_gather(global_idx, tp_axis, axis=0, tiled=False)
                n_chips = cand_v.shape[0]

                cv_flat = cand_v.transpose(1, 0, 2).reshape(BT, n_chips * k_max)
                ci_flat = cand_i.transpose(1, 0, 2).reshape(BT, n_chips * k_max)
                fv, sel = jax.lax.top_k(cv_flat, k_max)
                fi = ci_flat[jnp.arange(BT)[:, None], sel]
                return fv, fi

            _chip_topk_sharded = functools.partial(
                shard_map,
                mesh=mesh,
                in_specs=(P(), P(tp_axis, None)),
                out_specs=(P(), P()),
                check_rep=False,
            )(_chip_topk)

            final_v, final_i = _chip_topk_sharded(queries_flat, keys_w)
        else:
            # Single-chip path
            final_v, final_i = tiled_topk_fallback(
                queries_flat, keys_w, k_max,
                lambda_sharp, tau, T, bool(use_sigmoid),
            )

        # ------------------------------------------------------------------
        # 7. Reshape back to (batch, seq, k_max) and normalise alpha
        # ------------------------------------------------------------------
        top_vals_3d = final_v.reshape(B, seq, k_max)
        top_idx_3d  = final_i.reshape(B, seq, k_max)
        alpha = top_vals_3d / (jnp.sum(top_vals_3d, axis=-1, keepdims=True) + 1e-8)

        # ------------------------------------------------------------------
        # 8. Seq-mean sims for aux losses (cheap: mean query × all N_local keys)
        #    Cost: (B, d_k) @ (d_k, N_local) — tiny compared to per-position
        # ------------------------------------------------------------------
        z_mean      = z.mean(axis=1)                                  # (B, d_A)
        q_mean_s    = jnp.einsum('skd,bd->sbk', self.W_Q.value, z_mean)
        q_mean_s    = q_mean_s / (jnp.linalg.norm(q_mean_s, axis=-1, keepdims=True) + 1e-8)
        q_mean_w    = jnp.einsum('s,sbk->bk', w, q_mean_s)           # (B, d_k)
        sims_seq    = jnp.einsum('bk,nk->bn', q_mean_w, keys_w)      # (B, N_local)

        if use_sigmoid:
            alpha_raw_seq = jax.nn.sigmoid(lambda_sharp * (sims_seq - tau)) * jnp.exp(sims_seq / T)
        else:
            alpha_raw_seq = jnp.exp(sims_seq / T)

        return alpha, top_idx_3d, sims_seq, alpha_raw_seq
