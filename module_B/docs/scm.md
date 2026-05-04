# Visual-first SCM for CMGDR

## 1. Structural variables

The v1 causal graph fixes four core variables:

- `U`: latent user preference
- `I`: latent item utility
- `V`: observed visual style or appearance bias
- `Y`: observed interaction outcome

The implementation uses the following operational proxies:

- `U` is approximated by user embeddings propagated on the user-item graph.
- `I` is approximated by the causal item representation `z_i^c`.
- `V` is approximated by aligned visual embeddings plus visual cluster IDs.
- `Y` is the observed Amazon interaction event represented by positive user-item
  pairs in `interactions.parquet`.

## 2. Structural equations

The implementation follows the structural sketch below:

```text
U := f_u(G_u, e_u)
V := f_v(X_img, C_v, e_v)
I := f_i(G_i, e_i)
Y := f_y(U, I, V, e_y)
```

where:

- `G_u` and `G_i` denote graph-derived context from the training user-item
  graph.
- `X_img` denotes the aligned visual embedding.
- `C_v` denotes the visual style cluster.
- `e_*` are exogenous noise terms not explicitly modeled in v1.

The observed recommendation score is therefore contaminated by two paths:

1. the desired path `U -> I -> Y`
2. the confounding path `V -> Y` and `V -> item representation -> Y`

The modeling objective is not to erase all visual information, but to separate
visual bias from item utility strongly enough that the recommendation head is
closer to the deconfounded effect of `(U, I)` on `Y`.

## 3. Why `V` is treated as an observed confounder

In the project framing, visual style is not just another modality that improves
prediction. It also carries shortcut signals:

- certain product photography styles can be spuriously associated with higher
  conversion,
- similar visuals can dominate message passing once propagated through the user
  interaction graph,
- naive multimodal fusion can therefore amplify exposure bias rather than item
  utility.

Because the visual embedding and the visual cluster assignment are directly
observable, v1 treats `V` as an observed confounder rather than a latent hidden
 variable. This is narrower than a full multimodal SCM, but it is consistent
 with the PPT objective "block the confounding path of V".

## 4. Why the CMGDR decomposition is theoretically motivated

The implementation uses three complementary constraints:

### Residualization

`z_i^g` is the graph-driven item representation. We explicitly decompose it into

```text
z_i^g ~= z_i^c + z_i^b
```

where:

- `z_i^c` is intended to preserve utility-relevant structure
- `z_i^b` is intended to absorb the part recoverable from visual signals

This step operationalizes the idea that the observed representation mixes causal
item utility with a visual shortcut.

### Adversarial invariance

If `z_i^c` still linearly reveals visual cluster identity, then the model has
not actually removed visual confounding. The adversarial classifier is therefore
used as a negative test: minimizing ranking loss while making visual cluster
prediction harder from `z_i^c` encourages an invariant causal representation.

### Counterfactual consistency

For a fixed graph-derived item state, replacing the observed visual embedding by
a cluster prototype yields a counterfactual style perturbation `V'`. If the
causal score changes substantially under that perturbation, then the "causal"
representation still depends on style. Enforcing

```text
s_causal(U, I, V) ~= s_causal(U, I, V')
```

pushes the causal branch toward stability under controlled visual intervention,
while allowing the total score to remain sensitive for diagnostic purposes.

## 5. Why this approximates `p(Y | do(U, I))`

The v1 package does not claim exact identification of the true interventional
distribution. Instead, it approximates visual-confounding adjustment through the
following logic:

1. `V` is observed and aligned per item.
2. the bias branch explicitly captures visual information recoverable from
   embeddings and clusters.
3. the adversarial loss reduces the recoverability of `V` from the causal
   branch.
4. the counterfactual stability loss reduces variation in the causal score under
   style-preserving perturbations.

Taken together, these steps move the recommendation head away from the naive
observational score `p(Y | U, I, V)` and toward a deconfounded approximation of
`p(Y | do(U, I))`.

The appropriate claim for reports and papers is therefore:

- `visual-confounding adjustment`
- `debiased effect approximation`

and not:

- strict identification of the true structural causal effect

## 6. Limits of v1

- The SCM is visual-first only and does not yet include text as a confounder.
- The graph itself may still encode historical exposure bias.
- Cluster-based counterfactuals are approximations, not real interventions on
  raw images.
- The model targets practical debiasing and ablation-ready diagnostics rather
  than a complete identification proof.
