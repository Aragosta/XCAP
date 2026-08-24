# Voided: MoE routing-bias committed per call inside the HRM loop

`MoEFFN` passed `update_routing_bias=True` on every call. Inside a weight-shared
HRM recurrence the same MoE module is invoked once per cycle, so the routing bias
was recomputed and overwritten *between cycles of a single forward pass* --
4x per forward at H=2, 10x at H=5, versus 1x for the non-recurrent baselines.
Measured drift of the routing bias within one forward was 0.17-0.34.

The number of mid-forward mutations scales with loop count, so this penalised
exactly the variants with more loops. Every result here that has a MoE inside a
recurrent level is void; the "more loops is worse" conclusion drawn from them is
not supported.

Fixed by using upstream's own windowed API: one `begin_balance_accumulation()` /
`finalize_and_commit_balance()` per optimizer step. Within-forward drift is now 0.

Unaffected (kept in the main tables): non-recurrent baselines, and HRM variants
with dense SwiGLU instead of MoE.
