post-hoc GRAMification of a pretrained model.
GOAL: using <10% of pretraining compute, convert an unfiltered baseline model into a model equivalent to a GR pretrained model.
current approach:
1. noising of baseline MLP, targeting params associated with aux capabilities
2. instantiation of routed-MLP aux params
3. retraining on a mix of core and aux data, using GR to direct updates

ideas for 3:
- normal CEL loss on core and forget
- CEL loss on core, KL vs original on forget
- KL vs maxent'd original on core, KL vs original on forget
- KL vs distill-diff'd core, KL vs original on forget