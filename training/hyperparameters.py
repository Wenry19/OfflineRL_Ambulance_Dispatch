
"""Hyperparameters

The values that can take each hyperparameter (when executing the random search).
"""

#### HYPERPARAMETERS BEHAVIOURAL CLONING ####
cloning_hparam = dict()

cloning_hparam['lr'] = [0.0007] #[0.0001, 0.0003, 0.0005, 0.0007]
cloning_hparam['bs'] = [1024] #[128, 256, 512, 1024]
cloning_hparam['net'] = [2] #[0, 1, 2]

#### HYPERPARAMETERS CQL ####

cql_hparam = dict()

cql_hparam['lr'] = [0.0001] #[0.0001, 0.0003, 0.0005, 0.0007]
cql_hparam['bs'] = [256] #[128, 256, 512, 1024]
cql_hparam['disc_fact'] = [0.95] #[0.9, 0.95]
cql_hparam['T'] = [0.01]
cql_hparam['alpha'] = [0.9] #[0.1, 0.3, 0.5, 0.7, 0.9]
cql_hparam['net'] = [0] #[0, 1, 2]

#### HYPERPARAMETERS ACTOR CRITIC WITH KL-DIVERGENCE ####

ac_kl_hparam = dict()

ac_kl_hparam['lr_a'] = [0.0003] #[0.0001, 0.0002, 0.0003]
ac_kl_hparam['lr_c'] = [0.0005] #[0.0001, 0.0003, 0.0005, 0.0007]

ac_kl_hparam['bs_a'] = [1024] #[128, 256, 512, 1024]
ac_kl_hparam['bs_c'] = [1024] #[128, 256, 512, 1024]

ac_kl_hparam['gs'] = [100] #[25, 50, 100]

ac_kl_hparam['disc_fact'] = [0.95] #[0.9, 0.95]
ac_kl_hparam['alpha'] = [0.5] #[0.1, 0.3, 0.5, 0.7, 0.9]

ac_kl_hparam['net_actor'] = [0] #[0, 1, 2]
ac_kl_hparam['net_critic'] = [1] #[0, 1, 2]
