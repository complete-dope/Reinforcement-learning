# PPO algo 

# PG Loss = -R * log (policy (output | prompt ))
# Baselining : loss = -(R - R_avg) * log (POLICY (output | prompt))
# Advantage : loss = -A * log (POLICY (output | prompt))
# importance sampling : so we sample ones and want to make over it 4-8 gradient steps as inference is also expensive in case of LMs 

# ratio = log[policy(output | prompt)] / log[prev_step_policy(output | prompt )]
# importance ratio = exp(ratio) = policy(output | prompt) / prev_step_policy(output | prompt) = new/old

## ratio > 1 : model is more likely to output that , so (wrong / right) penalise on either side
 
# loss = - Adv * ratio ( how ? , do proof by verification here), we 

### policy : new , prev_step : old
# dl = -A * dr ... (1)
# dr = d(new/old)
# dr = (1/old) * d(new) ... (2)
# using this property : d(log(x)) = (1/x) * dx , we get
# d(new) = new * d(log(new)) , using this in (2)
# dr = (1/old) * new * d(log(new))
# dr = (new/old) * d(log(new))
# dr = Ratio * d (log(new))
# Putting this in (1)

# dl = -A * Ratio * d(log(new))
# d(pg_loss) = -R * d(log(new))

# --x-- SAME PATTERN FOR PG-LOSS AND FOR IMPT. SAMPLING LOSS , aka suggorate loss 

# policy_loss = -min (A * Ratio , A * min(ratio , 1-e, 1+e)) 

# KLD (dont diverge from the reference model)
# KL[ POLICY( output | prompt ) || REFERENCE( output | prompt )]

# kld_loss = Beta * (log(POLICY( output | prompt )) - log( REFERENCE( output | prompt )))

# loss = policy_loss + kld_loss

# ---x---
# maintain : policy, ref_model


