# Model results

windows: train 8,537 / holdout 3,174

beta_cross = 9.447e-07 (HAC se 3.492e-06)
bootstrap 95% CI = [-5.916e-06, 7.662e-06]

| model | holdout MSE | OOS R2 |
|---|---|---|
| baseline | 2.337e-07 | -0.003944 |
| extended | 2.337e-07 | -0.003816 |

hypothesis_supported = False

| factor | contemporaneous r | predictive r |
|---|---|---|
| s_k | -0.0013 | +0.0220 |
| g_k | +0.0778 | +0.0248 |
| zt_A | +0.0576 | +0.0332 |
| zt_B | +0.0588 | +0.0040 |
| z_C | +0.4156 | -0.0046 |
