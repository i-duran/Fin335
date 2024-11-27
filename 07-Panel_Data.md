# Panel Data Models

We are going to use the AIRFARE data set from Wooldridge’s Introductory
Econometrics which contains data on airline flights, used to study the
factors affecting airfare pricing.

This data set contains the following variables:

- `year`: 1997, 1998, 1999, 2000
- `origin`: flight’s origin
- `destin`: flight’s destination
- `id`: route identifier
- `dist`: distance, in miles
- `passen`: avg. passengers per day
- `fare`: avg. one-way fare, in \$
- `bmktshr`: fraction market, biggest carrier
- `ldist`: log(distance)
- `y98`: =1 if year == 1998
- `y99`: =1 if year == 1999
- `y00`: =1 if year == 2000
- `lfare`: log(fare)
- `ldistsq`: ldist^2
- `concen`: = bmktshr
- `lpassen`: log(passen)

## Import libraries and summarize data

We will be using the `linearmodels` library to estimate panel data
models:

``` python
import pandas as pd
from linearmodels.panel import PanelOLS, RandomEffects, BetweenOLS, compare
from linearmodels.panel import PooledOLS
```

Let’s import and take a look at the data:

``` python
data = pd.read_csv('data/airfare.csv')

# Set the index for both individuals and time
data = data.set_index(['id', 'year'])

print(data.head())
print(data.describe())
```

                  origin         destin  dist  passen  fare  bmktshr     ldist  \
    id year                                                                      
    1  1997  "AKRON, OH"  "ATLANTA, GA"   528     152   106   0.8386  6.269096   
       1998  "AKRON, OH"  "ATLANTA, GA"   528     265   106   0.8133  6.269096   
       1999  "AKRON, OH"  "ATLANTA, GA"   528     336   113   0.8262  6.269096   
       2000  "AKRON, OH"  "ATLANTA, GA"   528     298   123   0.8612  6.269096   
    2  1997  "AKRON, OH"  "ORLANDO, FL"   861     282   104   0.5798  6.758094   

             y98  y99  y00     lfare   ldistsq  concen   lpassen  
    id year                                                       
    1  1997    0    0    0  4.663439  39.30157  0.8386  5.023880  
       1998    1    0    0  4.663439  39.30157  0.8133  5.579730  
       1999    0    1    0  4.727388  39.30157  0.8262  5.817111  
       2000    0    0    1  4.812184  39.30157  0.8612  5.697093  
    2  1997    0    0    0  4.644391  45.67184  0.5798  5.641907  
                  dist       passen         fare      bmktshr        ldist  \
    count  4596.000000  4596.000000  4596.000000  4596.000000  4596.000000   
    mean    989.744996   636.824195   178.796780     0.610115     6.696482   
    std     611.831506   811.999966    74.881513     0.196435     0.659318   
    min      95.000000     2.000000    37.000000     0.160500     4.553877   
    25%     505.000000   215.000000   123.000000     0.465000     6.224558   
    50%     861.000000   357.000000   168.000000     0.603950     6.758094   
    75%    1304.000000   717.000000   225.000000     0.753100     7.173192   
    max    2724.000000  8497.000000   522.000000     1.000000     7.909857   

                  y98         y99         y00        lfare      ldistsq  \
    count  4596.00000  4596.00000  4596.00000  4596.000000  4596.000000   
    mean      0.25000     0.25000     0.25000     5.095601    45.277471   
    std       0.43306     0.43306     0.43306     0.436400     8.726898   
    min       0.00000     0.00000     0.00000     3.610918    20.737790   
    25%       0.00000     0.00000     0.00000     4.812184    38.745120   
    50%       0.00000     0.00000     0.00000     5.123964    45.671840   
    75%       0.25000     0.25000     0.25000     5.416101    51.454680   
    max       1.00000     1.00000     1.00000     6.257668    62.565830   

                concen      lpassen  
    count  4596.000000  4596.000000  
    mean      0.610115     6.017027  
    std       0.196435     0.883987  
    min       0.160500     0.693147  
    25%       0.465000     5.370638  
    50%       0.603950     5.877736  
    75%       0.753100     6.575076  
    max       1.000000     9.047468  

## Estimate Pooled OLS

**Model in levels**

``` python
pooled_ols = PooledOLS.from_formula('fare ~ 1 + dist + passen + bmktshr', data=data)
pooled_results = pooled_ols.fit()
print(pooled_results)
```

                              PooledOLS Estimation Summary                          
    ================================================================================
    Dep. Variable:                   fare   R-squared:                        0.4183
    Estimator:                  PooledOLS   R-squared (Between):              0.4440
    No. Observations:                4596   R-squared (Within):              -0.0154
    Date:                Sat, May 18 2024   R-squared (Overall):              0.4183
    Time:                        23:37:11   Log-likelihood                -2.511e+04
    Cov. Estimator:            Unadjusted                                           
                                            F-statistic:                      1100.7
    Entities:                        1149   P-value                           0.0000
    Avg Obs:                       4.0000   Distribution:                  F(3,4592)
    Min Obs:                       4.0000                                           
    Max Obs:                       4.0000   F-statistic (robust):             1100.7
                                            P-value                           0.0000
    Time periods:                       4   Distribution:                  F(3,4592)
    Avg Obs:                       1149.0                                           
    Min Obs:                       1149.0                                           
    Max Obs:                       1149.0                                           
                                                                                    
                                 Parameter Estimates                              
    ==============================================================================
                Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
    ------------------------------------------------------------------------------
    Intercept      52.995     4.4881     11.808     0.0000      44.197      61.794
    dist           0.0875     0.0016     53.177     0.0000      0.0843      0.0907
    passen        -0.0049     0.0011    -4.6770     0.0000     -0.0070     -0.0029
    bmktshr        69.431     5.1225     13.554     0.0000      59.388      79.473
    ==============================================================================

**Model in logs**

``` python
pooled_ols = PooledOLS.from_formula('lfare ~ 1 + ldist + ldistsq + lpassen + bmktshr', data=data)
pooled_results = pooled_ols.fit()
print(pooled_results)
```

                              PooledOLS Estimation Summary                          
    ================================================================================
    Dep. Variable:                  lfare   R-squared:                        0.4227
    Estimator:                  PooledOLS   R-squared (Between):              0.4396
    No. Observations:                4596   R-squared (Within):               0.1117
    Date:                Sat, May 18 2024   R-squared (Overall):              0.4227
    Time:                        23:37:11   Log-likelihood                   -1447.6
    Cov. Estimator:            Unadjusted                                           
                                            F-statistic:                      840.30
    Entities:                        1149   P-value                           0.0000
    Avg Obs:                       4.0000   Distribution:                  F(4,4591)
    Min Obs:                       4.0000                                           
    Max Obs:                       4.0000   F-statistic (robust):             840.30
                                            P-value                           0.0000
    Time periods:                       4   Distribution:                  F(4,4591)
    Avg Obs:                       1149.0                                           
    Min Obs:                       1149.0                                           
    Max Obs:                       1149.0                                           
                                                                                    
                                 Parameter Estimates                              
    ==============================================================================
                Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
    ------------------------------------------------------------------------------
    Intercept      7.0392     0.4185     16.819     0.0000      6.2187      7.8597
    ldist         -0.9683     0.1266    -7.6515     0.0000     -1.2164     -0.7202
    ldistsq        0.1065     0.0096     11.101     0.0000      0.0877      0.1253
    lpassen       -0.0773     0.0056    -13.745     0.0000     -0.0883     -0.0662
    bmktshr        0.3028     0.0298     10.150     0.0000      0.2443      0.3613
    ==============================================================================

## Estimate fixed effects model

Let’s just estimate the model in logs:

``` python
# fixed_effects = PanelOLS.from_formula('lfare ~ ldist + ldistsq + lpassen + bmktshr + EntityEffects', data=data, drop_absorbed=True)
fixed_effects = BetweenOLS.from_formula('lfare ~ 1 + ldist + ldistsq + lpassen + bmktshr', data=data)
fe_results = fixed_effects.fit()
print(fe_results)
```

                             BetweenOLS Estimation Summary                          
    ================================================================================
    Dep. Variable:                  lfare   R-squared:                        0.4401
    Estimator:                 BetweenOLS   R-squared (Between):              0.4401
    No. Observations:                1149   R-squared (Within):               0.0943
    Date:                Sat, May 18 2024   R-squared (Overall):              0.4222
    Time:                        23:37:11   Log-likelihood                   -313.88
    Cov. Estimator:            Unadjusted                                           
                                            F-statistic:                      224.77
    Entities:                        1149   P-value                           0.0000
    Avg Obs:                       4.0000   Distribution:                  F(4,1144)
    Min Obs:                       4.0000                                           
    Max Obs:                       4.0000   F-statistic (robust):             224.77
                                            P-value                           0.0000
    Time periods:                       4   Distribution:                  F(4,1144)
    Avg Obs:                       1149.0                                           
    Min Obs:                       1149.0                                           
    Max Obs:                       1149.0                                           
                                                                                    
                                 Parameter Estimates                              
    ==============================================================================
                Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
    ------------------------------------------------------------------------------
    Intercept      6.9388     0.8044     8.6258     0.0000      5.3605      8.5171
    ldist         -0.9681     0.2432    -3.9802     0.0001     -1.4453     -0.4909
    ldistsq        0.1069     0.0184     5.7995     0.0000      0.0708      0.1431
    lpassen       -0.0676     0.0110    -6.1441     0.0000     -0.0892     -0.0460
    bmktshr        0.3346     0.0607     5.5123     0.0000      0.2155      0.4536
    ==============================================================================

## Estimate random effects model

``` python
random_effects = RandomEffects.from_formula('lfare ~ 1 + ldist + ldistsq + lpassen + bmktshr', data=data)
re_results = random_effects.fit()
print(re_results)
```

                            RandomEffects Estimation Summary                        
    ================================================================================
    Dep. Variable:                  lfare   R-squared:                        0.2851
    Estimator:              RandomEffects   R-squared (Between):              0.3398
    No. Observations:                4596   R-squared (Within):               0.2619
    Date:                Sat, May 18 2024   R-squared (Overall):              0.3358
    Time:                        23:37:11   Log-likelihood                    4060.9
    Cov. Estimator:            Unadjusted                                           
                                            F-statistic:                      457.64
    Entities:                        1149   P-value                           0.0000
    Avg Obs:                       4.0000   Distribution:                  F(4,4591)
    Min Obs:                       4.0000                                           
    Max Obs:                       4.0000   F-statistic (robust):             457.64
                                            P-value                           0.0000
    Time periods:                       4   Distribution:                  F(4,4591)
    Avg Obs:                       1149.0                                           
    Min Obs:                       1149.0                                           
    Max Obs:                       1149.0                                           
                                                                                    
                                 Parameter Estimates                              
    ==============================================================================
                Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
    ------------------------------------------------------------------------------
    Intercept      8.5437     0.8268     10.334     0.0000      6.9228      10.165
    ldist         -1.0638     0.2508    -4.2425     0.0000     -1.5554     -0.5722
    ldistsq        0.1095     0.0190     5.7796     0.0000      0.0724      0.1467
    lpassen       -0.2236     0.0070    -32.061     0.0000     -0.2372     -0.2099
    bmktshr        0.0999     0.0251     3.9825     0.0001      0.0507      0.1491
    ==============================================================================

## Hausman Test

Let’s first compare the results of both models

``` python
hausman_test = compare({'Fixed Effects': fe_results, 'Random Effects': re_results})
print(hausman_test.summary)
```

                      Model Comparison                 
    ===================================================
                          Fixed Effects  Random Effects
    ---------------------------------------------------
    Dep. Variable                 lfare           lfare
    Estimator                BetweenOLS   RandomEffects
    No. Observations               1149            4596
    Cov. Est.                Unadjusted      Unadjusted
    R-squared                    0.4401          0.2851
    R-Squared (Within)           0.0943          0.2619
    R-Squared (Between)          0.4401          0.3398
    R-Squared (Overall)          0.4222          0.3358
    F-statistic                  224.77          457.64
    P-value (F-stat)             0.0000          0.0000
    =====================  ============ ===============
    Intercept                    6.9388          8.5437
                               (8.6258)        (10.334)
    ldist                       -0.9681         -1.0638
                              (-3.9802)       (-4.2425)
    ldistsq                      0.1069          0.1095
                               (5.7995)        (5.7796)
    lpassen                     -0.0676         -0.2236
                              (-6.1441)       (-32.061)
    bmktshr                      0.3346          0.0999
                               (5.5123)        (3.9825)
    ---------------------------------------------------

    T-stats reported in parentheses

Finally, let’s perform the Hausman test

``` python
import numpy as np
import scipy.stats as stats

# Calculate the difference in coefficients
diff = fe_results.params - re_results.params

# Calculate the variance of the difference
var_diff = fe_results.cov - re_results.cov

# Hausman test statistic
test_stat = diff.T @ np.linalg.inv(var_diff) @ diff

# Degrees of freedom
df = len(fe_results.params)

# p-value
p_value = 1 - stats.chi2.cdf(test_stat, df)

print(f'Hausman Test Statistic: {test_stat}')
print(f'p-value: {p_value}')
```

    Hausman Test Statistic: 336.86682319132865
    p-value: 0.0

Based on the results of the test, **we choose the Fixed effects model**, due
to the presence of correlation between individual effects and the regressors.
