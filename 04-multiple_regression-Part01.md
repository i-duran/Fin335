# Multiple Regression Analysis

We will use as an example an extended CAPM model in which we add
additional macroeconomic variables, on top of the traditional variables
described in the model.

## Import libraries

Let’s import the libraries first:

``` python
import pandas as pd
import statsmodels.formula.api as smf
```

## Load and summarize the data

Let’s load and summarize the data:

``` python
df = pd.read_csv("./data/macro.csv", index_col=0)
print(df.describe())
print(df.head())
```

            MICROSOFT        SANDP         CPI      INDPRO     M1SUPPLY  \
    count  385.000000   385.000000  385.000000  385.000000   385.000000   
    mean    23.301377  1066.036104  181.062234   86.629558  1514.690130   
    std     19.255689   602.397162   41.136587   15.887233   778.509244   
    min      0.100000   230.300000  108.600000   56.500000   624.300000   
    25%      2.890000   459.270000  147.200000   69.480000  1069.300000   
    50%     25.720000  1104.490000  178.800000   93.000000  1191.800000   
    75%     30.860000  1385.590000  218.180000  100.720000  1716.000000   
    max     95.010000  2823.810000  249.550000  106.660000  3684.700000   

               CCREDIT     BMINUSA      USTB3M     USTB10Y  
    count   385.000000  385.000000  385.000000  385.000000  
    mean   1897.814831    0.974623    3.296909    5.075403  
    std     949.565970    0.382047    2.589801    2.173512  
    min     606.800000    0.550000    0.010000    1.500000  
    25%     886.170000    0.720000    0.450000    3.330000  
    50%    1891.830000    0.900000    3.440000    4.910000  
    75%    2620.450000    1.130000    5.290000    6.740000  
    max    3843.420000    3.380000    9.140000    9.520000  
            MICROSOFT   SANDP    CPI  INDPRO  M1SUPPLY  CCREDIT  BMINUSA  USTB3M  \
    Date                                                                           
    Mar-86       0.10  238.90  108.8   56.54     624.3   606.80     1.50    6.76   
    Apr-86       0.11  235.52  108.6   56.57     647.0   614.37     1.40    6.24   
    May-86       0.12  247.35  108.9   56.69     645.7   621.92     1.20    6.33   
    Jun-86       0.11  250.84  109.5   56.50     662.8   627.89     1.21    6.40   
    Jul-86       0.10  236.12  109.5   56.81     673.4   633.61     1.28    6.00   

            USTB10Y  
    Date             
    Mar-86     7.78  
    Apr-86     7.30  
    May-86     7.71  
    Jun-86     7.80  
    Jul-86     7.30  

The data is comprised of:

- Microsoft stock price
- S&P500 index value
- Consumer price index
- Industrial production index,
- Measure of ‘narrow’ money supply (M1),
- Consumer credit series, and a ‘credit spread’ series.
- Treasury bill yields for three months and ten years

## Calculate variables

Calculate the return of Microsoft and S&P, as well as the corresponding
return premiums:

``` python
df["MS_r"] = df["MICROSOFT"].pct_change() * 100
df["MS_r_premium"] = df["MS_r"] - df["USTB3M"]
df["SANDP_r"] = df["SANDP"].pct_change() * 100
df["SANDP_r_premium"] = df["SANDP_r"] - df["USTB3M"]
```

## Model to estimate

Let’s add the inflation rate as covariate or explanatory variable, and
so we will estimate the following equation:

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + u$$

where $y$ is `MS_r_premium`, $x_1$ is the `SANDP_r_premium`, and $x_2$
is either `inflation` or money growth rate `m1_growth`.

We have the CPI already, so let’s calculate the inflation rate:

``` python
df["inflation"] = df["CPI"].pct_change() * 100
df["m1_growth"] = df["M1SUPPLY"].pct_change() * 100
```

Let’s drop the missing values from our data:

``` python
df = df.dropna()
print(df.head())
```

            MICROSOFT   SANDP    CPI  INDPRO  M1SUPPLY  CCREDIT  BMINUSA  USTB3M  \
    Date                                                                           
    Apr-86       0.11  235.52  108.6   56.57     647.0   614.37     1.40    6.24   
    May-86       0.12  247.35  108.9   56.69     645.7   621.92     1.20    6.33   
    Jun-86       0.11  250.84  109.5   56.50     662.8   627.89     1.21    6.40   
    Jul-86       0.10  236.12  109.5   56.81     673.4   633.61     1.28    6.00   
    Aug-86       0.10  252.93  109.7   56.73     678.4   640.51     1.46    5.69   

            USTB10Y       MS_r  MS_r_premium   SANDP_r  SANDP_r_premium  \
    Date                                                                  
    Apr-86     7.30  10.000000      3.760000 -1.414818        -7.654818   
    May-86     7.71   9.090909      2.760909  5.022928        -1.307072   
    Jun-86     7.80  -8.333333    -14.733333  1.410956        -4.989044   
    Jul-86     7.30  -9.090909    -15.090909 -5.868283       -11.868283   
    Aug-86     7.17   0.000000     -5.690000  7.119261         1.429261   

            inflation  m1_growth  
    Date                          
    Apr-86  -0.183824   3.636072  
    May-86   0.276243  -0.200927  
    Jun-86   0.550964   2.648289  
    Jul-86   0.000000   1.599276  
    Aug-86   0.182648   0.742501  

## Model specification in `statsmodels`

Let’s define our `statsmodels` specifications:

``` python
model1 = smf.ols("MS_r_premium ~ SANDP_r_premium", data=df)
model2 = smf.ols("MS_r_premium ~ SANDP_r_premium + inflation", data=df)
model3 = smf.ols("MS_r_premium ~ SANDP_r_premium + m1_growth", data=df)
```

## Model estimation

Let’s estimate the basic model first:

``` python
results1 = model1.fit()
print(results1.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:           MS_r_premium   R-squared:                       0.306
    Model:                            OLS   Adj. R-squared:                  0.304
    Method:                 Least Squares   F-statistic:                     168.2
    Date:                Wed, 23 Oct 2024   Prob (F-statistic):           3.99e-32
    Time:                        08:49:37   Log-Likelihood:                -1352.3
    No. Observations:                 384   AIC:                             2709.
    Df Residuals:                     382   BIC:                             2717.
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ===================================================================================
                          coef    std err          t      P>|t|      [0.025      0.975]
    -----------------------------------------------------------------------------------
    Intercept           1.7922      0.472      3.795      0.000       0.864       2.721
    SANDP_r_premium     1.1015      0.085     12.969      0.000       0.935       1.268
    ==============================================================================
    Omnibus:                       50.069   Durbin-Watson:                   2.069
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              122.328
    Skew:                           0.651   Prob(JB):                     2.73e-27
    Kurtosis:                       5.439   Cond. No.                         6.31
    ==============================================================================

    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

Now, let’s estimate the augmented model with `inflation`:

``` python
results2 = model2.fit()
print(results2.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:           MS_r_premium   R-squared:                       0.311
    Model:                            OLS   Adj. R-squared:                  0.308
    Method:                 Least Squares   F-statistic:                     86.10
    Date:                Wed, 23 Oct 2024   Prob (F-statistic):           1.40e-31
    Time:                        08:49:37   Log-Likelihood:                -1350.8
    No. Observations:                 384   AIC:                             2708.
    Df Residuals:                     381   BIC:                             2719.
    Df Model:                           2                                         
    Covariance Type:            nonrobust                                         
    ===================================================================================
                          coef    std err          t      P>|t|      [0.025      0.975]
    -----------------------------------------------------------------------------------
    Intercept           1.3377      0.537      2.490      0.013       0.281       2.394
    SANDP_r_premium     1.1194      0.085     13.121      0.000       0.952       1.287
    inflation           2.3068      1.312      1.759      0.079      -0.272       4.886
    ==============================================================================
    Omnibus:                       45.560   Durbin-Watson:                   2.050
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              105.516
    Skew:                           0.611   Prob(JB):                     1.22e-23
    Kurtosis:                       5.259   Cond. No.                         17.9
    ==============================================================================

    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

As a third specification, let’s estimate the augmented model with the
`CPI`, instead of inflation:

``` python
results3 = model3.fit()
print(results3.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:           MS_r_premium   R-squared:                       0.307
    Model:                            OLS   Adj. R-squared:                  0.303
    Method:                 Least Squares   F-statistic:                     84.24
    Date:                Wed, 23 Oct 2024   Prob (F-statistic):           5.07e-31
    Time:                        08:49:37   Log-Likelihood:                -1352.1
    No. Observations:                 384   AIC:                             2710.
    Df Residuals:                     381   BIC:                             2722.
    Df Model:                           2                                         
    Covariance Type:            nonrobust                                         
    ===================================================================================
                          coef    std err          t      P>|t|      [0.025      0.975]
    -----------------------------------------------------------------------------------
    Intercept           1.8978      0.495      3.832      0.000       0.924       2.872
    SANDP_r_premium     1.1065      0.085     12.975      0.000       0.939       1.274
    m1_growth          -0.1950      0.274     -0.712      0.477      -0.734       0.344
    ==============================================================================
    Omnibus:                       49.575   Durbin-Watson:                   2.069
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              118.674
    Skew:                           0.651   Prob(JB):                     1.70e-26
    Kurtosis:                       5.392   Cond. No.                         6.74
    ==============================================================================

    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

## Results Analysis

**Analyze the results!** When doing so, make sure to:

- Interpret the coefficients associated to each explanatory variable
  $x_j$.
- Use the units of the explanatory variables in the interpretation of
  the betas.
- If the variables are in logs, use the interpretation based on log-log,
  log-levels, or levels-logs.
- Analyze the significance of the coefficients based on their
  `p-value`’s
- Interpret the $R^2$’ and Adjusted `R^2`.

## Additional Hypothesis Test

Let’s **test whether the coefficient related to inflation is equal to
1** or greater than 1. That is, the null hypothesis is defined as beta
equal to 1, while the alternative is that it is greater than 1.

Based on the results above, we have that
$\text{t-statistic} = \dfrac{2.3068 - 1}{1.312} = 0.996$.

The **p-value** for this t-statistic is 0.1605 (that is, the probability
left to the right of 0.1605). Therefore, **we fail to reject the null
hypothesis** as it is greater than 0.1.

If we define the alternative hypothesis to be different than 1 (two
tails), rather than greater than 1 (right tail), the **p-value** is
$0.1605 \times 2 = 0.3210$, and so we fail to reject the null hypothesis
even with a larger probability.

**Conclusion**: statistically, we cannot say that the population or true
coefficient associated with inflation is different than 1.
