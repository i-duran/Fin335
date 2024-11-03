# Testing for Heteroscedasticity

## Data set

We will be using the same modified wages data set of the previous
section.

Remember that this data has the following variables:

- `wage`: average hourly earnings
- `educ`: years of education
- `exper`: years potential experience
- `tenure`: years with current employer
- `married`: =1 if married
- `numdep`: number of dependents
- `gender`: male or female
- `skin`: color of the skin (white or nonwhite)

## Base regression

``` python
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

data = pd.read_csv("./data/wage.csv")
data.head()
```

|     | wage | educ | exper | tenure | married | numdep | gender | skin  |
|-----|------|------|-------|--------|---------|--------|--------|-------|
| 0   | 3.10 | 11   | 2     | 0      | 0       | 2      | female | white |
| 1   | 3.24 | 12   | 22    | 2      | 1       | 3      | female | white |
| 2   | 3.00 | 11   | 2     | 0      | 0       | 2      | male   | white |
| 3   | 6.00 | 8    | 44    | 28     | 1       | 0      | male   | white |
| 4   | 5.30 | 12   | 7     | 2      | 1       | 1      | male   | white |


As before, we first want to construct the dummy variables for `gender`
and `skin`:

``` python
data = pd.get_dummies(data, columns=["gender", "skin"], drop_first=False, dtype=int)
data.head()
```

|     | wage | educ | exper | tenure | married | numdep | gender_female | gender_male | skin_nonwhite | skin_white |
|-----|------|------|-------|--------|---------|--------|---------------|-------------|---------------|------------|
| 0   | 3.10 | 11   | 2     | 0      | 0       | 2      | 1             | 0           | 0             | 1          |
| 1   | 3.24 | 12   | 22    | 2      | 1       | 3      | 1             | 0           | 0             | 1          |
| 2   | 3.00 | 11   | 2     | 0      | 0       | 2      | 0             | 1           | 0             | 1          |
| 3   | 6.00 | 8    | 44    | 28     | 1       | 0      | 0             | 1           | 0             | 1          |
| 4   | 5.30 | 12   | 7     | 2      | 1       | 1      | 0             | 1           | 0             | 1          |


Now, let’s run a regression:

``` python
model = smf.ols("wage ~ educ + exper + tenure + married + numdep + gender_female + skin_nonwhite", data=data)
results = model.fit()
print(results.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                   wage   R-squared:                       0.370
    Model:                            OLS   Adj. R-squared:                  0.362
    Method:                 Least Squares   F-statistic:                     43.54
    Date:                Sun, 03 Nov 2024   Prob (F-statistic):           2.53e-48
    Time:                        22:40:54   Log-Likelihood:                -1311.4
    No. Observations:                 526   AIC:                             2639.
    Df Residuals:                     518   BIC:                             2673.
    Df Model:                           7                                         
    Covariance Type:            nonrobust                                         
    =================================================================================
                        coef    std err          t      P>|t|      [0.025      0.975]
    ---------------------------------------------------------------------------------
    Intercept        -1.9799      0.783     -2.530      0.012      -3.517      -0.442
    educ              0.5746      0.052     11.034      0.000       0.472       0.677
    exper             0.0219      0.012      1.789      0.074      -0.002       0.046
    tenure            0.1383      0.021      6.541      0.000       0.097       0.180
    married           0.4597      0.295      1.556      0.120      -0.121       1.040
    numdep            0.1465      0.109      1.345      0.179      -0.067       0.360
    gender_female    -1.7594      0.267     -6.592      0.000      -2.284      -1.235
    skin_nonwhite    -0.1100      0.427     -0.257      0.797      -0.950       0.730
    ==============================================================================
    Omnibus:                      186.895   Durbin-Watson:                   1.788
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              711.513
    Skew:                           1.604   Prob(JB):                    3.14e-155
    Kurtosis:                       7.709   Cond. No.                         153.
    ==============================================================================

    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

## Performing the White test

Now, let’s perform the White test, which is also available in the
`statsmodels` library. **Note the following**:

- We use the same independent variables of the original model, obtained
  with `results.model.exog`.
- The estimated residual are obtained from the previous estimation with
  `results.resid`.
- `statsmodels` automatically include squares and interaction terms to
  run the auxiliary regression for the White test (so we don’t have to
  do this ourselves).

``` python
import statsmodels.stats.diagnostic as diag

white_test = diag.het_white(results.resid, results.model.exog)

# The output consists of the test statistic and the associated p-value
labels = ['Test Statistic', 'Test Statistic p-value', 'F-Statistic', 'F-Statistic p-value']
white_test_summary = dict(zip(labels, white_test))

for key, value in white_test_summary.items():
    print(f"{key}: {value}")
```

    Test Statistic: 76.30870031228504
    Test Statistic p-value: 1.7487062379433416e-05
    F-Statistic: 2.6143065587494148
    F-Statistic p-value: 6.246658985394247e-06

### Analyze the results of the initial White test

- Observe that `p-values` from the White test are close to zero
- Given that the null hypothesis in the White Test is that residuals are
  homoscedastic, in this case, **we reject the null hypothesis**,
  therefore we say that **residuals are heteroscedastic instead**, which
  is a problem and we should try to correct for this.

## Detect which variables affect the variance of the error term $\text{Var}(u)$

To accomplish this, we want to see which terms are significantly
affecting the estimated squared error term (i.e., $\widehat{u}^2$).
Therefore, we need to run the regression of the White test ourselves (or
something similar to it).

For example, we regress $\widehat{u}^2$ on the same independent
variables and their squares, but excluding the dummy variables and the
interaction between variables for simplicity:

``` python
# Calculate the squared residuals
data['resid_sq'] = results.resid ** 2

# Calculate the square of some of the independent variables

data['educ_sq'] = data['educ'] ** 2
data['exper_sq'] = data['exper'] ** 2
data['tenure_sq'] = data['tenure'] ** 2
data['numdep_sq'] = data['numdep'] ** 2

# Fit the auxiliary regression
aux_model = smf.ols('resid_sq ~ educ + exper + tenure + married + numdep + gender_female + skin_nonwhite + educ_sq + exper_sq + tenure_sq + numdep_sq ', data=data)
aux_results = aux_model.fit()

# Display the auxiliary regression results
print("\nAuxiliary Regression Summary (for White's Test):\n")
print(aux_results.summary())
```


    Auxiliary Regression Summary (for White's Test):

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:               resid_sq   R-squared:                       0.119
    Model:                            OLS   Adj. R-squared:                  0.101
    Method:                 Least Squares   F-statistic:                     6.337
    Date:                Sun, 03 Nov 2024   Prob (F-statistic):           7.20e-10
    Time:                        22:40:54   Log-Likelihood:                -2343.6
    No. Observations:                 526   AIC:                             4711.
    Df Residuals:                     514   BIC:                             4762.
    Df Model:                          11                                         
    Covariance Type:            nonrobust                                         
    =================================================================================
                        coef    std err          t      P>|t|      [0.025      0.975]
    ---------------------------------------------------------------------------------
    Intercept        16.8784     10.207      1.654      0.099      -3.175      36.931
    educ             -4.2397      1.598     -2.653      0.008      -7.380      -1.100
    exper             0.7940      0.289      2.749      0.006       0.227       1.361
    tenure            0.7152      0.362      1.975      0.049       0.004       1.427
    married          -2.7648      2.202     -1.256      0.210      -7.090       1.561
    numdep            2.0890      1.975      1.058      0.291      -1.790       5.968
    gender_female    -3.1165      1.933     -1.612      0.107      -6.914       0.681
    skin_nonwhite    -2.0770      3.067     -0.677      0.499      -8.102       3.948
    educ_sq           0.2346      0.066      3.575      0.000       0.106       0.364
    exper_sq         -0.0163      0.006     -2.605      0.009      -0.029      -0.004
    tenure_sq        -0.0070      0.012     -0.563      0.574      -0.031       0.017
    numdep_sq        -0.5881      0.493     -1.193      0.233      -1.556       0.380
    ==============================================================================
    Omnibus:                      616.162   Durbin-Watson:                   1.879
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):            38143.071
    Skew:                           5.683   Prob(JB):                         0.00
    Kurtosis:                      43.139   Cond. No.                     8.88e+03
    ==============================================================================

    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 8.88e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.

### Analyze the results of the White test regression

- We can see that some of the variables are significant in explaining
  the square of the estimated residuals
- In particular, the square of `educ` and `exper` are significant, so we
  will use them to try to correct for heteroscedasticity.

## Run a WLS estimation

To try fixing the problem of heteroscedasticity, lets run a WLS
estimation:

We will estimate the basic model again with OLS and the modified model
with Weighted Least Squares or WLS, for clarity:

``` python
# Load the dataset and construct dummies
data = pd.read_csv("data/wage.csv")
data = pd.get_dummies(data, columns=["gender", "skin"], drop_first=False, dtype=int)

# Initial OLS model
model_ols = smf.ols('wage ~ educ + exper + tenure + married + numdep + gender_female + skin_nonwhite', data=data)
results_ols = model_ols.fit()

# Estimate residuals
residuals = results_ols.resid

# Assuming variance is proportional to the square of 'educ' and 'exper' as concluded above
data['h'] = data['educ'] ** 2 + data['exper'] ** 2

# Calculate the inverse of the root of h
weights = 1 / np.sqrt(data['h'])

# Create weighted variables by multiplying each variable by the weights
data['w_wage'] = data['wage'] * weights
data['w_educ'] = data['educ'] * weights
data['w_exper'] = data['exper'] * weights
data['w_tenure'] = data['tenure'] * weights
data['w_married'] = data['married'] * weights
data['w_numdep'] = data['numdep'] * weights
data['w_gender_female'] = data['gender_female'] * weights
data['w_skin_nonwhite'] = data['skin_nonwhite'] * weights

# Run OLS with weighted variables (that is, WLS)
model_wls = smf.ols('w_wage ~ w_educ + w_exper + w_tenure + w_married + w_numdep + w_gender_female + w_skin_nonwhite', data=data)

results_wls = model_wls.fit()

# Output the results
print("\nWLS model summary:\n")
print(results_wls.summary())
```


    WLS model summary:

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                 w_wage   R-squared:                       0.382
    Model:                            OLS   Adj. R-squared:                  0.373
    Method:                 Least Squares   F-statistic:                     45.71
    Date:                Sun, 03 Nov 2024   Prob (F-statistic):           2.37e-50
    Time:                        22:40:54   Log-Likelihood:                 316.96
    No. Observations:                 526   AIC:                            -617.9
    Df Residuals:                     518   BIC:                            -583.8
    Df Model:                           7                                         
    Covariance Type:            nonrobust                                         
    ===================================================================================
                          coef    std err          t      P>|t|      [0.025      0.975]
    -----------------------------------------------------------------------------------
    Intercept          -0.4050      0.081     -5.000      0.000      -0.564      -0.246
    w_educ              0.7241      0.069     10.534      0.000       0.589       0.859
    w_exper             0.3463      0.063      5.492      0.000       0.222       0.470
    w_tenure            0.1565      0.027      5.731      0.000       0.103       0.210
    w_married           0.2079      0.253      0.821      0.412      -0.289       0.705
    w_numdep           -0.1505      0.078     -1.940      0.053      -0.303       0.002
    w_gender_female    -1.2054      0.204     -5.900      0.000      -1.607      -0.804
    w_skin_nonwhite     0.0921      0.350      0.263      0.793      -0.596       0.781
    ==============================================================================
    Omnibus:                      181.319   Durbin-Watson:                   1.740
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              612.932
    Skew:                           1.603   Prob(JB):                    8.01e-134
    Kurtosis:                       7.206   Cond. No.                         83.1
    ==============================================================================

    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

Now, we perform the White test once again to check whether errors are
now homoscedastic (which is what we want) or are still heteroscedastic:

``` python
white_test = diag.het_white(results_wls.resid, results_wls.model.exog)

# The output consists of the test statistic and the associated p-value
labels = ['Test Statistic', 'Test Statistic p-value', 'F-Statistic', 'F-Statistic p-value']
white_test_summary = dict(zip(labels, white_test))

for key, value in white_test_summary.items():
    print(f"{key}: {value}")
```

    Test Statistic: 42.91062121627159
    Test Statistic p-value: 0.140575269122546
    F-Statistic: 1.282743692289221
    F-Statistic p-value: 0.13561295323905587

As we can see, the `p-value` of the White test is larger than 0.1:

    Test Statistic p-value: 0.140575269122546

This means that we **fail to reject the null hypothesis of the White
test, and so we consider the residuals to be homoscedastic**.

**Conclusion**: we have managed to detect and correct for
heteroscedasticity!
