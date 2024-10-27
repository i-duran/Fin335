# Multiple Regression - Extended Applications

## Dummy variables

We will be using a modified data set originally obtained from
Wooldridge’ Introductory Econometrics data set on wages.

The modified data set used in this example has the following variables:

- `wage`: average hourly earnings
- `educ`: years of education
- `exper`: years potential experience
- `tenure`: years with current employer
- `married`: =1 if married
- `numdep`: number of dependents
- `gender`: male or female
- `skin`: color of the skin (white or nonwhite)

``` python
import pandas as pd
import statsmodels.formula.api as smf

df = pd.read_csv("./data/wage.csv")
df.head()
```

|     | wage | educ | exper | tenure | married | numdep | gender | skin  |
|-----|------|------|-------|--------|---------|--------|--------|-------|
| 0   | 3.10 | 11   | 2     | 0      | 0       | 2      | female | white |
| 1   | 3.24 | 12   | 22    | 2      | 1       | 3      | female | white |
| 2   | 3.00 | 11   | 2     | 0      | 0       | 2      | male   | white |
| 3   | 6.00 | 8    | 44    | 28     | 1       | 0      | male   | white |
| 4   | 5.30 | 12   | 7     | 2      | 1       | 1      | male   | white |

Let’s construct the dummy variables for `gender` and `skin`:

``` python
df = pd.get_dummies(df, columns=["gender", "skin"], drop_first=False, dtype=int)
df.head()
```

|     | wage | educ | exper | tenure | married | numdep | gender_female | gender_male | skin_nonwhite | skin_white |
|-----|------|------|-------|--------|---------|--------|---------------|-------------|---------------|------------|
| 0   | 3.10 | 11   | 2     | 0      | 0       | 2      | 1             | 0           | 0             | 1          |
| 1   | 3.24 | 12   | 22    | 2      | 1       | 3      | 1             | 0           | 0             | 1          |
| 2   | 3.00 | 11   | 2     | 0      | 0       | 2      | 0             | 1           | 0             | 1          |
| 3   | 6.00 | 8    | 44    | 28     | 1       | 0      | 0             | 1           | 0             | 1          |
| 4   | 5.30 | 12   | 7     | 2      | 1       | 1      | 0             | 1           | 0             | 1          |

Now, let’s run a regression using these dummy variables:

``` python
model = smf.ols("wage ~ educ + exper + tenure + married + numdep + gender_female + skin_nonwhite", data=df)
results = model.fit()
print(results.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                   wage   R-squared:                       0.370
    Model:                            OLS   Adj. R-squared:                  0.362
    Method:                 Least Squares   F-statistic:                     43.54
    Date:                Sun, 27 Oct 2024   Prob (F-statistic):           2.53e-48
    Time:                        21:34:41   Log-Likelihood:                -1311.4
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

### Analyze the results

- Do a significance analysis of the coefficients
- Interpret the meaning of the coefficients using the corresponding
  units
- Analyze the Adjusted R-squared

## Polynomial regression

Let’s consider experience as having potentially **non-linear effects**.
That is, we add `exper^2` to our regression to account for this
non-linearity:

``` python
# add the exper-squared to the dataset (df)
df["exper^2"] = df["exper"] ** 2
df.head()
```

|     | wage | educ | exper | tenure | married | numdep | gender_female | gender_male | skin_nonwhite | skin_white | exper^2 |
|-----|------|------|-------|--------|---------|--------|---------------|-------------|---------------|------------|---------|
| 0   | 3.10 | 11   | 2     | 0      | 0       | 2      | 1             | 0           | 0             | 1          | 4       |
| 1   | 3.24 | 12   | 22    | 2      | 1       | 3      | 1             | 0           | 0             | 1          | 484     |
| 2   | 3.00 | 11   | 2     | 0      | 0       | 2      | 0             | 1           | 0             | 1          | 4       |
| 3   | 6.00 | 8    | 44    | 28     | 1       | 0      | 0             | 1           | 0             | 1          | 1936    |
| 4   | 5.30 | 12   | 7     | 2      | 1       | 1      | 0             | 1           | 0             | 1          | 49      |


``` python
# specify and estimate the new model with exper^2
model2 = smf.ols("wage ~ educ + exper + exper^2 + tenure + married + numdep + gender_female + skin_nonwhite", data=df)
results2 = model2.fit()
print(results2.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                   wage   R-squared:                       0.371
    Model:                            OLS   Adj. R-squared:                  0.361
    Method:                 Least Squares   F-statistic:                     38.10
    Date:                Sun, 27 Oct 2024   Prob (F-statistic):           1.45e-47
    Time:                        21:34:41   Log-Likelihood:                -1311.2
    No. Observations:                 526   AIC:                             2640.
    Df Residuals:                     517   BIC:                             2679.
    Df Model:                           8                                         
    Covariance Type:            nonrobust                                         
    =================================================================================
                        coef    std err          t      P>|t|      [0.025      0.975]
    ---------------------------------------------------------------------------------
    Intercept        -2.0180      0.786     -2.569      0.010      -3.561      -0.475
    educ              0.5767      0.052     11.043      0.000       0.474       0.679
    exper             0.0620      0.067      0.921      0.357      -0.070       0.194
    exper ^ 2        -0.0395      0.065     -0.606      0.545      -0.168       0.089
    tenure            0.1379      0.021      6.517      0.000       0.096       0.180
    married           0.4583      0.296      1.550      0.122      -0.122       1.039
    numdep            0.1462      0.109      1.342      0.180      -0.068       0.360
    gender_female    -1.7677      0.267     -6.610      0.000      -2.293      -1.242
    skin_nonwhite    -0.1180      0.428     -0.276      0.783      -0.959       0.723
    ==============================================================================
    Omnibus:                      187.886   Durbin-Watson:                   1.790
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              720.337
    Skew:                           1.610   Prob(JB):                    3.81e-157
    Kurtosis:                       7.743   Cond. No.                         202.
    ==============================================================================

    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

### Analyze the results

- Do a significance analysis of the coefficients
- Interpret the meaning of the coefficients using the corresponding
  units
- Analyze the Adjusted R-squared

**Conclusion**: given that the estimated coefficients for `exper` and
`exper^2` are not significant, we conclude that experience has no linear
effects on the hourly wage.
