# Linear Regression

In this guide we demonstrate how to run a linear regression with one
independent variable.

We will be using the CAPM specification as example, as well as the
`capm.csv` data file:

According to the CAPM specification:

$$R_a - R_f = \beta_0 + \beta_1(R_m - R_f)$$

- $R_a - R_f$ is the dependent variable, where $R_a$ is the return of
  the stock, and $R_f$ is the return of the risk-free asset,
  corresponding to the 3 months treasury bills rate (‘USTB3M’ column in
  the file).
- $R_m - R_f$ is the explanatory or independent variable, where $R_m$ is
  the return of the market, which can be obtained from the S&P index
  found in the column ‘SANDP’ (: the SANDP column is the index value,
  not the return).
- **Note**: the variable ‘USTB3M’ is already in percentage points (that is, it
  has been multiplied by 100 already). Therefore, for consistency
  purposes, it is recommended that you also multiple $R_a$ and $R_m$
  when calculating the returns of the asset and the market,
  respectively.

## Step 1: Importing Libraries And Loading The Data

``` python
import pandas as pd
import statsmodels.formula.api as smf

df = pd.read_csv("./data/capm.csv", index_col=0)
```

## Step 2: Summary Statistics Of The Data

``` python
print(df.describe())
print(df.head())
```

                 SANDP        FORD          GE   MICROSOFT      ORACLE      USTB3M
    count   194.000000  194.000000  194.000000  194.000000  194.000000  194.000000
    mean   1468.897883   11.430361   26.593866   34.754433   25.972474    1.241495
    std     472.527116    3.552879    7.384621   15.044991   12.038702    1.570625
    min     729.570007    1.870000    8.510000   16.150000    7.860000    0.010000
    25%    1127.207459    8.527500   20.719999   25.867500   13.950000    0.080000
    50%    1323.525024   11.740000   26.950000   28.350000   22.710000    0.480000
    75%    1854.725037   14.167500   32.697501   38.265001   36.474999    1.725000
    max    2816.449951   17.650000   41.400002   95.010002   51.590000    5.160000
                  SANDP   FORD         GE  MICROSOFT     ORACLE  USTB3M
    Date                                                               
    Jan-02  1130.199951  15.30  37.150002  31.855000  17.260000    1.68
    Feb-02  1106.729980  14.88  38.500000  29.170000  16.620001    1.76
    Mar-02  1147.390015  16.49  37.400002  30.155001  12.800000    1.83
    Apr-02  1076.920044  16.00  31.549999  26.129999  10.040000    1.75
    May-02  1067.140015  17.65  31.139999  25.455000   7.920000    1.76

# Step 3: Linear Regression Estimation

First we need to prepare the data and calculate the required variables.
We will do this for **FORD** only at this point:

``` python
df["FORD_r"] = df["FORD"].pct_change() * 100
df["FORD_r_Rf"] = df["FORD_r"] - df["USTB3M"]
df["SANDP_r"] = df["SANDP"].pct_change() * 100
df["SANDP_r_Rf"] = df["SANDP_r"] - df["USTB3M"]

print(df.head())
print(df.describe())
df.dropna() # drop missing observations
```

                  SANDP   FORD         GE  MICROSOFT     ORACLE  USTB3M  \
    Date                                                                  
    Jan-02  1130.199951  15.30  37.150002  31.855000  17.260000    1.68   
    Feb-02  1106.729980  14.88  38.500000  29.170000  16.620001    1.76   
    Mar-02  1147.390015  16.49  37.400002  30.155001  12.800000    1.83   
    Apr-02  1076.920044  16.00  31.549999  26.129999  10.040000    1.75   
    May-02  1067.140015  17.65  31.139999  25.455000   7.920000    1.76   

               FORD_r  FORD_r_Rf   SANDP_r  SANDP_r_Rf  
    Date                                                
    Jan-02        NaN        NaN       NaN         NaN  
    Feb-02  -2.745098  -4.505098 -2.076621   -3.836621  
    Mar-02  10.819892   8.989892  3.673889    1.843889  
    Apr-02  -2.971498  -4.721498 -6.141763   -7.891763  
    May-02  10.312500   8.552500 -0.908148   -2.668148  
                 SANDP        FORD          GE   MICROSOFT      ORACLE  \
    count   194.000000  194.000000  194.000000  194.000000  194.000000   
    mean   1468.897883   11.430361   26.593866   34.754433   25.972474   
    std     472.527116    3.552879    7.384621   15.044991   12.038702   
    min     729.570007    1.870000    8.510000   16.150000    7.860000   
    25%    1127.207459    8.527500   20.719999   25.867500   13.950000   
    50%    1323.525024   11.740000   26.950000   28.350000   22.710000   
    75%    1854.725037   14.167500   32.697501   38.265001   36.474999   
    max    2816.449951   17.650000   41.400002   95.010002   51.590000   

               USTB3M      FORD_r   FORD_r_Rf     SANDP_r  SANDP_r_Rf  
    count  194.000000  193.000000  193.000000  193.000000  193.000000  
    mean     1.241495    0.733334   -0.505889    0.539248   -0.699975  
    std      1.570625   14.699870   14.926924    4.074064    4.430910  
    min      0.010000  -57.884615  -58.574615  -16.793085  -17.483085  
    25%      0.080000   -5.942492   -7.793910   -1.655718   -3.113510  
    50%      0.480000   -1.054713   -1.867900    1.106065   -0.336936  
    75%      1.725000    4.773869    4.180123    2.930465    2.065107  
    max      5.160000  127.376426  127.216426   10.589550   10.569550  

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|        | SANDP       | FORD  | GE        | MICROSOFT | ORACLE    | USTB3M | FORD_r     | FORD_r_Rf  | SANDP_r   | SANDP_r_Rf |
|--------|-------------|-------|-----------|-----------|-----------|--------|------------|------------|-----------|------------|
| Date   |             |       |           |           |           |        |            |            |           |            |
| Feb-02 | 1106.729980 | 14.88 | 38.500000 | 29.170000 | 16.620001 | 1.76   | -2.745098  | -4.505098  | -2.076621 | -3.836621  |
| Mar-02 | 1147.390015 | 16.49 | 37.400002 | 30.155001 | 12.800000 | 1.83   | 10.819892  | 8.989892   | 3.673889  | 1.843889   |
| Apr-02 | 1076.920044 | 16.00 | 31.549999 | 26.129999 | 10.040000 | 1.75   | -2.971498  | -4.721498  | -6.141763 | -7.891763  |
| May-02 | 1067.140015 | 17.65 | 31.139999 | 25.455000 | 7.920000  | 1.76   | 10.312500  | 8.552500   | -0.908148 | -2.668148  |
| Jun-02 | 989.820007  | 16.00 | 29.049999 | 27.350000 | 9.470000  | 1.73   | -9.348442  | -11.078442 | -7.245535 | -8.975535  |
| ...    | ...         | ...   | ...       | ...       | ...       | ...    | ...        | ...        | ...       | ...        |
| Oct-17 | 2583.209961 | 12.27 | 20.160000 | 83.180000 | 50.900002 | 1.09   | 2.506266   | 1.416266   | 2.459544  | 1.369544   |
| Nov-17 | 2645.100098 | 12.52 | 18.290001 | 84.169998 | 49.060001 | 1.25   | 2.037490   | 0.787490   | 2.395862  | 1.145862   |
| Dec-17 | 2683.729980 | 12.49 | 17.450001 | 85.540001 | 47.279999 | 1.34   | -0.239617  | -1.579617  | 1.460432  | 0.120432   |
| Jan-18 | 2816.449951 | 10.97 | 16.170000 | 95.010002 | 51.590000 | 1.43   | -12.169736 | -13.599736 | 4.945355  | 3.515355   |
| Feb-18 | 2715.219971 | 10.61 | 14.110000 | 93.769997 | 50.669998 | 1.59   | -3.281677  | -4.871677  | -3.594240 | -5.184240  |

<p>193 rows × 10 columns</p>
</div>

Now we can go ahead and use the `statsmodels (smf)` library to estimate
the linear equation based on the CAPM specification:

``` python
print("\n\n--- Simple Regression ---")

model = smf.ols("FORD_r_Rf ~ SANDP_r_Rf", data=df)
results = model.fit()
print(results.summary())
```



    --- Simple Regression ---
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:              FORD_r_Rf   R-squared:                       0.306
    Model:                            OLS   Adj. R-squared:                  0.302
    Method:                 Least Squares   F-statistic:                     84.17
    Date:                Mon, 30 Sep 2024   Prob (F-statistic):           7.41e-17
    Time:                        08:29:41   Log-Likelihood:                -759.83
    No. Observations:                 193   AIC:                             1524.
    Df Residuals:                     191   BIC:                             1530.
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept      0.7983      0.909      0.879      0.381      -0.994       2.591
    SANDP_r_Rf     1.8632      0.203      9.174      0.000       1.463       2.264
    ==============================================================================
    Omnibus:                      198.494   Durbin-Watson:                   2.293
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             7256.383
    Skew:                           3.810   Prob(JB):                         0.00
    Kurtosis:                      32.057   Cond. No.                         4.54
    ==============================================================================

    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

We can also print only the estimated coefficients $\beta_0$ and
$\beta_1$:

``` python
# Print just the estimated coefficients
print("Coeficients:")
print(results.params)
```

    Coeficients:
    Intercept     0.798295
    SANDP_r_Rf    1.863186
    dtype: float64

# Step 4: Interpretation and Analysis

Up to this point, you should be able to analyze the following:

- Analyze the slope coefficient and its interpretation, based on the
  CAPM theory.
- Analyze the $R^2$ and its interpretation.
