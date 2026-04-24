# Part B: Business Case Analysis:

## Scenario: Promotion Effectiveness at a Fashion Retail Chain

### B1. Problem Formulation :
**(a) ML Problem Formulation Target Variable:** Total number of items sold (Sales Volume) at the store-month level.

_Candidate Input Features:_ 

_Store-related:_ Location type (urban/rural), store size, local competition density, and customer demographics.

_Promotion-related:_ Promotion type (Flat Discount, BOGO, etc.), duration, and discount depth.

_Contextual/Temporal:_  Monthly footfall, seasonality (month of year), and presence of festivals/holidays.

_ML Problem Type:_ `Regression`.

**Justification:** The goal is to predict a continuous numerical value (quantity of items). By predicting the `volume for each of the five promotion types for a specific store, the business can select the one that yields the highest predicted value`.

**(b) Items Sold vs. Total Sales Revenue :**
Using items sold is more reliable because revenue is heavily influenced by the price point of the items being promoted. For example, a `"Category-Specific Offer"` on high-end coats might generate `high revenue but low volume`, whereas a `"BOGO"` on socks generates `high volume but lower revenue`.

_Broader Principle:_ This illustrates the importance of `Proximal vs. Distal Objectives`. The target variable must align strictly with the business lever being pulled (promotions are generally designed to clear inventory and drive traffic/volume) and should minimize "noise" from external factors like price inflation or product mix changes.

**(c) Alternative Modelling Strategy :**
An alternative is a `Clustered Modeling approach or a Hierarchical (Multi-level) Model.`

**Justification:** Instead of one global model that might wash out local nuances, or 50 individual models (which would suffer from data sparsity), we can group stores by location type (Urban, Semi-Urban, Rural). This allows the model to learn specific coefficients for "Urban" behavior while still benefiting from a larger pool of data than a single-store model would provide.

### B2. Data and EDA Strategy:

**(a) Table Joins and Data Grain :**
The tables should be joined using a series of `Left Joins` starting with the transaction table as the base.

**Transactions + Promotion Details** on `promo_id`.

**Transactions + Store Attributes** on `store_id`.

**Transactions + Calendar** on `transaction_date`.

**Final Data Grain:** One row = **One Store per Month per Promotion Type**.

**Aggregations:** Transactions must be _summed (total items) and averaged (average basket size, average discount)_ grouped by `store_id` and `month`.

**(b) EDA Plan :**

**1. Boxplot of Items Sold by Promo Type:**

_What to Look For :_ Median performance and variance for each promotion.

_Impact on Modeling :_ Identifies if certain promos are consistently superior regardless of store.

**2. Time-Series Plot (Total Volume vs. Time) :**

_What to Look For :_ `Monthly seasonality and yearly growth trends`.

_Impact on Modeling :_ Directs the creation of `month-of-year` or `lagged sales` features.

**3. Correlation Matrix (Heatmap) :**

_What to Look For :_ `High correlation` between footfall, competition, and sales.

_Impact on Modeling :_ Helps in feature selection and identifies `potential multicollinearity`.

**4. Scatter Plot: Footfall vs. Items Sold :** 

_What to Look For :_ Linearity and outliers in the relationship.

_Impact on Modeling :_ Determines if `non-linear transformations` (e.g., Log scale) are needed for footfall.

**(c) Handling Unpromoted Transactions :**
An 80% `No Promotion` rate creates a baseline bias. The model might struggle to learn the specific `uplift` caused by a promotion.

**Steps to Address:**

_1. Uplift Modeling:_ Instead of predicting total sales, predict the difference in sales between a promotion and the `No Promotion` baseline.

_2. Stratified Sampling:_ Ensure training batches contain a balanced representation of all promotion types to prevent the model from defaulting to `No Promotion` logic.


### B3. Model Evaluation and Deployment :

**(a) Train-Test Split and Metrics**

- **Split Strategy:**
 Time-Series Split (Forward Chaining). For example, train on Year 1 & 2, and test on Year 3.

- **Why Random is Inappropriate:**
 Random splitting leads to data leakage. If you train on data from June and test on data from May of the same year, the model `cheats` by knowing future trends (like inflation or fashion cycles) that wouldn't be available in a real-world deployment.

- **Metrics: * MAE (Mean Absolute Error):** Easy to communicate; tells us exactly how many items off we are on average.

- **RMSE (Root Mean Squared Error):**
Penalizes large errors, which is useful if a massive overstock/understock is costly.

**(b) Communicating Feature Importance** 
_To explain why Store 12 gets different recommendations in December vs. March:_

1. **SHAP Values or LIME:** Use these to show which features "pushed" the prediction.

2. **The Communication:** _"In December, the `Holiday Flag` and `High Footfall` features made `BOGO` the winner because customers are gift-shopping. In March, low `Footfall` and no festivals make `Flat Discounts` more effective at enticing price-sensitive local shoppers."_

**(c) Deployment and Monitoring :**

1. **Model Serialization:** Save the trained model using a format like `Joblib` or `Pickle`.

2. **Inference Pipeline:** At the start of each month, a script pulls the latest Store Attributes and the upcoming Calendar flags. It creates 5 `dummy` rows for each store _(one for each promo type)_.

3. **Prediction:** The model scores all 5 rows; the promo with the highest predicted volume is output as the recommendation.

4. **Monitoring:**

- **Performance Decay:** Compare actual items sold vs. predicted volume at month-end.

- **Data Drift:** Monitor if footfall or demographics have shifted significantly from the training distribution _(e.g., a new competitor opens nearby)_. Trigger retraining if error rates exceed a pre-defined threshold.
