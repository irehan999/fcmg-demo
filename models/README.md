# Models Folder

Place your trained XGBoost model here:

## Required Files:
- `demand_advanced_xgb.pkl` - Your trained XGBoost model
- `demand_advanced_xgb_info.json` - (Optional) Model feature information

## Example model_info.json:
```json
{
  "features": ["store", "item", "dayofweek", "quarter", "month", "year", "dayofyear", "dayofmonth", "weekofyear"],
  "model_type": "XGBoost",
  "trained_date": "2026-02-05"
}
```

After training your model in Kaggle, export it using:
```python
import joblib
joblib.dump(model, 'demand_advanced_xgb.pkl')
```
