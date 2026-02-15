# Demo Data Configuration

## Route Optimization Demo

### Warehouse (Depot)
- **Name:** FCMG Warehouse Islamabad
- **Location:** 33.7490°N, 72.8536°E (Islamabad, Pakistan)

### Demo Retailers (8 locations)

| # | Retailer Name | Location (Lat, Lon) | Demand (units) |
|---|---------------|---------------------|----------------|
| 1 | Metro Store - F-10 | 33.6844, 73.0479 | 20 |
| 2 | Al-Fatah - Blue Area | 33.7182, 72.9842 | 15 |
| 3 | Imtiaz - G-11 | 33.5651, 73.0169 | 25 |
| 4 | Carrefour - PWD | 33.6007, 73.0679 | 30 |
| 5 | Naheed - F-7 | 33.6973, 73.0515 | 18 |
| 6 | Utility Store - I-8 | 33.6502, 72.9875 | 22 |
| 7 | Hyperstar - DHA | 33.7294, 73.0931 | 28 |
| 8 | CSD Store - Rawalpindi | 33.5873, 72.9234 | 16 |

**Total Demand:** 174 units

### Vehicle Configuration
- **Number of Vans:** 3
- **Capacity per Van:** 100 units
- **Total Fleet Capacity:** 300 units

### Expected Demo Results
- ✅ All 8 retailers served
- 🚚 3 vans deployed
- 📏 Optimized routes with minimal distance
- ⚡ Real-time calculation using OR-Tools CVRP solver

## Demand Forecasting Demo

Place your trained XGBoost model in:
- `models/demand_advanced_xgb.pkl`
- `models/demand_advanced_xgb_info.json` (optional)

Dataset location (optional):
- `data/train.csv`

The app will:
1. Load model automatically on startup
2. Allow selection of stores and products
3. Generate forecasts for date range
4. Display interactive charts and metrics
