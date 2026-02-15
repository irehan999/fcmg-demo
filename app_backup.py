"""
FCMG AI Supply Chain Platform - Streamlit Demo
Demand Forecasting & Route Optimization
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from geopy.distance import geodesic
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# Page config
st.set_page_config(
    page_title="FCMG Supply Chain Platform",
    page_icon="🚚",
    layout="wide"
)

# ==================== ROUTE OPTIMIZATION FUNCTIONS ====================

def create_distance_matrix(locations):
    """Create distance matrix using geodesic distance"""
    n = len(locations)
    distance_matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                distance = 0
            else:
                distance = int(geodesic(locations[i], locations[j]).meters)
            row.append(distance)
        distance_matrix.append(row)
    return distance_matrix

def solve_vrp(distance_matrix, demands, num_vehicles, vehicle_capacity):
    """Solve Vehicle Routing Problem"""
    manager = pywrapcp.RoutingIndexManager(len(distance_matrix), num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)
    
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]
    
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return demands[from_node]
    
    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index, 0, [vehicle_capacity] * num_vehicles, True, 'Capacity'
    )
    
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.seconds = 10
    
    solution = routing.SolveWithParameters(search_parameters)
    
    if solution:
        routes = []
        for vehicle_id in range(num_vehicles):
            index = routing.Start(vehicle_id)
            route = {'vehicle_id': vehicle_id + 1, 'route_indices': [0], 'distance_km': 0, 'demand': 0}
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                if node > 0:
                    route['route_indices'].append(node)
                    route['demand'] += demands[node]
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route['distance_km'] += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
            route['route_indices'].append(0)
            route['distance_km'] = round(route['distance_km'] / 1000, 2)
            if len(route['route_indices']) > 2:
                routes.append(route)
        return routes
    return None

def route_optimization_tab():
    """Route Optimization Demo"""
    st.header("🚚 Route Optimization - Van Delivery Planning")
    
    # Demo data
    WAREHOUSE = {"name": "FCMG Warehouse", "lat": 33.6844, "lon": 73.0479}
    RETAILERS = [
        {"name": "Metro Store Saddar", "lat": 33.5975, "lon": 73.0551, "demand": 25},
        {"name": "Imtiaz Bahria Town", "lat": 33.5289, "lon": 73.1203, "demand": 18},
        {"name": "Al-Fatah Blue Area", "lat": 33.7185, "lon": 73.0551, "demand": 30},
        {"name": "CSD Store F-6", "lat": 33.7181, "lon": 73.0776, "demand": 22},
        {"name": "Utility Store G-9", "lat": 33.6938, "lon": 73.0357, "demand": 15},
        {"name": "Hyperstar Mall", "lat": 33.6539, "lon": 72.9883, "demand": 35},
        {"name": "Carrefour Express", "lat": 33.6688, "lon": 73.0679, "demand": 20},
        {"name": "Saveco Store", "lat": 33.6007, "lon": 73.0164, "demand": 28}
    ]
    
    # Auto-run demo on first load
    if 'route_optimized' not in st.session_state:
        with st.spinner("🚀 Running demo optimization..."):
            warehouse_loc = (WAREHOUSE['lat'], WAREHOUSE['lon'])
            retailer_locs = [(r['lat'], r['lon']) for r in RETAILERS]
            locations = [warehouse_loc] + retailer_locs
            demands = [0] + [r['demand'] for r in RETAILERS]
            
            distance_matrix = create_distance_matrix(locations)
            routes = solve_vrp(distance_matrix, demands, 3, 100)
            
            st.session_state['route_optimized'] = True
            st.session_state['routes'] = routes
            st.session_state['warehouse'] = WAREHOUSE
            st.session_state['retailers'] = RETAILERS
    
    # Display results
    routes = st.session_state.get('routes')
    if routes:
        st.success(f"✅ **Optimized {len(routes)} van routes for {len(RETAILERS)} retailers!**")
        
        # Metrics
        total_dist = sum(r['distance_km'] for r in routes)
        total_retailers = sum(len(r['route_indices'])-2 for r in routes)
        
        cols = st.columns(4)
        cols[0].metric("🚚 Vans Used", len(routes))
        cols[1].metric("🏪 Retailers", total_retailers)
        cols[2].metric("📏 Total Distance", f"{total_dist:.1f} km")
        cols[3].metric("⚡ Avg/Van", f"{total_dist/len(routes):.1f} km")
        
        # Map
        st.subheader("🗺️ Optimized Routes")
        fig = go.Figure()
        
        warehouse_loc = (WAREHOUSE['lat'], WAREHOUSE['lon'])
        retailer_locs = [(r['lat'], r['lon']) for r in RETAILERS]
        
        # Warehouse
        fig.add_trace(go.Scattermapbox(
            lat=[warehouse_loc[0]], lon=[warehouse_loc[1]],
            mode='markers', marker=dict(size=20, color='red'),
            name='Warehouse', text=['Warehouse']
        ))
        
        # Routes
        colors = ['blue', 'green', 'purple', 'orange', 'brown']
        for idx, route in enumerate(routes):
            route_lats, route_lons, route_names = [], [], []
            for node_idx in route['route_indices']:
                if node_idx == 0:
                    route_lats.append(warehouse_loc[0])
                    route_lons.append(warehouse_loc[1])
                    route_names.append('Warehouse')
                else:
                    route_lats.append(RETAILERS[node_idx-1]['lat'])
                    route_lons.append(RETAILERS[node_idx-1]['lon'])
                    route_names.append(RETAILERS[node_idx-1]['name'])
            
            fig.add_trace(go.Scattermapbox(
                lat=route_lats, lon=route_lons,
                mode='lines+markers',
                line=dict(width=3, color=colors[idx % len(colors)]),
                marker=dict(size=10, color=colors[idx % len(colors)]),
                name=f'Van {route["vehicle_id"]} ({route["distance_km"]} km)',
                text=route_names, hoverinfo='text'
            ))
        
        fig.update_layout(
            mapbox=dict(style='open-street-map', center=dict(lat=warehouse_loc[0], lon=warehouse_loc[1]), zoom=10),
            height=500, margin=dict(l=0, r=0, t=0, b=0), showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Route details
        st.subheader("📋 Route Details")
        for route in routes:
            retailer_names = [RETAILERS[i-1]['name'] for i in route['route_indices'][1:-1]]
            with st.expander(f"🚚 Van {route['vehicle_id']} - {len(retailer_names)} stops, {route['distance_km']} km, {route['demand']}/100 units"):
                st.write(f"**Route:** Warehouse → {' → '.join(retailer_names)} → Warehouse")
                st.write(f"**Capacity:** {route['demand']}/100 units ({route['demand']}%)")

# ==================== DEMAND FORECASTING FUNCTIONS ====================

@st.cache_resource
def load_model_and_data():
    """Load model and necessary data files"""
    try:
        model = joblib.load("models/demand_advanced_xgb.pkl")
        with open("models/demand_advanced_xgb_info.json", 'r') as f:
            model_info = json.load(f)
        
        # Load data files
        stores = pd.read_csv("data/stores.csv")
        train = pd.read_csv("data/train.csv", parse_dates=['date'], nrows=100000)  # Load subset for demo
        
        # Get unique values
        store_list = sorted(stores['store_nbr'].unique())
        family_list = sorted(train['family'].unique())
        
        return model, model_info, stores, store_list, family_list
    except Exception as e:
        st.error(f"Error loading model/data: {str(e)}")
        return None, None, None, None, None

def demand_forecasting_tab():
    """Demand Forecasting Demo"""
    st.header("📊 Demand Forecasting - Retail Sales Prediction")
    
    # Load model
    model, model_info, stores, store_list, family_list = load_model_and_data()
    
    if model is None:
        st.warning("⚠️ Model not found. Please ensure model file is in `streamlit_app/models/`")
        st.info("""
        **To enable forecasting:**
        1. Train XGBoost model in Kaggle
        2. Export: `joblib.dump(model, 'demand_advanced_xgb.pkl')`
        3. Place in `streamlit_app/models/` directory
        """)
        return
    
    st.success("✅ Model loaded successfully!")
    
    # Show model info
    with st.expander("📊 Model Information"):
        st.write(f"**Model Type:** XGBoost Regressor")
        st.write(f"**Features:** {len(model_info['features'])} features")
        st.write(f"**Key Features:** Lags (7 days), Seasonality (Fourier), Trend, Holidays, Oil prices")
        st.write(f"**Available Stores:** {len(store_list)} stores")
        st.write(f"**Product Families:** {len(family_list)} categories")
    
    # Demo prediction interface
    st.subheader("🔮 Sales Prediction Demo")
    
    st.info("""
    **Note:** This is a simplified demo interface. The actual model requires 44 engineered features including:
    - 7-day sales lags
    - Fourier seasonality features (weekly, monthly, annual)
    - Trend components
    - Holiday indicators
    - Oil prices
    - Store/family rankings
    
    For production use, implement the full feature engineering pipeline from the Kaggle notebook.
    """)
    
    # Simple selection interface
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_store = st.selectbox("Select Store", store_list[:10])  # Show first 10
    
    with col2:
        selected_family = st.selectbox("Select Product Family", family_list[:10])  # Show first 10
    
    with col3:
        forecast_days = st.slider("Forecast Days", 7, 30, 14)
    
    # Show store info
    store_info = stores[stores['store_nbr'] == selected_store].iloc[0]
    st.write(f"**Store Info:** {store_info['city']}, {store_info['state']} | Type: {store_info['type']} | Cluster: {store_info['cluster']}")
    
    if st.button("🚀 Predict Sales", type="primary"):
        st.warning("""
        **Demo Mode:** Full prediction requires complete feature engineering from the Kaggle notebook.
        
        To implement:
        1. Load all auxiliary data (oil, holidays, transactions)
        2. Create 7-day lags from historical sales
        3. Generate Fourier seasonality features
        4. Calculate trend and rankings
        5. Merge all features matching model's expected 44 features
        6. Apply log transformation: `np.log(sales + 1)`
        7. Make prediction
        8. Reverse transform: `np.exp(prediction) - 1`
        
        See the Kaggle notebook for complete implementation.
        """)
        
        # Show sample visualization
        dates = pd.date_range(datetime.now(), periods=forecast_days, freq='D')
        sample_sales = np.random.randint(50, 300, forecast_days)
        
        df_viz = pd.DataFrame({'Date': dates, 'Predicted Sales': sample_sales})
        
        fig = px.line(df_viz, x='Date', y='Predicted Sales', title=f"Sample Sales Forecast - Store {selected_store}, {selected_family}")
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("💡 **Note:** This is sample data. Implement full feature engineering for actual predictions.")

# ==================== MAIN APP ====================

def main():
    st.title("🚚 FCMG Supply Chain Platform - Demo")
    st.markdown("**AI-Powered Demand Forecasting & Route Optimization**")
    
    # Tabs
    tab1, tab2 = st.tabs(["🚚 Route Optimization", "📊 Demand Forecasting"])
    
    with tab1:
        route_optimization_tab()
    
    with tab2:
        demand_forecasting_tab()
    
    # Footer
    st.markdown("---")
    st.markdown("*Demo Version - FCMG Supply Chain Platform*")

if __name__ == "__main__":
    main()
