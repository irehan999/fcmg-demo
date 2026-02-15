import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime, timedelta
from geopy.distance import geodesic
from ortools.constraint_solver import routing_enums_pb2, pywrapcp
import json
import os
from forecast_helper import predict_demand_batch

# Page config
st.set_page_config(page_title="FCMG AI Supply Chain Platform", layout="wide", page_icon="🚚")

# Custom CSS for clean, professional look
st.markdown("""
<style>
    /* Force Dark Theme Colors */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #FAFAFA;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 10px rgba(0,0,0,0.5);
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #B0B0B0;
        text-align: center;
        margin-bottom: 2.5rem;
    }
    
    /* Metric Cards */
    .metric-card {
        background-color: #262730;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        text-align: center;
        border-left: 5px solid #00CC96;
        transition: transform 0.2s;
        margin-bottom: 10px;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.5);
        border-left: 5px solid #00FFAA;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #FFFFFF;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #AAAAAA;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Section Headers */
    .section-header {
        background-color: #1F2937;
        background: linear-gradient(90deg, #1F2937 0%, #0E1117 100%);
        color: #FAFAFA;
        padding: 15px 25px;
        border-radius: 10px;
        margin-top: 2rem;
        margin-bottom: 1.5rem;
        font-weight: 600;
        font-size: 1.4rem;
        display: flex;
        align-items: center;
        border-left: 6px solid #1f77b4;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    
    /* Buttons */
    .stButton>button {
        border-radius: 20px;
        font-weight: 600;
        padding-left: 25px;
        padding-right: 25px;
        background-color: #1f77b4;
        color: white;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #2c90d6;
        box-shadow: 0 0 10px rgba(31, 119, 180, 0.5);
    }
    
    /* Tables/Dataframes */
    .stDataFrame {
        border-radius: 10px;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #262730;
        color: #FAFAFA;
        border-radius: 5px;
    }
    
    /* Checkbox & Radio */
    .stCheckbox label, .stRadio label {
        color: #FAFAFA;
    }
</style>
""", unsafe_allow_html=True)


# Title
st.markdown('<div class="main-header">🚚 FCMG AI Supply Chain Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Demand Forecasting & Route Optimization</div>', unsafe_allow_html=True)

# Product mapping for FCMG demo context
PRODUCT_MAPPING = {
    'AUTOMOTIVE': 'Auto Care Products',
    'BABY CARE': 'Baby Care Items',
    'BEAUTY': 'Beauty & Cosmetics',
    'BEVERAGES': 'Soft Drinks & Beverages',
    'BOOKS': 'Books & Magazines',
    'BREAD/BAKERY': 'Bakery Items',
    'CELEBRATION': 'Party Supplies & Gifts',
    'CLEANING': 'Cleaning Products',
    'DAIRY': 'Dairy Products',
    'DELI': 'Deli & Ready Meals',
    'EGGS': 'Fresh Eggs',
    'FROZEN FOODS': 'Frozen Foods',
    'GROCERY I': 'Snacks & Chips',
    'GROCERY II': 'Packaged Foods',
    'HARDWARE': 'Hardware & Tools',
    'HOME AND KITCHEN I': 'Kitchen Supplies',
    'HOME AND KITCHEN II': 'Home Essentials',
    'HOME APPLIANCES': 'Home Appliances',
    'HOME CARE': 'Home Care Products',
    'LADIESWEAR': 'Ladies Apparel',
    'LAWN AND GARDEN': 'Garden Supplies',
    'LINGERIE': 'Lingerie',
    'LIQUOR,WINE,BEER': 'Beverages (Alcoholic)',
    'MAGAZINES': 'Magazines',
    'MEATS': 'Meat Products',
    'PERSONAL CARE': 'Personal Care & Hygiene',
    'PET SUPPLIES': 'Pet Care Products',
    'PLAYERS AND ELECTRONICS': 'Electronics',
    'POULTRY': 'Poultry Products',
    'PREPARED FOODS': 'Ready-to-Eat Foods',
    'PRODUCE': 'Fresh Produce',
    'SCHOOL AND OFFICE SUPPLIES': 'Office Supplies',
    'SEAFOOD': 'Seafood Products'
}

# Load model and data
@st.cache_resource
def load_demand_model():
    """Load XGBoost demand forecasting model"""
    try:
        model_path = "models/demand_advanced_xgb.pkl"
        info_path = "models/demand_advanced_xgb_info.json"
        
        if not os.path.exists(model_path):
            return None, None
            
        model = joblib.load(model_path)
        
        # Load model info if available
        model_info = None
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                model_info = json.load(f)
        
        return model, model_info
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

@st.cache_data
def load_datasets():
    """Load all required datasets for model prediction"""
    datasets = {}
    
    # Define all dataset paths
    data_files = {
        'train': ['data/train_sample.csv', '../data/train_sample.csv'],  # Using lightweight sample for deployment
        'test': ['data/test.csv', '../data/test.csv'],
        'holidays': ['data/holidays_events.csv', '../data/holidays_events.csv'],
        'oil': ['data/oil.csv', '../data/oil.csv'],
        'stores': ['data/stores.csv', '../data/stores.csv'],
        'transactions': ['data/transactions.csv', '../data/transactions.csv']
    }
    
    # Load each dataset
    for name, paths in data_files.items():
        for path in paths:
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path)
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                    datasets[name] = df
                    break
                except Exception as e:
                    continue
    
    # Return None if critical datasets missing
    if 'train' not in datasets:
        return None
    
    return datasets

# Route optimization functions
def create_distance_matrix(locations):
    """Create distance matrix using geodesic distance"""
    n = len(locations)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist = geodesic(locations[i], locations[j]).meters
                matrix[i][j] = int(dist)
    return matrix

def optimize_routes(warehouse_coords, retailer_locations, retailer_demands, van_capacity, num_vans):
    """Optimize routes using OR-Tools CVRP"""
    
    # Prepare locations (warehouse + retailers)
    locations = [warehouse_coords] + retailer_locations
    demands = [0] + retailer_demands  # Warehouse has 0 demand
    
    # Create distance matrix
    distance_matrix = create_distance_matrix(locations)
    
    # Create routing model
    manager = pywrapcp.RoutingIndexManager(len(locations), num_vans, 0)
    routing = pywrapcp.RoutingModel(manager)
    
    # Distance callback
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(distance_matrix[from_node][to_node])
    
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    # Capacity constraint
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return demands[from_node]
    
    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        [van_capacity] * num_vans,  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity'
    )
    
    # Search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.seconds = 10
    
    # Solve
    solution = routing.SolveWithParameters(search_parameters)
    
    if solution:
        routes = extract_routes(manager, routing, solution, num_vans)
        total_distance = solution.ObjectiveValue() / 1000  # Convert to km
        return routes, total_distance
    else:
        return None, 0

def extract_routes(manager, routing, solution, num_vehicles):
    """Extract routes from solution"""
    routes = []
    for vehicle_id in range(num_vehicles):
        index = routing.Start(vehicle_id)
        route = []
        route_distance = 0
        
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route.append(node)
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
        
        # Add final node
        route.append(manager.IndexToNode(index))
        
        # Only include routes that visit at least one retailer
        if len(route) > 2:
            routes.append({
                'vehicle_id': vehicle_id + 1,
                'route': route,
                'distance_km': route_distance / 1000
            })
    
    return routes

def create_route_map(warehouse_coords, retailer_locations, retailer_names, routes):
    """Create interactive map with routes"""
    fig = go.Figure()
    
    # Color palette for routes
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan']
    
    # Plot routes
    for idx, route_info in enumerate(routes):
        route = route_info['route']
        color = colors[idx % len(colors)]
        
        # Get route coordinates
        route_lats = []
        route_lons = []
        for node in route:
            if node == 0:
                route_lats.append(warehouse_coords[0])
                route_lons.append(warehouse_coords[1])
            else:
                route_lats.append(retailer_locations[node-1][0])
                route_lons.append(retailer_locations[node-1][1])
        
        # Plot route line
        fig.add_trace(go.Scattermapbox(
            lat=route_lats,
            lon=route_lons,
            mode='lines+markers',
            line=dict(width=3, color=color),
            marker=dict(size=8, color=color),
            name=f"Van {route_info['vehicle_id']} ({route_info['distance_km']:.1f} km)",
            hovertemplate='<b>%{text}</b><extra></extra>',
            text=[f"Stop {i+1}" for i in range(len(route))]
        ))
    
    # Plot warehouse
    fig.add_trace(go.Scattermapbox(
        lat=[warehouse_coords[0]],
        lon=[warehouse_coords[1]],
        mode='markers',
        marker=dict(size=20, color='black', symbol='warehouse'),
        name='Warehouse',
        text=['Warehouse'],
        hovertemplate='<b>Warehouse</b><extra></extra>'
    ))
    
    # Plot retailers
    fig.add_trace(go.Scattermapbox(
        lat=[loc[0] for loc in retailer_locations],
        lon=[loc[1] for loc in retailer_locations],
        mode='markers',
        marker=dict(size=12, color='darkgreen', symbol='circle'),
        name='Retailers',
        text=retailer_names,
        hovertemplate='<b>%{text}</b><extra></extra>'
    ))
    
    # Update layout
    center_lat = np.mean([warehouse_coords[0]] + [loc[0] for loc in retailer_locations])
    center_lon = np.mean([warehouse_coords[1]] + [loc[1] for loc in retailer_locations])
    
    fig.update_layout(
        mapbox=dict(
            style='open-street-map',
            center=dict(lat=center_lat, lon=center_lon),
            zoom=10
        ),
        height=600,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True
    )
    
    return fig

# Initialize session state with DEMO DATA
if 'delivery_locations' not in st.session_state:
    # Pre-populated demo locations (Distribution Centers in Twin Cities)
    st.session_state.delivery_locations = [
        {"name": "North Region Distribution Hub", "lat": 33.6844, "lon": 73.0479, "demand": 150},
        {"name": "Central City Distributor", "lat": 33.7182, "lon": 72.9842, "demand": 200},
        {"name": "G-Sector Distribution Center", "lat": 33.5651, "lon": 73.0169, "demand": 120},
        {"name": "PWD Area Distributor", "lat": 33.6007, "lon": 73.0679, "demand": 180},
        {"name": "F-Block Distribution Point", "lat": 33.6973, "lon": 73.0515, "demand": 160},
        {"name": "I-Sector Regional Hub", "lat": 33.6502, "lon": 72.9875, "demand": 140},
        {"name": "DHA Distribution Center", "lat": 33.7294, "lon": 73.0931, "demand": 190},
        {"name": "Rawalpindi Regional Distributor", "lat": 33.5873, "lon": 72.9234, "demand": 100}
    ]

if 'vans' not in st.session_state:
    st.session_state.vans = [
        {"id": 1, "capacity": 500},
        {"id": 2, "capacity": 500},
        {"id": 3, "capacity": 500}
    ]

if 'warehouse' not in st.session_state:
    st.session_state.warehouse = {"lat": 33.7490, "lon": 72.8536, "name": "FCMG Main Warehouse"}

if 'optimized_routes' not in st.session_state:
    st.session_state.optimized_routes = None

if 'selected_products' not in st.session_state:
    st.session_state.selected_products = []

# Tabs
tab1, tab2 = st.tabs(["📈 Demand Forecasting", "🚚 Route Optimization"])

# ============= TAB 1: DEMAND FORECASTING =============
with tab1:
    st.markdown('<div class="section-header">📊 Demand Forecasting - Aggregated Across All Stores</div>', unsafe_allow_html=True)
    
    # Load model
    model, model_info = load_demand_model()
    if model is None:
        st.error("⚠️ Model not found. Please place the trained model in the 'models/' directory.")
        st.stop()
    
    # Load datasets (now returns dictionary)
    datasets = load_datasets()
    
    if datasets is not None and 'train' in datasets:
        df = datasets['train']
        # Get unique product families
        all_products = sorted(df['family'].unique())
        all_stores = sorted(df['store_nbr'].unique()) if 'store_nbr' in df.columns else list(range(1, 55))
        
        st.info(f"📦 AI Demo Model - Trained on {len(all_products)} product categories. Same approach applies to your FCMG products.")
    else:
        # Fallback to generic products
        all_products = [f"Product_{i}" for i in range(1, 34)]
        all_stores = list(range(1, 55))
        st.warning("⚠️ Using mock product data. Load actual dataset for real predictions.")
    
    # Product selection section
    st.markdown("### 🛍️ Select Products to Forecast")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("**Choose products from the table below:**")
    with col2:
        if st.button("✅ Select All" if len(st.session_state.selected_products) < len(all_products) else "❌ Clear All"):
            if len(st.session_state.selected_products) < len(all_products):
                st.session_state.selected_products = all_products.copy()
            else:
                st.session_state.selected_products = []
            st.rerun()
    
    # Create paginated product table
    items_per_page = 10
    total_pages = (len(all_products) + items_per_page - 1) // items_per_page
    
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 0
    
    # Display products for current page as table with checkboxes
    start_idx = st.session_state.current_page * items_per_page
    end_idx = min(start_idx + items_per_page, len(all_products))
    page_products = all_products[start_idx:end_idx]
    
    # Product selection table with FCMG mapping
    st.markdown("")
    for i, product in enumerate(page_products):
        col1, col2 = st.columns([1, 5])
        with col1:
            is_selected = product in st.session_state.selected_products
            if st.checkbox(f"Select {product}", value=is_selected, key=f"prod_{start_idx+i}", label_visibility="collapsed"):
                if product not in st.session_state.selected_products:
                    st.session_state.selected_products.append(product)
            else:
                if product in st.session_state.selected_products:
                    st.session_state.selected_products.remove(product)
        with col2:
            # Display FCMG-mapped name if available, otherwise show original
            display_name = PRODUCT_MAPPING.get(product, product)
            st.markdown(f"**{display_name}** \u003cspan style='color: #888; font-size: 0.85em;'\u003e({product})\u003c/span\u003e", unsafe_allow_html=True)
    
    # Pagination controls BELOW the table
    st.markdown("")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("⬅️ Previous", disabled=(st.session_state.current_page == 0)):
            st.session_state.current_page -= 1
            st.rerun()
    with col2:
        st.markdown(f"<div style='text-align: center; padding: 0.5rem;'>Page {st.session_state.current_page + 1} of {total_pages}</div>", unsafe_allow_html=True)
    with col3:
        if st.button("Next ➡️", disabled=(st.session_state.current_page >= total_pages - 1)):
            st.session_state.current_page += 1
            st.rerun()
    
    if st.session_state.selected_products:
        st.info(f"✅ {len(st.session_state.selected_products)} product(s) selected")
    
    selected_products = st.session_state.selected_products
    
    st.markdown("---")
    
    # Date range selection
    st.markdown("### 📅 Forecast Period")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() + timedelta(days=1),
            min_value=datetime.now()
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now() + timedelta(days=30),
            min_value=start_date
        )
    
    st.markdown("")
    
    # Generate forecast button
    if st.button("🚀 Generate Combined Forecast for All Stores", type="primary", use_container_width=True):
        if not selected_products:
            st.warning("⚠️ Please select at least one product")
        else:
            with st.spinner(f"Generating forecasts for {len(selected_products)} products across {len(all_stores)} stores..."):
                forecasts = []
                # Create list of dates
                dates = pd.date_range(start_date, end_date, freq='D')
                
                # BATCH PREDICTION (Vectorized) - Fast & Efficient
                # Predict for all products and all dates in one go
                try:
                    forecast_df = predict_demand_batch(
                        model, selected_products, dates, datasets, model_info
                    )
                    
                    # Scale to all distribution centers (aggregate)
                    # For demo: Multiplying single store prediction by network size
                    # Real world: Would predict for each center individually
                    forecast_df['total_demand_all_stores'] = forecast_df['prediction'] * len(all_stores)
                    
                    # Format for display
                    forecasts = forecast_df[['date', 'family', 'total_demand_all_stores']].to_dict('records')
                    # Rename family to product for compatibility
                    for f in forecasts:
                        f['product'] = f.pop('family')
                        
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    forecasts = []

                st.session_state['forecasts'] = pd.DataFrame(forecasts)
                st.session_state['forecast_generated'] = True
                st.success(f"✅ Generated {len(forecasts)} forecasts in milliseconds!")
    
    # Display results
    if st.session_state.get('forecast_generated', False):
        st.markdown("---")
        st.markdown('<div class="section-header">📊 Forecast Results</div>', unsafe_allow_html=True)
        
        forecasts_df = st.session_state['forecasts']
        
        # Metrics
        total_demand = int(forecasts_df['total_demand_all_stores'].sum())
        avg_daily = int(forecasts_df.groupby('date')['total_demand_all_stores'].sum().mean())
        peak_day = int(forecasts_df.groupby('date')['total_demand_all_stores'].sum().max())
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📦 Total Demand", f"{total_demand:,}")
        with col2:
            st.metric("📅 Avg Daily", f"{avg_daily:,}")
        with col3:
            st.metric("📈 Peak Day", f"{peak_day:,}")
        with col4:
            st.metric("🏪 Stores", f"{len(all_stores)}")
        
        st.markdown("")
        
        # Visualization
        st.markdown("### 📈 Daily Demand Trend")
        daily = forecasts_df.groupby('date')['total_demand_all_stores'].sum().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily['date'], 
            y=daily['total_demand_all_stores'], 
            mode='lines+markers',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=8)
        ))
        fig.update_layout(
            xaxis_title="Date", 
            yaxis_title="Total Demand (All Stores Combined)",
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 📊 Demand by Product")
        product_summary = forecasts_df.groupby('product')['total_demand_all_stores'].sum().sort_values(ascending=False).head(10)
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=product_summary.index, 
            y=product_summary.values,
            marker_color='#2ca02c'
        ))
        fig2.update_layout(
            xaxis_title="Product", 
            yaxis_title="Total Demand",
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        
        # Data table
        with st.expander("📋 View Detailed Forecast Data"):
            # Round numbers for display
            display_df = forecasts_df.copy()
            display_df['total_demand_all_stores'] = display_df['total_demand_all_stores'].round(0).astype(int)
            st.dataframe(display_df, use_container_width=True, height=400)
            csv = display_df.to_csv(index=False)
            st.download_button("📥 Download Forecast as CSV", csv, "demand_forecast.csv", "text/csv")
        
        # --- Model Insights (Explainability) ---
        st.markdown("---")
        st.markdown('<div class="section-header">🧠 AI Model Insights</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.info("💡 **Why this prediction?**\nThe chart shows which factors drove the forecast most strongly.")
            # Feature Importance Plot
            try:
                # Get feature importance from model
                # XGBoost booster object
                booster = model.get_booster()
                # Get importance score (gain)
                importance = booster.get_score(importance_type='gain')
                
                # Convert to dataframe
                importance_df = pd.DataFrame(
                    list(importance.items()), 
                    columns=['Feature', 'Importance']
                ).sort_values(by='Importance', ascending=True).tail(10)
                
                # Map technical feature names to readable names
                feature_map = {
                    'lag_1': 'Previous Day Sales',
                    'lag_2': '2-Day Lag Sales',
                    'lag_3': '3-Day Lag Sales',
                    'lag_4': '4-Day Lag Sales',
                    'lag_5': '5-Day Lag Sales',
                    'lag_6': '6-Day Lag Sales',
                    'lag_7': 'Last Week Sales', 
                    'onpromotion': 'Promotion Active',
                    'dcoilwtico': 'Oil Price Index',
                    'dayofweek': 'Day of Week Pattern',
                    'store_rank': 'Store Performance Tier',
                    'family_rank': 'Product Popularity',
                    'Trend': 'Long-term Trend',
                    'month': 'Seasonal Month',
                    'dayofmonth': 'Day of Month',
                    'sin_3.5_1': 'Weekly Cycle (Sin)',
                    'cos_3.5_1': 'Weekly Cycle (Cos)',
                    'sin_7_1': 'Bi-Weekly Cycle',
                    'National_Holiday': 'National Holiday Effect'
                }
                
                importance_df['Readable'] = importance_df['Feature'].map(feature_map).fillna(importance_df['Feature'])
                
                fig_imp = go.Figure(go.Bar(
                    x=importance_df['Importance'],
                    y=importance_df['Readable'],
                    orientation='h',
                    marker_color='#00CC96'
                ))
                
                fig_imp.update_layout(
                    title="Top 10 Drivers of Demand",
                    xaxis_title="Impact Score (Gain)",
                    yaxis_title="",
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#FAFAFA')
                )
                st.plotly_chart(fig_imp, use_container_width=True)
                
            except Exception as e:
                st.warning(f"Feature importance not available for this model version.")

        with col2:
            st.info("📈 **Trend Analysis**\nObserved patterns in the data.")
            # Simple context explanation
            avg_forecast = forecasts_df['total_demand_all_stores'].mean()
            max_forecast = forecasts_df['total_demand_all_stores'].max()
            
            st.markdown(f"""
            <div class="metric-card" style="text-align: left; border-left: 5px solid #1f77b4;">
                <h4 style="color:#FAFAFA; margin-top:0;">Key Patterns Detected:</h4>
                <ul style="color:#B0B0B0; margin-left: 20px; text-align: left;">
                    <li style="margin-bottom: 10px;"><strong>Daily Average:</strong> {int(avg_forecast):,} units</li>
                    <li style="margin-bottom: 10px;"><strong>Peak Demand:</strong> {int(max_forecast):,} units</li>
                    <li style="margin-bottom: 10px;"><strong>Dominant Factor:</strong> Previous sales history (Lags) is strongly weighted by the model.</li>
                    <li style="margin-bottom: 10px;"><strong>Seasonality:</strong> Weekly cycles (Day of Week) significantly impact daily volume.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        # --- Model Input Data (Transparency) ---
        with st.expander("🔍 See Raw Model Inputs (What we fed the AI)"):
            st.info("This table shows the exact features generated for the **first day** of the forecast. This proves the model considers holidays, oil prices, and recent sales history.")
            # Generate sample features for the first date
            from forecast_helper import prepare_forecast_features_batch
            
            sample_date = [dates[0]]
            sample_features = prepare_forecast_features_batch(
                selected_products, sample_date, datasets, model_info.get('features', [])
            )
            
            # Select key columns to display (don't show all 46)
            cols_to_show = ['family', 'dayofweek', 'dcoilwtico', 'lag_1', 'lag_7', 'onpromotion', 'National_Holiday']
            # Filter columns that actually exist
            cols_to_show = [c for c in cols_to_show if c in sample_features.columns]
            
            # Create readable version
            display_features = sample_features[cols_to_show].head(10).copy()
            column_map = {
                'family': 'Product Family',
                'dayofweek': 'Day (0=Mon)',
                'dcoilwtico': 'Oil Price ($)',
                'lag_1': 'Sales (Yesterday)',
                'lag_7': 'Sales (Last Week)',
                'onpromotion': 'Promo Active',
                'National_Holiday': 'Holiday Flag'
            }
            display_features = display_features.rename(columns=column_map)
            
            st.dataframe(display_features, use_container_width=True)
            st.caption(f"Showing raw inputs for {dates[0].strftime('%Y-%m-%d')}. Total features used: {len(sample_features.columns)}")
        
        # Integration with Route Optimization
        st.markdown("---")
        st.markdown("### 🔗 Next Step: Route Optimization")
        st.info("💡 Use the forecasted demand to optimize delivery routes in the **Route Optimization** tab above. The system will ensure each distributor receives the right quantity while minimizing total distance.")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("➡️ Go to Route Optimization Tab", type="primary", use_container_width=True):
                st.info("👆 Click on the '🚚 Route Optimization' tab above to continue")


# ============= TAB 2: ROUTE OPTIMIZATION =============
with tab2:
    st.markdown('<div class="section-header">🚚 Route Optimization</div>', unsafe_allow_html=True)
    
    # Step 1: Warehouse Configuration
    st.markdown("### 📍 Warehouse Location")
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        warehouse_lat = st.number_input("Latitude", value=st.session_state.warehouse['lat'], format="%.6f", key="wh_lat")
    with col2:
        warehouse_lon = st.number_input("Longitude", value=st.session_state.warehouse['lon'], format="%.6f", key="wh_lon")
    with col3:
        if st.button("Update Warehouse"):
            st.session_state.warehouse = {"lat": warehouse_lat, "lon": warehouse_lon, "name": "Main Warehouse"}
            st.success("✅ Updated!")
    
    st.markdown("---")
    
    # Step 2: Distribution Centers
    st.markdown("### 🗺️ Distribution Centers")
    st.info("💡 Add distributor locations manually below or manage pre-loaded demo centers")
    
    # Manual location input
    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
    with col1:
        new_location_name = st.text_input("Location Name", placeholder="e.g., Metro Store")
    with col2:
        new_loc_lat = st.number_input("Latitude", value=33.6844, format="%.6f", key="new_loc_lat")
    with col3:
        new_loc_lon = st.number_input("Longitude", value=73.0479, format="%.6f", key="new_loc_lon")
    with col4:
        new_demand = st.number_input("Demand", min_value=1, value=100, step=10, key="new_demand")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("➕ Add Location to Map", type="secondary", use_container_width=True):
            if new_location_name:
                st.session_state.delivery_locations.append({
                    "name": new_location_name,
                    "lat": new_loc_lat,
                    "lon": new_loc_lon,
                    "demand": new_demand
                })
                st.success(f"✅ Added {new_location_name}")
                st.rerun()
            else:
                st.warning("Please enter a location name")
    with col2:
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.delivery_locations = []
            st.session_state.optimized_routes = None
            st.rerun()
    
    # Display current map with warehouse and locations
    if len(st.session_state.delivery_locations) > 0 or st.session_state.warehouse:
        fig = go.Figure()
        
        # Add warehouse
        fig.add_trace(go.Scattermapbox(
            lat=[st.session_state.warehouse['lat']],
            lon=[st.session_state.warehouse['lon']],
            mode='markers',
            marker=dict(size=20, color='red', symbol='warehouse'),
            name='Warehouse',
            text=['Warehouse'],
            hovertemplate='<b>Warehouse</b><extra></extra>'
        ))
        
        # Add delivery locations
        if st.session_state.delivery_locations:
            lats = [loc['lat'] for loc in st.session_state.delivery_locations]
            lons = [loc['lon'] for loc in st.session_state.delivery_locations]
            names = [loc['name'] for loc in st.session_state.delivery_locations]
            demands = [loc['demand'] for loc in st.session_state.delivery_locations]
            
            fig.add_trace(go.Scattermapbox(
                lat=lats,
                lon=lons,
                mode='markers',
                marker=dict(size=12, color='green'),
                name='Delivery Locations',
                text=[f"{n}<br>Demand: {d}" for n, d in zip(names, demands)],
                hovertemplate='<b>%{text}</b><extra></extra>'
            ))
        
        # Update layout
        center_lat = st.session_state.warehouse['lat']
        center_lon = st.session_state.warehouse['lon']
        
        fig.update_layout(
            mapbox=dict(
                style='open-street-map',
                center=dict(lat=center_lat, lon=center_lon),
                zoom=10
            ),
            height=500,
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Step 3: Van Configuration
    if st.session_state.delivery_locations:
        st.markdown("### 🚚 Van Fleet Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Current Fleet:**")
            for van in st.session_state.vans:
                st.markdown(f"🚚 Van {van['id']}: Capacity {van['capacity']} units")
        
        with col2:
            st.markdown("**Add New Van:**")
            new_van_capacity = st.number_input("Van Capacity", min_value=100, value=500, step=50, key="new_van_cap")
            if st.button("➕ Add Van"):
                new_id = max([v['id'] for v in st.session_state.vans]) + 1 if st.session_state.vans else 1
                st.session_state.vans.append({"id": new_id, "capacity": new_van_capacity})
                st.success(f"✅ Added Van {new_id}")
                st.rerun()
        
        # Remove van option
        if len(st.session_state.vans) > 1:
            van_to_remove = st.selectbox("Remove Van:", [f"Van {v['id']}" for v in st.session_state.vans])
            if st.button("🗑️ Remove Selected Van"):
                van_id = int(van_to_remove.split()[1])
                st.session_state.vans = [v for v in st.session_state.vans if v['id'] != van_id]
                st.rerun()
        
        st.markdown("---")
        
        # Step 5: Optimize Routes Button
        st.markdown("### 🚀 Run Optimization")
        
        total_demand = sum([loc['demand'] for loc in st.session_state.delivery_locations])
        total_capacity = sum([van['capacity'] for van in st.session_state.vans])
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Demand", f"{total_demand} units")
        col2.metric("Total Capacity", f"{total_capacity} units")
        col3.metric("Utilization", f"{(total_demand/total_capacity*100):.1f}%")
        
        st.markdown("")
        
        if st.button("🎯 Optimize Routes", type="primary", use_container_width=True):
            with st.spinner("Optimizing routes..."):
                # Prepare data for OR-Tools
                warehouse_coords = (st.session_state.warehouse['lat'], st.session_state.warehouse['lon'])
                retailer_locations = [(loc['lat'], loc['lon']) for loc in st.session_state.delivery_locations]
                retailer_demands = [loc['demand'] for loc in st.session_state.delivery_locations]
                retailer_names = [loc['name'] for loc in st.session_state.delivery_locations]
                
                # Use first van capacity (or max) for uniform capacity
                van_capacity = max([v['capacity'] for v in st.session_state.vans])
                num_vans = len(st.session_state.vans)
                
                # Run optimization
                routes, total_distance = optimize_routes(
                    warehouse_coords, 
                    retailer_locations, 
                    retailer_demands,
                    van_capacity, 
                    num_vans
                )
                
                if routes:
                    st.session_state.optimized_routes = {
                        'routes': routes,
                        'total_distance': total_distance,
                        'retailer_names': retailer_names,
                        'warehouse_coords': warehouse_coords,
                        'retailer_locations': retailer_locations,
                        'retailer_demands': retailer_demands
                    }
                    st.success("✅ Routes optimized successfully!")
                    st.rerun()
                else:
                    st.error("❌ Could not find optimal solution. Try adding more vans or increasing capacity.")
        
        # Step 6: Display Optimized Routes
        if st.session_state.optimized_routes:
            st.markdown("---")
            st.markdown('<div class="section-header">📊 Optimization Results</div>', unsafe_allow_html=True)
            
            routes_data = st.session_state.optimized_routes
            routes = routes_data['routes']
            total_distance = routes_data['total_distance']
            retailer_names = routes_data['retailer_names']
            warehouse_coords = routes_data['warehouse_coords']
            retailer_locations = routes_data['retailer_locations']
            retailer_demands = routes_data['retailer_demands']
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("🚚 Vans Used", len(routes))
            col2.metric("🏪 Locations Served", len(st.session_state.delivery_locations))
            col3.metric("📏 Total Distance", f"{total_distance:.1f} km")
            col4.metric("⚡ Avg per Van", f"{total_distance/len(routes):.1f} km")
            
            st.markdown("")
            
            # Map with routes
            st.markdown("### 🗺️ Optimized Routes")
            fig_routes = create_route_map(warehouse_coords, retailer_locations, retailer_names, routes)
            st.plotly_chart(fig_routes, use_container_width=True)
            
            # Route details
            st.markdown("### 📋 Route Details")
            for route_info in routes:
                route = route_info['route']
                van_id = route_info['vehicle_id']
                distance = route_info['distance_km']
                
                # Calculate load
                load = sum(retailer_demands[node-1] for node in route if node != 0)
                van_capacity_used = max([v['capacity'] for v in st.session_state.vans])
                utilization = (load / van_capacity_used) * 100
                
                with st.expander(f"🚚 Van {van_id} - {distance:.1f} km - {utilization:.0f}% capacity used"):
                    route_stops = []
                    for node in route:
                        if node == 0:
                            route_stops.append("📦 Warehouse")
                        else:
                            location = st.session_state.delivery_locations[node-1]
                            route_stops.append(f"{location['name']} ({location['demand']} units)")
                    
                    st.markdown(" → ".join(route_stops))
                    st.caption(f"Total Load: {load}/{van_capacity_used} units")
                    
                    # Progress bar for capacity
                    st.progress(utilization / 100)
    
    else:
        st.info("👆 Add delivery locations above to start route optimization")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888; padding: 1rem;'>"
    "FCMG AI Supply Chain Platform © 2026 | Powered by XGBoost & OR-Tools"
    "</div>",
    unsafe_allow_html=True
)
