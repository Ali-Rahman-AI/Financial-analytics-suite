"""
Financial Analytics Suite - Data Sources Page
Data connectors, connection wizard, and data catalog
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import yfinance as yf
import requests
import sqlalchemy
from sqlalchemy import create_engine, text
import sqlite3


# Import theme utilities
from pages_new.theme_utils import get_theme_colors, inject_premium_styles


# Default data sources - will be initialized in session state
DEFAULT_SOURCES = [
    {
        'id': 'demo-1',
        'name': 'Portfolio_Holdings.csv',
        'type': 'CSV',
        'icon': '📄',
        'status': 'connected',
        'rows': 1247,
        'columns': 12,
        'last_sync': '2 hours ago',
        'freshness': 'fresh',
        'is_demo': True,
        'schema': ['ticker', 'shares', 'avg_cost', 'current_price', 'market_value', 'weight', 'sector']
    },
    {
        'id': 'demo-2',
        'name': 'Market_Data.xlsx',
        'type': 'Excel',
        'icon': '📊',
        'status': 'connected',
        'rows': 8453,
        'columns': 24,
        'last_sync': '1 day ago',
        'freshness': 'stale',
        'is_demo': True,
        'schema': ['date', 'open', 'high', 'low', 'close', 'volume', 'adj_close']
    },
]


def init_data_sources():
    """Initialize data sources in session state - start empty"""
    if 'data_sources' not in st.session_state:
        st.session_state.data_sources = []  # Start empty - no demo sources
    if 'show_demo_sources' not in st.session_state:
        st.session_state.show_demo_sources = False


def add_data_source(source: Dict):
    """Add a new data source"""
    if 'data_sources' not in st.session_state:
        st.session_state.data_sources = []
    st.session_state.data_sources.append(source)


def remove_data_source(source_id: str):
    """Remove/disconnect a data source"""
    if 'data_sources' in st.session_state:
        st.session_state.data_sources = [s for s in st.session_state.data_sources if s.get('id') != source_id]


def clear_all_demo_sources():
    """Remove all demo data sources"""
    if 'data_sources' in st.session_state:
        st.session_state.data_sources = [s for s in st.session_state.data_sources if not s.get('is_demo', False)]



CONNECTOR_TYPES = [
    {'name': 'CSV File', 'icon': '📄', 'desc': 'Upload CSV files directly', 'popular': True},
    {'name': 'Excel File', 'icon': '📊', 'desc': 'Upload Excel workbooks (.xlsx, .xls)', 'popular': True},
    {'name': 'PostgreSQL', 'icon': '🐘', 'desc': 'Connect to PostgreSQL databases', 'popular': True},
    {'name': 'MySQL', 'icon': '🐬', 'desc': 'Connect to MySQL databases', 'popular': False},
    {'name': 'SQLite', 'icon': '💽', 'desc': 'Connect to SQLite database files', 'popular': False},
    {'name': 'BigQuery', 'icon': '☁️', 'desc': 'Google BigQuery data warehouse', 'popular': False},
    {'name': 'Snowflake', 'icon': '❄️', 'desc': 'Snowflake cloud data platform', 'popular': False},
    {'name': 'REST API', 'icon': '🌐', 'desc': 'Connect to any REST API endpoint', 'popular': True},
    {'name': 'Yahoo Finance', 'icon': '📈', 'desc': 'Real-time market data from Yahoo', 'popular': True},
    {'name': 'Alpha Vantage', 'icon': '🔷', 'desc': 'Financial data API (stocks, forex, crypto)', 'popular': False},
    {'name': 'Polygon.io', 'icon': '🌀', 'desc': 'Real-time stock & crypto market data', 'popular': True},
    {'name': 'QuickBooks', 'icon': '📗', 'desc': 'Import accounting & sales data', 'popular': True},
    {'name': 'Xero', 'icon': '🟦', 'desc': 'Connect Xero financial data', 'popular': False},
]


def render_connector_card(connector: Dict, c: Dict, key_prefix: str) -> bool:
    """Render a connector type card"""
    popular_badge = '<span style="background: linear-gradient(135deg, #6366f1, #8b5cf6); color: white; padding: 2px 8px; border-radius: 9999px; font-size: 9px; font-weight: 600; margin-left: 8px;">POPULAR</span>' if connector.get('popular') else ""
    
    card_html = f'''<div style="background: {c['glass_bg']}; backdrop-filter: blur(20px); border: 1px solid {c['glass_border']}; border-radius: 12px; padding: 16px; margin-bottom: 8px;"><div style="display: flex; align-items: center; gap: 12px;"><span style="font-size: 28px;">{connector['icon']}</span><div><div style="font-size: 14px; font-weight: 600; color: {c['text']};">{connector['name']}{popular_badge}</div><div style="font-size: 11px; color: {c['text_muted']}; margin-top: 2px;">{connector['desc']}</div></div></div></div>'''
    
    st.markdown(card_html, unsafe_allow_html=True)
    return st.button("Connect", key=f"{key_prefix}_{connector['name']}", use_container_width=True)



def render_source_row(source: Dict, c: Dict) -> None:
    """Render a data source row in the catalog"""
    status_colors = {'connected': c['success'], 'disconnected': c['error'], 'syncing': c['warning']}
    freshness_colors = {'fresh': c['success'], 'stale': c['warning'], 'live': c['accent'], 'unknown': c['text_muted']}
    
    status_color = status_colors.get(source['status'], c['text_muted'])
    freshness_color = freshness_colors.get(source['freshness'], c['text_muted'])
    rows_display = source['rows'] if source['rows'] else '-'
    
    row_html = f'''<div style="display: grid; grid-template-columns: 2fr 1fr 1fr 1fr 1fr; align-items: center; padding: 16px 20px; background: {c['bg_card']}; border: 1px solid {c['border']}; border-radius: 12px; margin-bottom: 12px;"><div style="display: flex; align-items: center; gap: 12px;"><span style="font-size: 24px;">{source['icon']}</span><div><div style="font-size: 14px; font-weight: 600; color: {c['text']};">{source['name']}</div><div style="font-size: 11px; color: {c['text_muted']};">{source['type']}</div></div></div><div style="text-align: center;"><span style="display: inline-flex; align-items: center; gap: 6px; padding: 4px 10px; border-radius: 9999px; background: {status_color}15; color: {status_color}; font-size: 11px; font-weight: 600;"><span style="width: 6px; height: 6px; border-radius: 50%; background: {status_color};"></span>{source['status'].title()}</span></div><div style="text-align: center; font-size: 13px; color: {c['text']};">{rows_display}</div><div style="text-align: center;"><span style="padding: 4px 10px; border-radius: 8px; background: {freshness_color}15; color: {freshness_color}; font-size: 11px; font-weight: 500;">{source['freshness'].title()}</span></div><div style="text-align: center; font-size: 12px; color: {c['text_muted']};">{source['last_sync']}</div></div>'''
    
    st.markdown(row_html, unsafe_allow_html=True)



def render_connection_wizard(c: Dict) -> None:
    """Render the connection wizard modal/stepper"""
    steps = ['Select Type', 'Configure', 'Test Connection', 'Import']
    current_step = st.session_state.get('connection_wizard_step', 0)
    
    # Step indicator - build as single line HTML to avoid rendering issues
    steps_html = ""
    for i, step in enumerate(steps):
        if i < current_step:
            bg = c['success']
            check_mark = "✓"
            text_color = c['text_secondary']
        elif i == current_step:
            bg = c['gradient']
            check_mark = str(i + 1)
            text_color = c['text_secondary']
        else:
            bg = c['bg_elevated']
            check_mark = str(i + 1)
            text_color = c['text_muted']
        
        # Build connector line
        connector_bg = c['success'] if i < current_step else c['border']
        connector = f'<div style="flex: 1; height: 2px; background: {connector_bg}; margin: 0 8px;"></div>' if i < len(steps) - 1 else ''
        
        # Build step indicator - all on one line
        steps_html += f'<div style="display: flex; align-items: center; flex: 1;"><div style="display: flex; flex-direction: column; align-items: center;"><div style="width: 32px; height: 32px; border-radius: 50%; background: {bg}; display: flex; align-items: center; justify-content: center; color: white; font-size: 12px; font-weight: 600;">{check_mark}</div><span style="font-size: 10px; color: {text_color}; margin-top: 6px; white-space: nowrap;">{step}</span></div>{connector}</div>'
    
    # Render the stepper
    st.markdown(f'<div style="display: flex; align-items: flex-start; padding: 20px; background: {c["glass_bg"]}; border: 1px solid {c["glass_border"]}; border-radius: 12px; margin-bottom: 24px;">{steps_html}</div>', unsafe_allow_html=True)



def render():
    """Render the Data Sources page"""
    c = get_theme_colors()
    
    # Inject Premium styles
    inject_premium_styles()
    
    # Initialize session state
    if 'show_new_connection' not in st.session_state:
        st.session_state.show_new_connection = False
    if 'connection_wizard_step' not in st.session_state:
        st.session_state.connection_wizard_step = 0
    
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🔌 Data Sources")
        st.markdown(f"""
        <p style="color: {c['text_secondary']}; font-size: 14px; margin-top: -10px;">
            Connect, manage, and monitor your data sources and connectors
        </p>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        if st.button("➕ New Connection", key="new_connection_btn", type="primary", use_container_width=True):
            st.session_state.show_new_connection = True
    
    st.markdown("---")
    
    # Show connection wizard or main view
    if st.session_state.show_new_connection:
        # Connection Wizard
        st.subheader("New Data Connection")
        render_connection_wizard(c)
        
        if st.session_state.connection_wizard_step == 0:
            # Step 1: Select connector type
            st.markdown("#### Select Data Source Type")
            
            cols = st.columns(3)
            for i, connector in enumerate(CONNECTOR_TYPES):
                with cols[i % 3]:
                    if render_connector_card(connector, c, "connector"):
                        st.session_state.selected_connector = connector
                        st.session_state.connection_wizard_step = 1
                        st.rerun()
        
        elif st.session_state.connection_wizard_step == 1:
            # Step 2: Configure
            connector = st.session_state.get('selected_connector', CONNECTOR_TYPES[0])
            st.markdown(f"#### Configure {connector['name']} Connection")
            
            if connector['name'] in ['CSV File', 'Excel File']:
                uploaded_file = st.file_uploader(
                    "Upload your file",
                    type=['csv', 'xlsx', 'xls'] if 'Excel' in connector['name'] else ['csv'],
                    help="Drag and drop or click to browse",
                    key="wizard_file_upload"
                )
                
                if uploaded_file:
                    # Read and store the file immediately
                    try:
                        if uploaded_file.name.endswith('.csv'):
                            df = pd.read_csv(uploaded_file)
                        else:
                            df = pd.read_excel(uploaded_file)
                        
                        # Store in session state for wizard
                        st.session_state.wizard_uploaded_df = df
                        st.session_state.wizard_uploaded_filename = uploaded_file.name
                        
                        st.success(f"✅ File loaded: {uploaded_file.name} ({len(df):,} rows, {len(df.columns)} columns)")
                        st.dataframe(df.head(), use_container_width=True)
                        
                        if st.button("Continue →", type="primary", key="continue_step1"):
                            st.session_state.connection_wizard_step = 2
                            st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error reading file: {str(e)}")
            
            elif connector['name'] in ['PostgreSQL', 'MySQL', 'BigQuery', 'Snowflake', 'SQLite']:
                # Database connection
                st.markdown(f"##### Configure {connector['name']} Connection")
                
                if connector['name'] == 'SQLite':
                    st.file_uploader("SQLite Database File", type=['db', 'sqlite', 'sqlite3'], key="sqlite_file")
                else:
                    st.text_input("Host", placeholder="localhost or db.example.com", key="db_host")
                    col1, col2 = st.columns(2)
                    with col1:
                        default_port = {'PostgreSQL': 5432, 'MySQL': 3306, 'BigQuery': 443, 'Snowflake': 443}.get(connector['name'], 5432)
                        st.number_input("Port", value=default_port, key="db_port")
                    with col2:
                        st.text_input("Database Name", placeholder="my_database", key="db_name")
                    st.text_input("Username", placeholder="db_user", key="db_user")
                    st.text_input("Password", type="password", placeholder="••••••••", key="db_pass")
                
                # Use session state to persist connection status within the wizard
                if 'db_connected' not in st.session_state:
                    st.session_state.db_connected = False
                
                # Connect/Disconnect Button logic
                if not st.session_state.db_connected:
                    if st.button("🔌 Test & Connect", type="primary", key="test_db", use_container_width=True):
                        with st.spinner(f"Connecting to {connector['name']}..."):
                            try:
                                # Construct connection string
                                if connector['name'] == 'SQLite':
                                    if 'sqlite_file' in st.session_state and st.session_state.sqlite_file:
                                        import os
                                        file_details = st.session_state.sqlite_file
                                        temp_path = f"temp_{file_details.name}"
                                        with open(temp_path, "wb") as f:
                                            f.write(file_details.getbuffer())
                                        db_url = f"sqlite:///{temp_path}"
                                    else:
                                        st.error("Please upload a SQLite file")
                                        st.stop()
                                else:
                                    db_type_map = {
                                        'PostgreSQL': 'postgresql',
                                        'MySQL': 'mysql+pymysql',
                                        'Snowflake': 'snowflake', 
                                        'BigQuery': 'bigquery'
                                    }
                                    driver = db_type_map.get(connector['name'])
                                    host = st.session_state.get('db_host')
                                    port = st.session_state.get('db_port')
                                    db_name = st.session_state.get('db_name')
                                    user = st.session_state.get('db_user')
                                    password = st.session_state.get('db_pass')
                                    
                                    if not all([host, db_name, user]):
                                        st.error("Please fill in all required fields")
                                        st.stop()
                                        
                                    db_url = f"{driver}://{user}:{password}@{host}:{port}/{db_name}"

                                # Test connection and fetch tables
                                engine = create_engine(db_url)
                                inspector = sqlalchemy.inspect(engine)
                                tables = inspector.get_table_names()
                                
                                st.session_state.db_tables = tables
                                st.session_state.db_url_cache = db_url # Cache URL for next step (security note: usually better to reconstruct)
                                st.session_state.db_connected = True
                                st.success(f"✅ Connected! Found {len(tables)} tables.")
                                st.rerun()

                            except Exception as e:
                                st.error(f"❌ Connection failed: {str(e)}")
                
                else:
                    st.success(f"✅ Connected to {connector['name']}")
                    if st.button("Disconnect/Change", type="secondary", key="disconnect_db"):
                        st.session_state.db_connected = False
                        if 'db_tables' in st.session_state: del st.session_state.db_tables
                        st.rerun()
                        
                    st.markdown("---")
                    st.markdown("##### Select Data to Import")
                    
                    import_mode = st.radio("Import Mode", ["Select Table", "Custom Query"], horizontal=True)
                    
                    if import_mode == "Select Table":
                        tables = st.session_state.get('db_tables', [])
                        if tables:
                            selected_table = st.selectbox("Select Table", tables)
                            limit = st.number_input("Row Limit", min_value=1, value=1000, step=100)
                            query = f"SELECT * FROM {selected_table} LIMIT {limit}"
                        else:
                            st.warning("No tables found in this database.")
                            query = None
                    else:
                        query = st.text_area("SQL Query", value="SELECT * FROM table_name LIMIT 100", height=100)
                    
                    if query:
                        st.code(query, language="sql")
                        
                        if st.button("🚀 Run Query & Import", type="primary", key="run_db_query", use_container_width=True):
                            try:
                                engine = create_engine(st.session_state.db_url_cache)
                                with engine.connect() as conn:
                                    df = pd.read_sql(text(query), conn)
                                    
                                st.success(f"✅ Fetched {len(df)} rows")
                                st.session_state.wizard_uploaded_df = df
                                st.session_state.wizard_uploaded_filename = f"{connector['name'].lower()}_export_{datetime.now().strftime('%Y%m%d')}.csv"
                                st.session_state.connection_wizard_step = 2
                                # Clear db state for next time
                                st.session_state.db_connected = False
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Query failed: {str(e)}")
            
            elif connector['name'] == 'Yahoo Finance':
                # Yahoo Finance - ticker input
                st.markdown("##### Yahoo Finance API")
                st.info("📈 Enter stock tickers to fetch real-time market data")
                
                tickers = st.text_input("Stock Tickers", value="AAPL, MSFT, GOOGL", 
                                        help="Comma-separated list of stock symbols", key="yf_tickers")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.date_input("Start Date", key="yf_start")
                with col2:
                    st.date_input("End Date", key="yf_end")
                
                if st.button("📥 Fetch Data", type="primary", key="fetch_yf", use_container_width=True):
                    with st.spinner("Fetching data from Yahoo Finance..."):
                        try:
                            # Process tickers
                            ticker_list = [t.strip().upper() for t in tickers.split(',')]
                            start_date = st.session_state.get('yf_start')
                            end_date = st.session_state.get('yf_end')
                            
                            # Fetch data using yfinance
                            if len(ticker_list) == 1:
                                data = yf.download(ticker_list[0], start=start_date, end=end_date)
                                data['Ticker'] = ticker_list[0]
                                df = data.reset_index()
                            else:
                                data = yf.download(ticker_list, start=start_date, end=end_date)
                                # Flatten multi-index columns if necessary
                                if isinstance(data.columns, pd.MultiIndex):
                                    # Stack to get Ticker as a column
                                    df = data.stack(level=1).reset_index().rename(columns={'level_1': 'Ticker'})
                                else:
                                    df = data.reset_index()
                            
                            if df.empty:
                                st.error("No data found for the given parameters.")
                            else:
                                # Clean column names
                                df.columns = [str(c).lower().replace(' ', '_') for c in df.columns]
                                
                                st.success(f"✅ Data fetched successfully! ({len(df)} rows)")
                                st.session_state.wizard_uploaded_df = df
                                st.session_state.wizard_uploaded_filename = f"yahoo_finance_{datetime.now().strftime('%Y%m%d')}.csv"
                                st.session_state.connection_wizard_step = 2
                                st.rerun()
                                
                        except Exception as e:
                            st.error(f"❌ Error fetching data: {str(e)}")
            
            elif connector['name'] == 'Alpha Vantage':
                # Alpha Vantage API
                st.markdown("##### Alpha Vantage API")
                st.info("🔷 Get financial data for stocks, forex, and crypto")
                
                st.text_input("API Key", type="password", placeholder="Your Alpha Vantage API key", key="av_key")
                symbol = st.text_input("Symbol", value="AAPL", help="Stock symbol", key="av_symbol")
                data_type = st.selectbox("Data Type", ["Daily", "Weekly", "Monthly", "Intraday"], key="av_type")
                
                if st.button("📥 Fetch Data", type="primary", key="fetch_av", use_container_width=True):
                    with st.spinner("Fetching data from Alpha Vantage..."):
                        try:
                            api_key = st.session_state.get('av_key')
                            symbol = st.session_state.get('av_symbol')
                            
                            if not api_key:
                                st.error("Please enter an API Key")
                                st.stop()
                                
                            # Mappings for user selection to API function
                            function_map = {
                                "Daily": "TIME_SERIES_DAILY",
                                "Weekly": "TIME_SERIES_WEEKLY",
                                "Monthly": "TIME_SERIES_MONTHLY",
                                "Intraday": "TIME_SERIES_INTRADAY"
                            }
                            ftype = function_map.get(data_type, "TIME_SERIES_DAILY")
                            
                            url = f"https://www.alphavantage.co/query?function={ftype}&symbol={symbol}&apikey={api_key}&datatype=csv"
                            if ftype == "TIME_SERIES_INTRADAY":
                                url += "&interval=5min"
                                
                            df = pd.read_csv(url)
                            
                            # Check for API error response (Alpha Vantage returns text in CSV sometimes for errors)
                            if 'Error Message' in df.columns or len(df.columns) < 2:
                                # Try reading as text to see error
                                r = requests.get(url)
                                error_msg = r.json().get('Error Message') or r.json().get('Note') or "Unknown API error"
                                st.error(f"❌ API Error: {error_msg}")
                            else:
                                # Add ticker column
                                df['ticker'] = symbol
                                
                                st.success(f"✅ Data fetched successfully! ({len(df)} rows)")
                                st.session_state.wizard_uploaded_df = df
                                st.session_state.wizard_uploaded_filename = f"alpha_vantage_{symbol}_{datetime.now().strftime('%Y%m%d')}.csv"
                                st.session_state.connection_wizard_step = 2
                                st.rerun()

                        except Exception as e:
                            st.error(f"❌ Connection failed: {str(e)}")

            elif connector['name'] == 'Polygon.io':
                # Polygon.io API
                st.markdown("##### Polygon.io API")
                st.info("🌀 High-performance market data for Stocks, Forex, and Crypto")
                
                st.text_input("API Key", type="password", placeholder="Your Polygon.io API key", key="poly_key")
                symbol = st.text_input("Ticker", value="AAPL", help="Stock ticker symbol", key="poly_symbol")
                
                if st.button("📥 Fetch Data", type="primary", key="fetch_poly", use_container_width=True):
                    with st.spinner("Fetching data from Polygon.io..."):
                        try:
                            api_key = st.session_state.get('poly_key')
                            symbol = st.session_state.get('poly_symbol')
                            
                            if not api_key:
                                st.error("Please enter an API Key")
                                st.stop()
                                
                            # Basic Aggregates (Bars) endpoint
                            today = datetime.now().strftime('%Y-%m-%d')
                            last_year = (datetime.now() - pd.Timedelta(days=365)).strftime('%Y-%m-%d')
                            
                            url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{last_year}/{today}?adjusted=true&sort=asc&limit=5000&apiKey={api_key}"
                            
                            r = requests.get(url)
                            data = r.json()
                            
                            if data.get('status') == 'OK' and data.get('resultsCount', 0) > 0:
                                results = data['results']
                                df = pd.DataFrame(results)
                                
                                # Rename columns to standard schema
                                rename_map = {
                                    'v': 'volume', 'vw': 'vwap', 'o': 'open', 
                                    'c': 'close', 'h': 'high', 'l': 'low', 
                                    't': 'timestamp', 'n': 'transactions'
                                }
                                df = df.rename(columns=rename_map)
                                
                                # Convert timestamp
                                df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
                                df['ticker'] = symbol
                                
                                st.success(f"✅ Data fetched successfully! ({len(df)} rows)")
                                st.session_state.wizard_uploaded_df = df
                                st.session_state.wizard_uploaded_filename = f"polygon_{symbol}_{datetime.now().strftime('%Y%m%d')}.csv"
                                st.session_state.connection_wizard_step = 2
                                st.rerun()
                            else:
                                error_msg = data.get('error', 'Unknown error')
                                if data.get('resultsCount') == 0:
                                    error_msg = "No data found for this range/ticker."
                                st.error(f"❌ API Error: {error_msg}")
                                
                        except Exception as e:
                            st.error(f"❌ Connection failed: {str(e)}")

            elif connector['name'] in ['QuickBooks', 'Xero']:
                # Accounting / ERP Integration (Mock/Simulated for Commercial Demo)
                service = connector['name']
                icon = connector['icon']
                
                st.markdown(f"##### {icon} Connect to {service}")
                st.info(f"Securely authorize Financial Analytics Suite to access your {service} data.")
                
                st.markdown(f"""
                <div style="background: {c['bg_elevated']}; padding: 20px; border-radius: 12px; border: 1px dashed {c['border']}; text-align: center; margin-bottom: 20px;">
                    <div style="font-size: 48px; margin-bottom: 15px;">🔒</div>
                    <div style="font-weight: 600; margin-bottom: 5px;">OAuth2 Authentication</div>
                    <div style="font-size: 13px; color: {c['text_muted']}; margin-bottom: 15px;">
                        We use industry-standard OAuth2 to securely connect to your accounting software. <br>
                        We never see or store your login credentials.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                mode = st.radio("Connection Mode", ["Simulated (Demo)", "Live API (Requires Token)"], index=0, horizontal=True)
                
                if mode == "Simulated (Demo)":
                    col1, col2 = st.columns(2)
                    with col1:
                        st.text_input("Client ID", placeholder="Ex: AB12345...", key="oauth_client_id")
                    with col2:
                        st.text_input("Client Secret", type="password", placeholder="••••••••••••", key="oauth_client_secret")
                else:
                    st.info(f"ℹ️ For Live connection, paste your **Access Token** from the {service} Developer Portal.")
                    st.text_input("Access Token (Bearer)", type="password", key="oauth_token")
                    label_id = "Tenant ID" if service == 'Xero' else "Company ID (Realm ID)"
                    st.text_input(label_id, key="oauth_tenant_id")

                if st.button(f"🔗 Connect to {service}", type="primary", use_container_width=True):
                    with st.spinner(f"Connecting to {service}..."):
                        
                        if mode == "Simulated (Demo)":
                            # Use existing simulation logic
                            import time
                            time.sleep(1.5) # Simulate network delay
                            
                            # Simulate fetch of P&L or Ledger
                            st.success(f"✅ Successfully authorized with {service} (Simulated)!")
                            
                            # Generate dummy accounting data
                            dates = pd.date_range(end=datetime.now(), periods=12, freq='M')
                            data = []
                            for d in dates:
                                revenue = np.random.uniform(50000, 150000)
                                cogs = revenue * np.random.uniform(0.3, 0.5)
                                expenses = revenue * np.random.uniform(0.2, 0.4)
                                net_income = revenue - cogs - expenses
                                data.append({
                                    'Date': d,
                                    'Revenue': round(revenue, 2),
                                    'COGS': round(cogs, 2),
                                    'Expenses': round(expenses, 2),
                                    'Net_Income': round(net_income, 2),
                                    'Source': service
                                })
                            
                            df = pd.DataFrame(data)
                            
                            st.session_state.wizard_uploaded_df = df
                            st.session_state.wizard_uploaded_filename = f"{service.lower()}_pl_export_{datetime.now().strftime('%Y%m%d')}.csv"
                            st.session_state.connection_wizard_step = 2
                            st.rerun()

                        else:
                            # Real API Logic
                            token = st.session_state.get('oauth_token')
                            tenant_id = st.session_state.get('oauth_tenant_id')
                            
                            if not token or not tenant_id:
                                st.error("Token and ID are required for live connection")
                                st.stop()

                            try:
                                headers = {
                                    'Authorization': f'Bearer {token}',
                                    'Accept': 'application/json'
                                }
                                
                                # Xero Logic
                                if service == 'Xero':
                                    headers['Xero-tenant-id'] = tenant_id
                                    # Fetch Invoices as a test
                                    url = "https://api.xero.com/api.xro/2.0/Invoices"
                                    r = requests.get(url, headers=headers)
                                    
                                    if r.status_code == 200:
                                        data = r.json()
                                        invoices = data.get('Invoices', [])
                                        if invoices:
                                            df = pd.DataFrame(invoices)
                                            # Flatten nested fields if needed (simple flatten)
                                            df = df[['InvoiceID', 'InvoiceNumber', 'Type', 'Status', 'Date', 'DueDate', 'Total', 'AmountDue', 'AmountPaid']]
                                            # Fix types
                                            df['Total'] = pd.to_numeric(df['Total'], errors='coerce')
                                            st.success(f"✅ Successfully fetched {len(df)} invoices from Xero!")
                                        else:
                                            st.warning("Connected, but no invoices found.")
                                            df = pd.DataFrame(columns=['Date', 'Total', 'Type'])
                                    else:
                                        st.error(f"❌ Xero API Error: {r.status_code} - {r.text}")
                                        st.stop()

                                # QuickBooks Logic
                                elif service == 'QuickBooks':
                                    # Use Sandbox URL for specific test, or Prod
                                    # Assuming Production URL structure, user must verify environment
                                    base_url = f"https://quickbooks.api.intuit.com/v3/company/{tenant_id}"
                                    query = "select * from Invoice MAXRESULTS 100"
                                    url = f"{base_url}/query?query={query}&minorversion=65"
                                    
                                    r = requests.get(url, headers=headers)
                                    
                                    if(r.status_code == 401): 
                                         # Try sandbox if 401 (common mistake)
                                         base_url = f"https://sandbox-quickbooks.api.intuit.com/v3/company/{tenant_id}"
                                         url = f"{base_url}/query?query={query}&minorversion=65"
                                         r = requests.get(url, headers=headers)

                                    if r.status_code == 200:
                                        data = r.json()
                                        query_response = data.get('QueryResponse', {})
                                        invoices = query_response.get('Invoice', [])
                                        if invoices:
                                            df = pd.DataFrame(invoices)
                                            # Keep key columns
                                            cols_to_keep = [c for c in ['Id', 'DocNumber', 'TxnDate', 'TotalAmt', 'Balance'] if c in df.columns]
                                            df = df[cols_to_keep]
                                            if 'TotalAmt' in df.columns:
                                                df['TotalAmt'] = pd.to_numeric(df['TotalAmt'], errors='coerce')
                                            st.success(f"✅ Successfully fetched {len(df)} invoices from QuickBooks!")
                                        else:
                                            st.warning("Connected, but no invoices found.")
                                            df = pd.DataFrame(columns=['TxnDate', 'TotalAmt'])
                                    else:
                                        st.error(f"❌ QuickBooks API Error: {r.status_code} - {r.text}")
                                        st.stop()
                                
                                # Finalize
                                st.session_state.wizard_uploaded_df = df
                                st.session_state.wizard_uploaded_filename = f"{service.lower()}_live_export_{datetime.now().strftime('%Y%m%d')}.csv"
                                st.session_state.connection_wizard_step = 2
                                st.rerun()

                            except Exception as e:
                                st.error(f"❌ Connection Error: {str(e)}")

            
            else:
                # REST API / Generic
                st.markdown("##### REST API Connection")
                st.text_input("API Endpoint URL", placeholder="https://api.example.com/v1/data", key="api_url")
                st.text_input("API Key (if required)", type="password", placeholder="Bearer token or API key", key="api_key")
                st.selectbox("HTTP Method", ["GET", "POST"], key="api_method")
                
                if st.button("🔌 Test & Connect", type="primary", key="test_api", use_container_width=True):
                    with st.spinner("Testing API connection..."):
                        try:
                            url = st.session_state.get('api_url')
                            key = st.session_state.get('api_key')
                            method = st.session_state.get('api_method')
                            
                            headers = {}
                            if key:
                                headers['Authorization'] = key
                                
                            if method == "GET":
                                response = requests.get(url, headers=headers)
                            else:
                                response = requests.post(url, headers=headers)
                                
                            if response.status_code == 200:
                                st.success("✅ API connection successful!")
                                
                                # Try to parse json to frame
                                try:
                                    data = response.json()
                                    # Handle different JSON structures
                                    if isinstance(data, list):
                                        df = pd.DataFrame(data)
                                    elif isinstance(data, dict):
                                        # Heuristic: try to find the list inside the dict
                                        found_list = False
                                        for k, v in data.items():
                                            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                                                df = pd.DataFrame(v)
                                                found_list = True
                                                break
                                        if not found_list:
                                            # Treat single dict as one row
                                            df = pd.DataFrame([data])
                                    else:
                                        st.error("Could not parse API response into a table")
                                        st.stop()
                                        
                                    st.session_state.wizard_uploaded_df = df
                                    st.session_state.wizard_uploaded_filename = f"api_import_{datetime.now().strftime('%Y%m%d')}.csv"
                                    st.session_state.connection_wizard_step = 2
                                    st.rerun()
                                    
                                except Exception as parse_e:
                                    st.error(f"❌ Error parsing JSON: {str(parse_e)}")
                            else:
                                st.error(f"❌ API returned status {response.status_code}")
                                
                        except Exception as e:
                            st.error(f"❌ Connection failed: {str(e)}")
            
            if st.button("← Back", key="back_step1"):
                st.session_state.connection_wizard_step = 0
                st.rerun()
        
        elif st.session_state.connection_wizard_step == 2:
            # Step 3: Test Connection & Preview
            connector = st.session_state.get('selected_connector', CONNECTOR_TYPES[0])
            st.markdown("#### Connection Successful!")
            
            st.success("✅ Data loaded and ready to import")
            
            # Show preview
            st.markdown("##### Data Preview")
            if 'wizard_uploaded_df' in st.session_state and st.session_state.wizard_uploaded_df is not None:
                preview_df = st.session_state.wizard_uploaded_df
                st.dataframe(preview_df.head(10), use_container_width=True)
                st.caption(f"Showing 10 of {len(preview_df):,} rows")
            else:
                st.warning("No data to preview")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("← Back", key="back_step2"):
                    st.session_state.connection_wizard_step = 1
                    st.rerun()
            with col2:
                if st.button("🚀 Import Data", type="primary", key="import_data"):
                    # ACTUALLY STORE THE DATA
                    if 'wizard_uploaded_df' in st.session_state and st.session_state.wizard_uploaded_df is not None:
                        df = st.session_state.wizard_uploaded_df
                        filename = st.session_state.get('wizard_uploaded_filename', 'imported_data.csv')
                        
                        # Store in main session state for other pages
                        st.session_state.uploaded_data = df
                        st.session_state.uploaded_filename = filename
                        
                        # Also add to data sources list
                        source_id = f"import-{filename.replace(' ', '_')}"
                        new_source = {
                            'id': source_id,
                            'name': filename,
                            'type': connector['name'],
                            'icon': connector.get('icon', '📄'),
                            'status': 'connected',
                            'rows': len(df),
                            'columns': len(df.columns),
                            'last_sync': 'Just now',
                            'freshness': 'fresh',
                            'is_demo': False,
                            'schema': df.columns.tolist()[:10]
                        }
                        add_data_source(new_source)
                        st.session_state[f"data_{source_id}"] = df
                        
                    st.session_state.connection_wizard_step = 3
                    st.rerun()
        
        elif st.session_state.connection_wizard_step == 3:
            # Step 4: Complete
            st.balloons()
            st.success("🎉 Data source connected successfully!")
            
            filename = st.session_state.get('wizard_uploaded_filename', 'imported_data.csv')
            df = st.session_state.get('uploaded_data')
            
            st.markdown(f"""
            #### ✅ Import Complete!
            
            - **File:** {filename}
            - **Rows:** {len(df) if df is not None else 0:,}
            - **Columns:** {len(df.columns) if df is not None else 0}
            
            Your data is now available across all pages:
            - 📈 **Overview** - View portfolio metrics
            - 📊 **Dashboard Builder** - Auto-generate dashboards
            - 🔮 **Time Series** - Forecast your data
            - 🧪 **ML Lab** - Train models
            - 💼 **Portfolio** - Analyze holdings
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📊 Go to Overview", type="primary", use_container_width=True, key="go_overview"):
                    st.session_state.current_page = 'overview'
                    st.session_state.show_new_connection = False
                    st.session_state.connection_wizard_step = 0
                    st.rerun()
            with col2:
                if st.button("Done", use_container_width=True, key="wizard_done"):
                    st.session_state.show_new_connection = False
                    st.session_state.connection_wizard_step = 0
                    # Clean up wizard state
                    if 'wizard_uploaded_df' in st.session_state:
                        del st.session_state.wizard_uploaded_df
                    if 'wizard_uploaded_filename' in st.session_state:
                        del st.session_state.wizard_uploaded_filename
                    st.rerun()
        
        # Cancel button
        st.markdown("---")
        if st.button("Cancel", key="cancel_wizard"):
            st.session_state.show_new_connection = False
            st.session_state.connection_wizard_step = 0
            st.rerun()
    
    else:
        # Main View - Data Catalog
        init_data_sources()
        
        sources = st.session_state.data_sources
        connected_count = len([s for s in sources if s.get('status') == 'connected'])
        total_rows = sum(s.get('rows', 0) for s in sources if isinstance(s.get('rows'), int))
        
        # Stats overview
        stat_cols = st.columns(4)
        with stat_cols[0]:
            st.metric("Total Sources", len(sources))
        with stat_cols[1]:
            st.metric("Connected", connected_count)
        with stat_cols[2]:
            st.metric("Total Records", f"{total_rows:,}" if total_rows else "0")
        with stat_cols[3]:
            demo_count = len([s for s in sources if s.get('is_demo')])
            st.metric("Demo Sources", demo_count)
        
        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        
        # Action buttons - always show Reset to Default
        btn_cols = st.columns([1, 1, 2])
        with btn_cols[0]:
            if any(s.get('is_demo') for s in sources):
                if st.button("🗑️ Clear Demo Data", key="clear_demo_btn", type="secondary", use_container_width=True):
                    clear_all_demo_sources()
                    st.success("✅ Demo data sources removed!")
                    st.rerun()
        with btn_cols[1]:
            if st.button("🔄 Reset to Default", key="reset_default_btn", use_container_width=True):
                # Generate sample financial data
                np.random.seed(42)
                dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
                n_days = len(dates)
                
                # Create realistic stock price data
                tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META']
                initial_prices = {'AAPL': 150, 'MSFT': 280, 'GOOGL': 120, 'AMZN': 100, 'NVDA': 200, 'META': 180}
                
                sample_data = []
                for ticker in tickers:
                    returns = np.random.normal(0.0005, 0.02, n_days)  # Daily returns
                    prices = initial_prices[ticker] * np.cumprod(1 + returns)
                    volumes = np.random.randint(1000000, 50000000, n_days)
                    
                    for i, date in enumerate(dates):
                        daily_range = prices[i] * 0.03
                        sample_data.append({
                            'date': date,
                            'ticker': ticker,
                            'open': prices[i] - np.random.uniform(0, daily_range/2),
                            'high': prices[i] + np.random.uniform(0, daily_range),
                            'low': prices[i] - np.random.uniform(0, daily_range),
                            'close': prices[i],
                            'volume': volumes[i],
                            'adj_close': prices[i]
                        })
                
                sample_df = pd.DataFrame(sample_data)
                sample_df = sample_df.round(2)
                
                # Store the actual data
                st.session_state.uploaded_data = sample_df
                st.session_state.uploaded_filename = "sample_market_data.csv"
                st.session_state.data_sources = DEFAULT_SOURCES.copy()
                
                st.success(f"✅ Reset with sample data! ({len(sample_df):,} rows)")
                st.rerun()
        
        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        
        # Quick file upload - SHOW PROMINENTLY AT TOP
        with st.expander("📥 Quick Upload - Add Your Data", expanded=len(sources) == 0):
            st.markdown("**Upload CSV or Excel files to analyze your own data**")
            uploaded = st.file_uploader(
                "Drop files here to quickly add data sources",
                type=['csv', 'xlsx', 'xls'],
                accept_multiple_files=True,
                key="quick_upload_main"
            )
            if uploaded:
                for f in uploaded:
                    try:
                        # Read the file
                        if f.name.endswith('.csv'):
                            df = pd.read_csv(f)
                        else:
                            df = pd.read_excel(f)
                        
                        # Check if source already exists
                        existing_ids = [s.get('id', '') for s in st.session_state.data_sources]
                        source_id = f"upload-{f.name.replace(' ', '_')}"
                        
                        if source_id not in existing_ids:
                            # Create source entry
                            new_source = {
                                'id': source_id,
                                'name': f.name,
                                'type': 'CSV' if f.name.endswith('.csv') else 'Excel',
                                'icon': '📄' if f.name.endswith('.csv') else '📊',
                                'status': 'connected',
                                'rows': len(df),
                                'columns': len(df.columns),
                                'last_sync': 'Just now',
                                'freshness': 'fresh',
                                'is_demo': False,
                                'schema': df.columns.tolist()[:10]
                            }
                            add_data_source(new_source)
                            
                            # Store the data in session state
                            st.session_state[f"data_{source_id}"] = df
                            st.session_state.uploaded_data = df
                            st.session_state.uploaded_filename = f.name
                            
                            st.success(f"✅ Added: {f.name} ({len(df):,} rows)")
                        else:
                            # Update existing source data
                            st.session_state[f"data_{source_id}"] = df
                            st.session_state.uploaded_data = df
                            st.session_state.uploaded_filename = f.name
                            st.info(f"📊 Updated: {f.name} ({len(df):,} rows)")
                    except Exception as e:
                        st.error(f"❌ Error loading {f.name}: {str(e)}")
                
                st.rerun()
        
        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        
        # Reload sources after potential upload
        sources = st.session_state.data_sources
        
        # Source table
        if sources:
            # Table header
            header_html = f'<div style="display: grid; grid-template-columns: 2fr 1fr 1fr 1fr 120px; padding: 12px 20px; background: {c["bg_surface"]}; border-radius: 12px 12px 0 0; border: 1px solid {c["border"]}; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: {c["text_secondary"]};"><div>Source</div><div style="text-align: center;">Status</div><div style="text-align: center;">Rows</div><div style="text-align: center;">Freshness</div><div style="text-align: center;">Actions</div></div>'
            st.markdown(header_html, unsafe_allow_html=True)
            
            for i, source in enumerate(sources):
                status_colors = {'connected': c['success'], 'disconnected': c['error'], 'syncing': c['warning']}
                freshness_colors = {'fresh': c['success'], 'stale': c['warning'], 'live': c['accent'], 'unknown': c['text_muted']}
                
                status_color = status_colors.get(source.get('status', 'unknown'), c['text_muted'])
                freshness_color = freshness_colors.get(source.get('freshness', 'unknown'), c['text_muted'])
                rows_display = source.get('rows', '-') if source.get('rows') else '-'
                demo_badge = '<span style="background: #f59e0b20; color: #f59e0b; padding: 2px 6px; border-radius: 4px; font-size: 9px; margin-left: 8px;">DEMO</span>' if source.get('is_demo') else ''
                
                # Create columns for row content and action button
                row_cols = st.columns([6, 1])
                
                with row_cols[0]:
                    row_html = f'<div style="display: grid; grid-template-columns: 2fr 1fr 1fr 1fr; align-items: center; padding: 16px 20px; background: {c["bg_card"]}; border: 1px solid {c["border"]}; border-radius: 0; border-top: none;"><div style="display: flex; align-items: center; gap: 12px;"><span style="font-size: 24px;">{source.get("icon", "📄")}</span><div><div style="font-size: 14px; font-weight: 600; color: {c["text"]};">{source["name"]}{demo_badge}</div><div style="font-size: 11px; color: {c["text_muted"]};">{source.get("type", "Unknown")}</div></div></div><div style="text-align: center;"><span style="display: inline-flex; align-items: center; gap: 6px; padding: 4px 10px; border-radius: 9999px; background: {status_color}15; color: {status_color}; font-size: 11px; font-weight: 600;">{source.get("status", "unknown").title()}</span></div><div style="text-align: center; font-size: 13px; color: {c["text"]};">{rows_display}</div><div style="text-align: center;"><span style="padding: 4px 10px; border-radius: 8px; background: {freshness_color}15; color: {freshness_color}; font-size: 11px; font-weight: 500;">{source.get("freshness", "unknown").title()}</span></div></div>'
                    st.markdown(row_html, unsafe_allow_html=True)
                
                with row_cols[1]:
                    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
                    if st.button("🔌 Disconnect", key=f"disconnect_{source.get('id', i)}_{i}", use_container_width=True):
                        remove_data_source(source.get('id'))
                        # Also clear the data from session
                        if f"data_{source.get('id')}" in st.session_state:
                            del st.session_state[f"data_{source.get('id')}"]
                        st.success(f"✅ Disconnected: {source['name']}")
                        st.rerun()
        else:
            st.info("📭 No data sources connected. Use 'Quick Upload' above to add your data, or 'Reset to Default' to restore demo data.")



if __name__ == "__main__":
    render()
