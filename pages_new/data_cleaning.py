"""
Financial Analytics Suite - Data Cleaning & Editor Page
View, edit, clean, transform, and prepare your data for analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Optional
from io import BytesIO

# Import data manager
from pages_new.data_manager import (
    get_working_data, get_working_data_info, has_data
)


# Import theme utilities
from pages_new.theme_utils import get_theme_colors, inject_premium_styles


def get_data_profile(df: pd.DataFrame) -> Dict:
    """Get comprehensive data profile"""
    profile = {
        'rows': len(df),
        'columns': len(df.columns),
        'missing_total': df.isnull().sum().sum(),
        'missing_pct': (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100) if len(df) > 0 else 0,
        'duplicates': df.duplicated().sum(),
        'memory': df.memory_usage(deep=True).sum(),
        'dtypes': df.dtypes.value_counts().to_dict(),
        'column_info': []
    }
    
    for col in df.columns:
        col_info = {
            'name': col,
            'dtype': str(df[col].dtype),
            'missing': df[col].isnull().sum(),
            'missing_pct': (df[col].isnull().sum() / len(df) * 100) if len(df) > 0 else 0,
            'unique': df[col].nunique(),
        }
        
        if pd.api.types.is_numeric_dtype(df[col]):
            col_info['min'] = df[col].min()
            col_info['max'] = df[col].max()
            col_info['mean'] = df[col].mean()
            col_info['std'] = df[col].std()
        
        profile['column_info'].append(col_info)
    
    return profile


def render_no_data_message(c: Dict):
    """Render a message when no data is available"""
    st.markdown(f'''
    <div class="glass-card" style="padding: 60px 40px; text-align: center; margin: 20px 0; animation: fadeIn 0.8s ease-out;">
        <div style="font-size: 80px; margin-bottom: 24px; filter: drop-shadow(0 0 20px {c['primary']}40);">🧹</div>
        <h2 class="glass-header" style="font-size: 32px; margin-bottom: 16px;">No Data to Clean</h2>
        <p style="color: {c['text_secondary']}; font-size: 16px; max-width: 450px; margin: 0 auto 30px;">
            Your data playground is waiting. Connect to a data source first to enable the advanced cleaning and transformation suite.
        </p>
    </div>
    ''', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("📥 Go to Data Sources", type="primary", key="goto_data_src", use_container_width=True):
            st.session_state.current_page = 'data_sources'
            st.rerun()


def format_bytes(size: int) -> str:
    """Format byte size to human readable"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def render():
    """Render the Data Cleaning & Editor page"""
    c = get_theme_colors()
    inject_premium_styles()
    
    st.title("🧹 Data Cleaning & Editor")
    st.markdown(f"<p style='color: {c['text_secondary']};'>View, edit, clean, and transform your data</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Check if we have data
    if not has_data():
        render_no_data_message(c)
        return
    
    # Get the working data
    df = get_working_data()
    
    # Extra safety check
    if df is None:
        render_no_data_message(c)
        return
    
    data_info = get_working_data_info()
    
    # Initialize cleaned data in session state - reset if None or if original data changed
    if 'cleaned_data' not in st.session_state or st.session_state.cleaned_data is None:
        st.session_state.cleaned_data = df.copy()
    
    # Working data reference
    working_df = st.session_state.cleaned_data
    
    # Safety check for working_df
    if working_df is None:
        st.session_state.cleaned_data = df.copy()
        working_df = st.session_state.cleaned_data
    
    profile = get_data_profile(working_df)
    
    # Data info bar
    st.markdown(f'<div style="background: rgba(99, 102, 241, 0.1); border: 1px solid rgba(99, 102, 241, 0.3); border-radius: 12px; padding: 12px 20px; margin-bottom: 20px; display: flex; align-items: center; gap: 20px;"><span style="font-size: 16px;">📊</span><div><span style="color: {c["text"]}; font-weight: 600;">Working Data:</span> <span style="color: {c["text_secondary"]};">{data_info["name"]}</span></div><div style="color: {c["text_muted"]};">|</div><div><span style="color: {c["text_muted"]};">{profile["rows"]:,} rows × {profile["columns"]} columns</span></div><div style="color: {c["text_muted"]};">|</div><div><span style="color: {c["text_muted"]};">{format_bytes(profile["memory"])}</span></div></div>', unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📝 Data Editor", "📋 Profile", "🧹 Clean", "🔧 Transform", "📤 Export"])
    
    # =============================================
    # TAB 1: DATA EDITOR - View all data
    # =============================================
    with tab1:
        st.markdown("#### 📝 Full Data View")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.caption(f"Showing all {profile['rows']:,} rows and {profile['columns']} columns")
        with col2:
            show_index = st.checkbox("Show row index", value=True, key="show_idx")
        
        # Display full dataframe with pagination option
        rows_per_page = st.selectbox("Rows per page", [50, 100, 250, 500, 1000, "All"], index=1, key="rows_pp")
        
        if rows_per_page == "All":
            display_df = working_df
        else:
            total_pages = (len(working_df) - 1) // rows_per_page + 1
            page_cols = st.columns([1, 3, 1])
            with page_cols[1]:
                page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, key="page_num")
            
            start_idx = (page - 1) * rows_per_page
            end_idx = min(start_idx + rows_per_page, len(working_df))
            display_df = working_df.iloc[start_idx:end_idx]
            st.caption(f"Showing rows {start_idx + 1:,} to {end_idx:,} of {len(working_df):,}")
        
        # Display the dataframe - full view with scrolling
        if show_index:
            st.dataframe(display_df, use_container_width=True, height=500)
        else:
            st.dataframe(display_df.reset_index(drop=True), use_container_width=True, height=500)
        
        # Column selection for focused view
        st.markdown("---")
        st.markdown("#### 🔍 Column Focus")
        selected_cols = st.multiselect(
            "Select columns to view",
            options=list(working_df.columns),
            default=list(working_df.columns[:5]) if len(working_df.columns) > 5 else list(working_df.columns),
            key="focus_cols"
        )
        
        if selected_cols:
            st.dataframe(working_df[selected_cols], use_container_width=True, height=300)
        
        # Quick stats for selected column
        st.markdown("---")
        st.markdown("#### 📊 Column Statistics")
        stat_col = st.selectbox("Select column for statistics", working_df.columns, key="stat_col")
        
        if stat_col:
            col_data = working_df[stat_col]
            
            stat_cols = st.columns(5)
            with stat_cols[0]:
                st.metric("Non-null Count", f"{col_data.count():,}")
            with stat_cols[1]:
                st.metric("Null Count", f"{col_data.isnull().sum():,}")
            with stat_cols[2]:
                st.metric("Unique Values", f"{col_data.nunique():,}")
            with stat_cols[3]:
                st.metric("Data Type", str(col_data.dtype))
            with stat_cols[4]:
                if pd.api.types.is_numeric_dtype(col_data):
                    st.metric("Mean", f"{col_data.mean():.4f}")
                else:
                    st.metric("Most Common", str(col_data.mode().iloc[0]) if len(col_data.mode()) > 0 else "N/A")
            
            if pd.api.types.is_numeric_dtype(col_data):
                num_cols = st.columns(4)
                with num_cols[0]:
                    st.metric("Min", f"{col_data.min():.4f}")
                with num_cols[1]:
                    st.metric("Max", f"{col_data.max():.4f}")
                with num_cols[2]:
                    st.metric("Std Dev", f"{col_data.std():.4f}")
                with num_cols[3]:
                    st.metric("Median", f"{col_data.median():.4f}")
    
    # =============================================
    # TAB 2: DATA PROFILE
    # =============================================
    with tab2:
        st.markdown("#### 📋 Data Profile")
        
        # Overview metrics
        metric_cols = st.columns(5)
        with metric_cols[0]:
            st.metric("Total Rows", f"{profile['rows']:,}")
        with metric_cols[1]:
            st.metric("Total Columns", f"{profile['columns']}")
        with metric_cols[2]:
            st.metric("Missing Values", f"{profile['missing_total']:,}", delta=f"{profile['missing_pct']:.1f}%")
        with metric_cols[3]:
            st.metric("Duplicate Rows", f"{profile['duplicates']:,}")
        with metric_cols[4]:
            st.metric("Memory Usage", format_bytes(profile['memory']))
        
        st.markdown("---")
        
        # Column details
        st.markdown("#### Column Details")
        
        col_details = []
        for col_info in profile['column_info']:
            row = {
                'Column': col_info['name'],
                'Type': col_info['dtype'],
                'Missing': col_info['missing'],
                'Missing %': f"{col_info['missing_pct']:.1f}%",
                'Unique': col_info['unique'],
            }
            if 'mean' in col_info:
                row['Mean'] = f"{col_info['mean']:.2f}"
                row['Std'] = f"{col_info['std']:.2f}"
            else:
                row['Mean'] = '-'
                row['Std'] = '-'
            col_details.append(row)
        
        st.dataframe(pd.DataFrame(col_details), use_container_width=True, hide_index=True)
        
        # Data types distribution
        st.markdown("---")
        st.markdown("#### Data Types Distribution")
        dtype_cols = st.columns(len(profile['dtypes']))
        for i, (dtype, count) in enumerate(profile['dtypes'].items()):
            with dtype_cols[i]:
                st.metric(str(dtype), count)
    
    # =============================================
    # TAB 3: CLEANING OPERATIONS
    # =============================================
    with tab3:
        st.markdown("#### 🧹 Data Cleaning")
        
        cleaning_col1, cleaning_col2 = st.columns([1, 1])
        
        with cleaning_col1:
            # Missing Values
            st.markdown("##### 🔍 Handle Missing Values")
            
            missing_cols = working_df.columns[working_df.isnull().any()].tolist()
            
            if missing_cols:
                st.warning(f"Found {len(missing_cols)} columns with missing values")
                
                missing_strategy = st.selectbox(
                    "Strategy",
                    ["Drop rows with missing", "Drop columns with missing", "Fill with mean", 
                     "Fill with median", "Fill with mode", "Fill with value", "Forward fill", "Backward fill"],
                    key="missing_strat"
                )
                
                if missing_strategy == "Fill with value":
                    fill_value = st.text_input("Fill value", value="0", key="fill_val")
                
                apply_to_cols = st.multiselect(
                    "Apply to columns",
                    options=missing_cols,
                    default=missing_cols,
                    key="missing_cols"
                )
                
                if st.button("🔧 Apply Missing Value Fix", type="primary", key="fix_missing", use_container_width=True):
                    temp_df = st.session_state.cleaned_data.copy()
                    
                    if missing_strategy == "Drop rows with missing":
                        temp_df = temp_df.dropna(subset=apply_to_cols)
                    elif missing_strategy == "Drop columns with missing":
                        temp_df = temp_df.drop(columns=apply_to_cols)
                    elif missing_strategy == "Fill with mean":
                        for col in apply_to_cols:
                            if pd.api.types.is_numeric_dtype(temp_df[col]):
                                temp_df[col] = temp_df[col].fillna(temp_df[col].mean())
                    elif missing_strategy == "Fill with median":
                        for col in apply_to_cols:
                            if pd.api.types.is_numeric_dtype(temp_df[col]):
                                temp_df[col] = temp_df[col].fillna(temp_df[col].median())
                    elif missing_strategy == "Fill with mode":
                        for col in apply_to_cols:
                            mode_val = temp_df[col].mode()
                            if len(mode_val) > 0:
                                temp_df[col] = temp_df[col].fillna(mode_val.iloc[0])
                    elif missing_strategy == "Fill with value":
                        for col in apply_to_cols:
                            temp_df[col] = temp_df[col].fillna(fill_value)
                    elif missing_strategy == "Forward fill":
                        temp_df[apply_to_cols] = temp_df[apply_to_cols].ffill()
                    elif missing_strategy == "Backward fill":
                        temp_df[apply_to_cols] = temp_df[apply_to_cols].bfill()
                    
                    st.session_state.cleaned_data = temp_df
                    st.success(f"✅ Applied: {missing_strategy}")
                    st.rerun()
            else:
                st.success("✅ No missing values found!")
            
            st.markdown("---")
            
            # Duplicates
            st.markdown("##### 🔄 Remove Duplicates")
            
            dup_count = working_df.duplicated().sum()
            st.metric("Duplicate Rows", dup_count)
            
            if dup_count > 0:
                dup_subset = st.multiselect(
                    "Check duplicates based on columns (leave empty for all)",
                    options=list(working_df.columns),
                    key="dup_cols"
                )
                
                keep_option = st.selectbox("Keep which duplicate?", ["first", "last", "none"], key="dup_keep")
                
                if st.button("🗑️ Remove Duplicates", type="primary", key="remove_dups", use_container_width=True):
                    if dup_subset:
                        st.session_state.cleaned_data = st.session_state.cleaned_data.drop_duplicates(
                            subset=dup_subset, keep=keep_option if keep_option != 'none' else False
                        )
                    else:
                        st.session_state.cleaned_data = st.session_state.cleaned_data.drop_duplicates(
                            keep=keep_option if keep_option != 'none' else False
                        )
                    st.success(f"✅ Removed {dup_count} duplicate rows")
                    st.rerun()
        
        with cleaning_col2:
            # Data Type Conversion
            st.markdown("##### 🔢 Convert Data Types")
            
            col_to_convert = st.selectbox("Select column", working_df.columns, key="dtype_col")
            current_type = str(working_df[col_to_convert].dtype)
            st.caption(f"Current type: **{current_type}**")
            
            new_type = st.selectbox(
                "Convert to",
                ["int64", "float64", "string", "datetime64", "category", "bool"],
                key="new_dtype"
            )
            
            if st.button("🔄 Convert Type", key="convert_type", use_container_width=True):
                try:
                    temp_df = st.session_state.cleaned_data.copy()
                    if new_type == "datetime64":
                        temp_df[col_to_convert] = pd.to_datetime(temp_df[col_to_convert])
                    elif new_type == "category":
                        temp_df[col_to_convert] = temp_df[col_to_convert].astype('category')
                    else:
                        temp_df[col_to_convert] = temp_df[col_to_convert].astype(new_type)
                    st.session_state.cleaned_data = temp_df
                    st.success(f"✅ Converted {col_to_convert} to {new_type}")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
            
            st.markdown("---")
            
            # Drop Columns
            st.markdown("##### 🗑️ Drop Columns")
            
            cols_to_drop = st.multiselect(
                "Select columns to remove",
                options=list(working_df.columns),
                key="drop_cols"
            )
            
            if cols_to_drop:
                if st.button("🗑️ Drop Selected Columns", type="secondary", key="drop_cols_btn", use_container_width=True):
                    st.session_state.cleaned_data = st.session_state.cleaned_data.drop(columns=cols_to_drop)
                    st.success(f"✅ Dropped {len(cols_to_drop)} columns")
                    st.rerun()
            
            st.markdown("---")
            
            # Rename Column
            st.markdown("##### ✏️ Rename Column")
            
            col_to_rename = st.selectbox("Select column to rename", working_df.columns, key="rename_col")
            new_name = st.text_input("New name", value=col_to_rename, key="new_col_name")
            
            if st.button("✏️ Rename", key="rename_btn", use_container_width=True):
                if new_name and new_name != col_to_rename:
                    st.session_state.cleaned_data = st.session_state.cleaned_data.rename(
                        columns={col_to_rename: new_name}
                    )
                    st.success(f"✅ Renamed {col_to_rename} to {new_name}")
                    st.rerun()
        
        st.markdown("---")
        
        # Reset button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🔄 Reset to Original Data", key="reset_data", use_container_width=True):
                st.session_state.cleaned_data = df.copy()
                st.success("✅ Reset to original data")
                st.rerun()
    
    # =============================================
    # TAB 4: TRANSFORM
    # =============================================
    with tab4:
        st.markdown("#### 🔧 Data Transformations")
        
        trans_col1, trans_col2 = st.columns([1, 1])
        
        with trans_col1:
            # Filter Data
            st.markdown("##### 🔍 Filter Data")
            
            filter_col = st.selectbox("Filter column", working_df.columns, key="filter_col")
            
            if pd.api.types.is_numeric_dtype(working_df[filter_col]):
                filter_op = st.selectbox("Operator", [">", ">=", "<", "<=", "==", "!="], key="filter_op")
                filter_val = st.number_input("Value", value=0.0, key="filter_val")
            else:
                filter_op = st.selectbox("Operator", ["equals", "contains", "starts with", "ends with"], key="filter_op_str")
                filter_val = st.text_input("Value", key="filter_val_str")
            
            if st.button("🔍 Apply Filter", key="apply_filter", use_container_width=True):
                temp_df = st.session_state.cleaned_data.copy()
                
                if pd.api.types.is_numeric_dtype(temp_df[filter_col]):
                    if filter_op == ">":
                        temp_df = temp_df[temp_df[filter_col] > filter_val]
                    elif filter_op == ">=":
                        temp_df = temp_df[temp_df[filter_col] >= filter_val]
                    elif filter_op == "<":
                        temp_df = temp_df[temp_df[filter_col] < filter_val]
                    elif filter_op == "<=":
                        temp_df = temp_df[temp_df[filter_col] <= filter_val]
                    elif filter_op == "==":
                        temp_df = temp_df[temp_df[filter_col] == filter_val]
                    elif filter_op == "!=":
                        temp_df = temp_df[temp_df[filter_col] != filter_val]
                else:
                    if filter_op == "equals":
                        temp_df = temp_df[temp_df[filter_col] == filter_val]
                    elif filter_op == "contains":
                        temp_df = temp_df[temp_df[filter_col].astype(str).str.contains(filter_val, na=False)]
                    elif filter_op == "starts with":
                        temp_df = temp_df[temp_df[filter_col].astype(str).str.startswith(filter_val, na=False)]
                    elif filter_op == "ends with":
                        temp_df = temp_df[temp_df[filter_col].astype(str).str.endswith(filter_val, na=False)]
                
                st.session_state.cleaned_data = temp_df
                st.success(f"✅ Applied filter: {len(temp_df):,} rows remaining")
                st.rerun()
            
            st.markdown("---")
            
            # Sort Data
            st.markdown("##### ↕️ Sort Data")
            
            sort_col = st.selectbox("Sort by column", working_df.columns, key="sort_col")
            sort_order = st.radio("Order", ["Ascending", "Descending"], horizontal=True, key="sort_order")
            
            if st.button("↕️ Apply Sort", key="apply_sort", use_container_width=True):
                st.session_state.cleaned_data = st.session_state.cleaned_data.sort_values(
                    by=sort_col, ascending=(sort_order == "Ascending")
                ).reset_index(drop=True)
                st.success(f"✅ Sorted by {sort_col} ({sort_order})")
                st.rerun()
        
        with trans_col2:
            # Create New Column
            st.markdown("##### ➕ Create New Column")
            
            new_col_name = st.text_input("New column name", placeholder="e.g., returns_pct", key="new_col")
            
            transform_type = st.selectbox(
                "Transform type",
                ["Mathematical operation", "Percentage change", "Rolling mean", "Lag/Lead", "Cumulative sum"],
                key="transform_type"
            )
            
            if transform_type == "Mathematical operation":
                math_col = st.selectbox("Source column", working_df.select_dtypes(include=[np.number]).columns, key="math_col")
                math_op = st.selectbox("Operation", ["* (multiply)", "/ (divide)", "+ (add)", "- (subtract)", "** (power)"], key="math_op")
                math_val = st.number_input("Value", value=1.0, key="math_val")
            elif transform_type == "Percentage change":
                pct_col = st.selectbox("Source column", working_df.select_dtypes(include=[np.number]).columns, key="pct_col")
            elif transform_type == "Rolling mean":
                roll_col = st.selectbox("Source column", working_df.select_dtypes(include=[np.number]).columns, key="roll_col")
                window_size = st.number_input("Window size", min_value=2, max_value=100, value=5, key="window")
            elif transform_type == "Lag/Lead":
                lag_col = st.selectbox("Source column", working_df.columns, key="lag_col")
                lag_periods = st.number_input("Periods (negative for lead)", min_value=-50, max_value=50, value=1, key="lag_per")
            elif transform_type == "Cumulative sum":
                cum_col = st.selectbox("Source column", working_df.select_dtypes(include=[np.number]).columns, key="cum_col")
            
            if st.button("➕ Create Column", type="primary", key="create_col", use_container_width=True):
                if not new_col_name:
                    st.error("Please enter a column name")
                else:
                    try:
                        temp_df = st.session_state.cleaned_data.copy()
                        
                        if transform_type == "Mathematical operation":
                            op_map = {"* (multiply)": "*", "/ (divide)": "/", "+ (add)": "+", "- (subtract)": "-", "** (power)": "**"}
                            op = op_map[math_op]
                            if op == "*":
                                temp_df[new_col_name] = temp_df[math_col] * math_val
                            elif op == "/":
                                temp_df[new_col_name] = temp_df[math_col] / math_val
                            elif op == "+":
                                temp_df[new_col_name] = temp_df[math_col] + math_val
                            elif op == "-":
                                temp_df[new_col_name] = temp_df[math_col] - math_val
                            elif op == "**":
                                temp_df[new_col_name] = temp_df[math_col] ** math_val
                        elif transform_type == "Percentage change":
                            temp_df[new_col_name] = temp_df[pct_col].pct_change() * 100
                        elif transform_type == "Rolling mean":
                            temp_df[new_col_name] = temp_df[roll_col].rolling(window=window_size).mean()
                        elif transform_type == "Lag/Lead":
                            temp_df[new_col_name] = temp_df[lag_col].shift(lag_periods)
                        elif transform_type == "Cumulative sum":
                            temp_df[new_col_name] = temp_df[cum_col].cumsum()
                        
                        st.session_state.cleaned_data = temp_df
                        st.success(f"✅ Created column: {new_col_name}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
    
    # =============================================
    # TAB 5: EXPORT
    # =============================================
    with tab5:
        st.markdown("#### 📤 Export Cleaned Data")
        
        # Export options
        exp_col1, exp_col2 = st.columns(2)
        
        with exp_col1:
            export_format = st.selectbox("Format", ["CSV", "Excel", "JSON"], key="exp_format")
            include_index = st.checkbox("Include row index", value=False, key="exp_idx")
        
        with exp_col2:
            filename = st.text_input("Filename (without extension)", value="cleaned_data", key="exp_filename")
        
        st.markdown("---")
        
        # Preview
        st.markdown("##### Preview (first 5 rows)")
        st.dataframe(working_df.head(), use_container_width=True)
        
        st.markdown(f"**Total rows to export:** {len(working_df):,}")
        
        # Download button
        if export_format == "CSV":
            csv_data = working_df.to_csv(index=include_index)
            st.download_button(
                "📥 Download CSV",
                data=csv_data,
                file_name=f"{filename}.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_csv"
            )
        elif export_format == "Excel":
            buffer = BytesIO()
            working_df.to_excel(buffer, index=include_index, engine='openpyxl')
            buffer.seek(0)
            st.download_button(
                "📥 Download Excel",
                data=buffer,
                file_name=f"{filename}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key="download_xlsx"
            )
        elif export_format == "JSON":
            json_data = working_df.to_json(orient='records', indent=2)
            st.download_button(
                "📥 Download JSON",
                data=json_data,
                file_name=f"{filename}.json",
                mime="application/json",
                use_container_width=True,
                key="download_json"
            )
        
        st.markdown("---")
        
        # Save to session (update main data)
        if st.button("💾 Save as Working Data", type="primary", key="save_working", use_container_width=True):
            st.session_state.uploaded_data = working_df.copy()
            st.success("✅ Cleaned data saved as working data! Use this in other analysis pages.")


if __name__ == "__main__":
    render()
