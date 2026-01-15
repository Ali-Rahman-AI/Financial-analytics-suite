"""
Financial Analytics Suite - Model Results Manager
Centralized storage for all model runs and their results
This allows the Dashboard to show ONLY models that have been run
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
import json


def init_model_results():
    """Initialize the model results storage in session state"""
    if 'model_results' not in st.session_state:
        st.session_state.model_results = {
            'forecasting': {},      # Stores forecasting model results
            'ml_models': {},        # Stores ML model results
            'portfolio': {},        # Stores portfolio analysis results
            'scenario': {},         # Stores scenario analysis results
            'risk': {},             # Stores risk analysis results
            'comparison_queue': [], # Models queued for comparison
            'run_history': [],      # History of all model runs
        }


def get_model_results() -> Dict:
    """Get all stored model results"""
    init_model_results()
    return st.session_state.model_results


def save_model_result(
    category: str,
    model_name: str,
    result: Dict,
    metrics: Optional[Dict] = None,
    plot_data: Optional[Dict] = None,
    parameters: Optional[Dict] = None
) -> str:
    """
    Save a model result to session state
    
    Args:
        category: One of 'forecasting', 'ml_models', 'portfolio', 'scenario', 'risk'
        model_name: Name of the model (e.g., 'ARIMA', 'XGBoost', 'Monte Carlo')
        result: The main result data (predictions, values, etc.)
        metrics: Performance metrics (RMSE, MAE, R², etc.)
        plot_data: Data needed to recreate the plot
        parameters: Parameters used for the model run
        
    Returns:
        run_id: Unique identifier for this run
    """
    init_model_results()
    
    run_id = f"{category}_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    run_record = {
        'run_id': run_id,
        'model_name': model_name,
        'category': category,
        'timestamp': datetime.now().isoformat(),
        'result': result,
        'metrics': metrics or {},
        'plot_data': plot_data or {},
        'parameters': parameters or {},
        'status': 'completed'
    }
    
    # Store in the appropriate category
    if category in st.session_state.model_results:
        st.session_state.model_results[category][model_name] = run_record
    
    # Add to run history
    history_entry = {
        'run_id': run_id,
        'model_name': model_name,
        'category': category,
        'timestamp': run_record['timestamp'],
        'metrics_summary': metrics.get('primary_metric', 'N/A') if metrics else 'N/A'
    }
    st.session_state.model_results['run_history'].append(history_entry)
    
    # Keep only last 50 history entries
    if len(st.session_state.model_results['run_history']) > 50:
        st.session_state.model_results['run_history'] = \
            st.session_state.model_results['run_history'][-50:]
    
    return run_id


def get_category_results(category: str) -> Dict:
    """Get all results for a specific category"""
    init_model_results()
    return st.session_state.model_results.get(category, {})


def get_model_result(category: str, model_name: str) -> Optional[Dict]:
    """Get a specific model result"""
    init_model_results()
    return st.session_state.model_results.get(category, {}).get(model_name)


def has_model_been_run(category: str, model_name: str) -> bool:
    """Check if a specific model has been run"""
    init_model_results()
    return model_name in st.session_state.model_results.get(category, {})


def get_run_history() -> List[Dict]:
    """Get the run history"""
    init_model_results()
    return st.session_state.model_results.get('run_history', [])


def get_all_run_models() -> Dict[str, List[str]]:
    """Get all models that have been run, organized by category"""
    init_model_results()
    
    run_models = {}
    for category in ['forecasting', 'ml_models', 'portfolio', 'scenario', 'risk']:
        models = list(st.session_state.model_results.get(category, {}).keys())
        if models:
            run_models[category] = models
    
    return run_models


def get_models_for_comparison(category: str = None) -> List[Dict]:
    """Get models available for comparison (only those that have been run)"""
    init_model_results()
    
    models = []
    categories = [category] if category else ['forecasting', 'ml_models', 'portfolio', 'scenario', 'risk']
    
    for cat in categories:
        cat_results = st.session_state.model_results.get(cat, {})
        for model_name, result in cat_results.items():
            models.append({
                'category': cat,
                'model_name': model_name,
                'run_id': result.get('run_id'),
                'timestamp': result.get('timestamp'),
                'metrics': result.get('metrics', {}),
                'plot_data': result.get('plot_data', {})
            })
    
    return models


def add_to_comparison_queue(category: str, model_name: str):
    """Add a model to the comparison queue"""
    init_model_results()
    
    queue_item = {'category': category, 'model_name': model_name}
    if queue_item not in st.session_state.model_results['comparison_queue']:
        st.session_state.model_results['comparison_queue'].append(queue_item)


def remove_from_comparison_queue(category: str, model_name: str):
    """Remove a model from the comparison queue"""
    init_model_results()
    
    queue_item = {'category': category, 'model_name': model_name}
    if queue_item in st.session_state.model_results['comparison_queue']:
        st.session_state.model_results['comparison_queue'].remove(queue_item)


def get_comparison_queue() -> List[Dict]:
    """Get the current comparison queue"""
    init_model_results()
    return st.session_state.model_results.get('comparison_queue', [])


def clear_comparison_queue():
    """Clear the comparison queue"""
    init_model_results()
    st.session_state.model_results['comparison_queue'] = []


def clear_model_result(category: str, model_name: str):
    """Clear a specific model result"""
    init_model_results()
    if category in st.session_state.model_results:
        if model_name in st.session_state.model_results[category]:
            del st.session_state.model_results[category][model_name]


def clear_category_results(category: str):
    """Clear all results for a category"""
    init_model_results()
    if category in st.session_state.model_results:
        st.session_state.model_results[category] = {}


def clear_all_results():
    """Clear all model results"""
    st.session_state.model_results = {
        'forecasting': {},
        'ml_models': {},
        'portfolio': {},
        'scenario': {},
        'risk': {},
        'comparison_queue': [],
        'run_history': [],
    }


def get_dashboard_data() -> Dict:
    """
    Get all data needed for the dashboard
    Returns only models that have been run
    """
    init_model_results()
    
    dashboard_data = {
        'has_results': False,
        'categories': {},
        'total_runs': 0,
        'latest_run': None,
        'metrics_summary': {}
    }
    
    run_models = get_all_run_models()
    
    if not run_models:
        return dashboard_data
    
    dashboard_data['has_results'] = True
    
    # Collect all results
    for category, models in run_models.items():
        dashboard_data['categories'][category] = []
        
        for model_name in models:
            result = get_model_result(category, model_name)
            if result:
                dashboard_data['categories'][category].append({
                    'model_name': model_name,
                    'metrics': result.get('metrics', {}),
                    'plot_data': result.get('plot_data', {}),
                    'timestamp': result.get('timestamp')
                })
                dashboard_data['total_runs'] += 1
    
    # Get latest run
    history = get_run_history()
    if history:
        dashboard_data['latest_run'] = history[-1]
    
    return dashboard_data


def count_run_models() -> int:
    """Count total number of models that have been run"""
    init_model_results()
    total = 0
    for category in ['forecasting', 'ml_models', 'portfolio', 'scenario', 'risk']:
        total += len(st.session_state.model_results.get(category, {}))
    return total
