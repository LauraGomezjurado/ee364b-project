#!/usr/bin/env python3
"""
Interactive Dashboard for Experiment Monitoring

This creates a web-based dashboard to monitor experiments in real-time,
compare results, and analyze performance across different configurations.

Run with: python interactive_dashboard.py
Then open http://localhost:8050 in your browser
"""

import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
import json
from pathlib import Path
import time
from datetime import datetime
import threading
import queue

# Try to import the experiment components
try:
    from experiment_runner import ExperimentRunner, ExperimentConfig
    from quick_wins import run_experiment
    EXPERIMENTS_AVAILABLE = True
except ImportError:
    EXPERIMENTS_AVAILABLE = False
    print("Warning: Experiment modules not available. Dashboard will show demo data.")

class ExperimentDashboard:
    """Interactive dashboard for experiment monitoring"""
    
    def __init__(self, port=8050):
        self.app = dash.Dash(__name__)
        self.port = port
        self.experiment_queue = queue.Queue()
        self.results_cache = {}
        self.setup_layout()
        self.setup_callbacks()
        
    def setup_layout(self):
        """Setup the dashboard layout"""
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("🧪 Experiment Dashboard", 
                       style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 30}),
                html.P("Real-time monitoring and analysis of dual optimization experiments",
                       style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': 18})
            ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'marginBottom': '20px'}),
            
            # Control Panel
            html.Div([
                html.H3("🎛️ Control Panel"),
                html.Div([
                    # Experiment Selection
                    html.Div([
                        html.Label("Experiment Type:"),
                        dcc.Dropdown(
                            id='experiment-type',
                            options=[
                                {'label': 'Quick Wins Suite', 'value': 'quick_wins'},
                                {'label': 'Stress Test Suite', 'value': 'stress_test'},
                                {'label': 'Algorithm Comparison', 'value': 'algo_compare'},
                                {'label': 'Custom Experiment', 'value': 'custom'}
                            ],
                            value='quick_wins'
                        )
                    ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '5%'}),
                    
                    # Parameters
                    html.Div([
                        html.Label("Iterations:"),
                        dcc.Input(id='iterations-input', type='number', value=2000, min=100, max=10000)
                    ], style={'width': '20%', 'display': 'inline-block', 'marginRight': '5%'}),
                    
                    # Run Button
                    html.Div([
                        html.Button('🚀 Run Experiments', id='run-button', n_clicks=0,
                                   style={'backgroundColor': '#3498db', 'color': 'white', 
                                         'border': 'none', 'padding': '10px 20px', 
                                         'borderRadius': '5px', 'cursor': 'pointer'})
                    ], style={'width': '20%', 'display': 'inline-block', 'marginTop': '25px'}),
                    
                    # Status
                    html.Div([
                        html.Div(id='status-indicator', children="Ready", 
                                style={'padding': '10px', 'borderRadius': '5px', 
                                      'backgroundColor': '#2ecc71', 'color': 'white', 'textAlign': 'center'})
                    ], style={'width': '15%', 'display': 'inline-block', 'marginLeft': '5%', 'marginTop': '25px'})
                ])
            ], style={'backgroundColor': '#f8f9fa', 'padding': '20px', 'marginBottom': '20px', 'borderRadius': '10px'}),
            
            # Real-time Monitoring
            html.Div([
                html.H3("📊 Real-time Monitoring"),
                dcc.Graph(id='live-convergence-plot'),
                dcc.Interval(id='interval-component', interval=2000, n_intervals=0)  # Update every 2 seconds
            ], style={'marginBottom': '20px'}),
            
            # Results Analysis
            html.Div([
                html.H3("📈 Results Analysis"),
                html.Div([
                    # Performance Comparison
                    html.Div([
                        dcc.Graph(id='performance-comparison')
                    ], style={'width': '50%', 'display': 'inline-block'}),
                    
                    # Trade-off Analysis
                    html.Div([
                        dcc.Graph(id='tradeoff-analysis')
                    ], style={'width': '50%', 'display': 'inline-block'})
                ])
            ], style={'marginBottom': '20px'}),
            
            # Detailed Results Table
            html.Div([
                html.H3("📋 Detailed Results"),
                html.Div(id='results-table')
            ], style={'marginBottom': '20px'}),
            
            # Algorithm Insights
            html.Div([
                html.H3("🧠 Algorithm Insights"),
                html.Div(id='insights-panel')
            ]),
            
            # Hidden div to store data
            html.Div(id='experiment-data', style={'display': 'none'})
        ])
    
    def setup_callbacks(self):
        """Setup dashboard callbacks"""
        
        @self.app.callback(
            [Output('status-indicator', 'children'),
             Output('status-indicator', 'style'),
             Output('experiment-data', 'children')],
            [Input('run-button', 'n_clicks')],
            [dash.dependencies.State('experiment-type', 'value'),
             dash.dependencies.State('iterations-input', 'value')]
        )
        def run_experiments(n_clicks, exp_type, iterations):
            if n_clicks == 0:
                return "Ready", {'padding': '10px', 'borderRadius': '5px', 
                               'backgroundColor': '#2ecc71', 'color': 'white', 'textAlign': 'center'}, ""
            
            if not EXPERIMENTS_AVAILABLE:
                # Return demo data
                demo_data = self.generate_demo_data()
                return "Demo Mode", {'padding': '10px', 'borderRadius': '5px', 
                                   'backgroundColor': '#f39c12', 'color': 'white', 'textAlign': 'center'}, json.dumps(demo_data)
            
            # Run actual experiments
            try:
                results = self.run_experiment_suite(exp_type, iterations)
                return "Completed", {'padding': '10px', 'borderRadius': '5px', 
                                   'backgroundColor': '#2ecc71', 'color': 'white', 'textAlign': 'center'}, json.dumps(results)
            except Exception as e:
                return f"Error: {str(e)}", {'padding': '10px', 'borderRadius': '5px', 
                                          'backgroundColor': '#e74c3c', 'color': 'white', 'textAlign': 'center'}, ""
        
        @self.app.callback(
            Output('live-convergence-plot', 'figure'),
            [Input('interval-component', 'n_intervals'),
             Input('experiment-data', 'children')]
        )
        def update_live_plot(n_intervals, experiment_data):
            if not experiment_data:
                return self.create_empty_plot("No data available")
            
            try:
                data = json.loads(experiment_data)
                return self.create_convergence_plot(data)
            except:
                return self.create_empty_plot("Error loading data")
        
        @self.app.callback(
            Output('performance-comparison', 'figure'),
            [Input('experiment-data', 'children')]
        )
        def update_performance_plot(experiment_data):
            if not experiment_data:
                return self.create_empty_plot("No data available")
            
            try:
                data = json.loads(experiment_data)
                return self.create_performance_comparison(data)
            except:
                return self.create_empty_plot("Error loading data")
        
        @self.app.callback(
            Output('tradeoff-analysis', 'figure'),
            [Input('experiment-data', 'children')]
        )
        def update_tradeoff_plot(experiment_data):
            if not experiment_data:
                return self.create_empty_plot("No data available")
            
            try:
                data = json.loads(experiment_data)
                return self.create_tradeoff_plot(data)
            except:
                return self.create_empty_plot("Error loading data")
        
        @self.app.callback(
            Output('results-table', 'children'),
            [Input('experiment-data', 'children')]
        )
        def update_results_table(experiment_data):
            if not experiment_data:
                return html.P("No results available")
            
            try:
                data = json.loads(experiment_data)
                return self.create_results_table(data)
            except:
                return html.P("Error loading results")
        
        @self.app.callback(
            Output('insights-panel', 'children'),
            [Input('experiment-data', 'children')]
        )
        def update_insights(experiment_data):
            if not experiment_data:
                return html.P("No insights available")
            
            try:
                data = json.loads(experiment_data)
                return self.create_insights_panel(data)
            except:
                return html.P("Error generating insights")
    
    def run_experiment_suite(self, exp_type, iterations):
        """Run the selected experiment suite"""
        if exp_type == 'quick_wins':
            return self.run_quick_wins_suite(iterations)
        elif exp_type == 'stress_test':
            return self.run_stress_test_suite(iterations)
        elif exp_type == 'algo_compare':
            return self.run_algorithm_comparison(iterations)
        else:
            return self.generate_demo_data()
    
    def run_quick_wins_suite(self, iterations):
        """Run the quick wins experiment suite"""
        experiments = [
            {'name': 'Baseline', 'env_params': {}, 'algo_params': {'tau': 0.05}},
            {'name': 'Hazard Relocation', 'env_params': {'unsafe_cells': (6, 11)}, 'algo_params': {'tau': 0.05}},
            {'name': 'High Slip', 'env_params': {'slip': 0.2}, 'algo_params': {'tau': 0.05}},
            {'name': 'Tight Constraint', 'env_params': {}, 'algo_params': {'tau': 0.02}}
        ]
        
        results = []
        for exp in experiments:
            result = run_experiment(exp['env_params'], exp['algo_params'], exp['name'], iterations)
            results.append(result)
        
        return results
    
    def run_stress_test_suite(self, iterations):
        """Run stress test experiments"""
        # Simplified stress tests for dashboard
        experiments = [
            {'name': 'τ=0.05', 'env_params': {}, 'algo_params': {'tau': 0.05}},
            {'name': 'τ=0.02', 'env_params': {}, 'algo_params': {'tau': 0.02}},
            {'name': 'τ=0.01', 'env_params': {}, 'algo_params': {'tau': 0.01}},
            {'name': 'Slip=0.3', 'env_params': {'slip': 0.3}, 'algo_params': {'tau': 0.05}}
        ]
        
        results = []
        for exp in experiments:
            result = run_experiment(exp['env_params'], exp['algo_params'], exp['name'], iterations)
            results.append(result)
        
        return results
    
    def run_algorithm_comparison(self, iterations):
        """Run algorithm comparison (simplified for dashboard)"""
        # For now, just run different learning rates as a proxy for different algorithms
        experiments = [
            {'name': 'OGD (η=1.0)', 'env_params': {}, 'algo_params': {'tau': 0.05, 'eta_lam': 1.0}},
            {'name': 'Fast OGD (η=5.0)', 'env_params': {}, 'algo_params': {'tau': 0.05, 'eta_lam': 5.0}},
            {'name': 'Slow OGD (η=0.1)', 'env_params': {}, 'algo_params': {'tau': 0.05, 'eta_lam': 0.1}},
            {'name': 'High μ (η_μ=50)', 'env_params': {}, 'algo_params': {'tau': 0.05, 'eta_mu': 50.0}}
        ]
        
        results = []
        for exp in experiments:
            result = run_experiment(exp['env_params'], exp['algo_params'], exp['name'], iterations)
            results.append(result)
        
        return results
    
    def generate_demo_data(self):
        """Generate demo data for when experiments aren't available"""
        np.random.seed(42)
        results = []
        
        for i, name in enumerate(['Baseline', 'Hazard Relocation', 'High Slip', 'Tight Constraint']):
            # Generate synthetic learning curves
            iterations = np.arange(100, 3001, 100)
            f_values = np.exp(-iterations/1000) * (1 + 0.1*np.random.randn(len(iterations))) + 0.001
            unsafe_values = 0.05 + 0.02*np.exp(-iterations/800) * (1 + 0.1*np.random.randn(len(iterations)))
            
            if 'Tight' in name:
                unsafe_values *= 0.4  # Tighter constraint
            if 'Slip' in name:
                f_values *= 1.5  # Worse performance with slip
                unsafe_values *= 1.2
            
            result = {
                'name': name,
                'history': {
                    'f': f_values.tolist(),
                    'unsafe': unsafe_values.tolist(),
                    'iteration': iterations.tolist()
                },
                'runtime': 30 + 10*np.random.randn(),
                'final_metrics': {
                    'f_value': f_values[-1],
                    'unsafe_prob': unsafe_values[-1],
                    'constraint_violation': max(0, unsafe_values[-1] - 0.05),
                    'tau': 0.02 if 'Tight' in name else 0.05
                }
            }
            results.append(result)
        
        return results
    
    def create_convergence_plot(self, data):
        """Create convergence plot"""
        fig = go.Figure()
        
        for result in data:
            if 'history' in result and 'f' in result['history']:
                fig.add_trace(go.Scatter(
                    x=result['history']['iteration'],
                    y=result['history']['f'],
                    mode='lines',
                    name=result['name'],
                    line=dict(width=2)
                ))
        
        fig.update_layout(
            title="Objective Function Convergence",
            xaxis_title="Iteration",
            yaxis_title="f(d) = ||d - d_E||²",
            yaxis_type="log",
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def create_performance_comparison(self, data):
        """Create performance comparison bar chart"""
        names = [r['name'] for r in data]
        f_values = [r['final_metrics']['f_value'] for r in data]
        
        fig = go.Figure(data=[
            go.Bar(x=names, y=f_values, 
                  marker_color=px.colors.qualitative.Set3[:len(names)])
        ])
        
        fig.update_layout(
            title="Final Performance Comparison",
            xaxis_title="Experiment",
            yaxis_title="Final f(d)",
            yaxis_type="log",
            template='plotly_white'
        )
        
        return fig
    
    def create_tradeoff_plot(self, data):
        """Create trade-off scatter plot"""
        f_values = [r['final_metrics']['f_value'] for r in data]
        unsafe_values = [r['final_metrics']['unsafe_prob'] for r in data]
        names = [r['name'] for r in data]
        taus = [r['final_metrics']['tau'] for r in data]
        
        # Color based on constraint satisfaction
        colors = ['green' if unsafe <= tau else 'red' for unsafe, tau in zip(unsafe_values, taus)]
        
        fig = go.Figure(data=go.Scatter(
            x=f_values,
            y=unsafe_values,
            mode='markers+text',
            text=names,
            textposition="top center",
            marker=dict(size=12, color=colors, opacity=0.7),
            hovertemplate='<b>%{text}</b><br>f(d): %{x:.4f}<br>Unsafe: %{y:.4f}<extra></extra>'
        ))
        
        # Add constraint line
        if taus:
            fig.add_hline(y=taus[0], line_dash="dash", line_color="red", 
                         annotation_text=f"τ = {taus[0]}")
        
        fig.update_layout(
            title="Performance Trade-off: Imitation vs Safety",
            xaxis_title="Objective Value f(d)",
            yaxis_title="Unsafe Probability",
            xaxis_type="log",
            template='plotly_white'
        )
        
        return fig
    
    def create_empty_plot(self, message):
        """Create empty plot with message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(template='plotly_white')
        return fig
    
    def create_results_table(self, data):
        """Create results table"""
        table_data = []
        for result in data:
            metrics = result['final_metrics']
            table_data.append([
                result['name'],
                f"{metrics['f_value']:.4f}",
                f"{metrics['unsafe_prob']:.3f}",
                f"{metrics['constraint_violation']:.4f}",
                f"{result['runtime']:.1f}s"
            ])
        
        return html.Table([
            html.Thead([
                html.Tr([
                    html.Th("Experiment"),
                    html.Th("Final f(d)"),
                    html.Th("Unsafe Prob"),
                    html.Th("Violation"),
                    html.Th("Runtime")
                ])
            ]),
            html.Tbody([
                html.Tr([html.Td(cell) for cell in row]) for row in table_data
            ])
        ], style={'width': '100%', 'textAlign': 'center'})
    
    def create_insights_panel(self, data):
        """Create insights panel"""
        insights = []
        
        # Find best performer
        best = min(data, key=lambda x: x['final_metrics']['f_value'])
        insights.append(html.P(f"🏆 Best performer: {best['name']} (f={best['final_metrics']['f_value']:.4f})"))
        
        # Find constraint violations
        violations = [r for r in data if r['final_metrics']['constraint_violation'] > 0]
        if violations:
            insights.append(html.P(f"⚠️ {len(violations)} experiments violated constraints"))
        else:
            insights.append(html.P("✅ All constraints satisfied!"))
        
        # Runtime analysis
        avg_runtime = np.mean([r['runtime'] for r in data])
        insights.append(html.P(f"⏱️ Average runtime: {avg_runtime:.1f}s"))
        
        # Convergence analysis
        converged = []
        for result in data:
            if 'history' in result and 'f' in result['history']:
                f_vals = result['history']['f']
                if len(f_vals) > 10:
                    recent_std = np.std(f_vals[-10:])
                    if recent_std < 0.001:
                        converged.append(result['name'])
        
        if converged:
            insights.append(html.P(f"📈 Converged experiments: {', '.join(converged)}"))
        
        return html.Div(insights)
    
    def run(self):
        """Run the dashboard"""
        print(f"🚀 Starting Experiment Dashboard...")
        print(f"📊 Open http://localhost:{self.port} in your browser")
        print(f"🔬 Experiments available: {EXPERIMENTS_AVAILABLE}")
        
        self.app.run_server(debug=True, port=self.port, host='0.0.0.0')

def main():
    dashboard = ExperimentDashboard(port=8050)
    dashboard.run()

if __name__ == "__main__":
    main() 