"""
Dashboard UI Components package for Customer Churn Decision Engine.
"""
from app.dashboard.components.comparison import render_comparison_component
from app.dashboard.components.diagnostics import render_diagnostics_component
from app.dashboard.components.priority_matrix import render_priority_matrix
from app.dashboard.components.simulator import render_simulator_component
from app.dashboard.components.uploader import render_uploader_component
__all__=[
    "render_comparison_component",
    "render_diagnostics_component",
    "render_priority_matrix",
    "render_simulator_component",
    "render_uploader_component",
]
