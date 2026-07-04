import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging
import os

# Set global aesthetics for capstone quality
sns.set_theme(style="whitegrid", context="talk", palette="deep")

def _save_plot(filename: str):
    """Helper function to save plots to the reports directory."""
    os.makedirs('reports/figures', exist_ok=True)
    path = os.path.join('reports/figures', filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    logging.info(f"Saved plot to {path}")
    plt.close()

def plot_line_chart(df: pd.DataFrame, column: str, filename='plot_1_line_plot.png'):
    """Generate a line plot of a numeric variable sorted by row index."""
    plt.figure(figsize=(12, 6))
    plt.plot(df[column].sort_values().reset_index(drop=True), color='coral', linewidth=2)
    plt.title(f'Line Plot of {column} (Sorted by Value)')
    plt.xlabel('Sorted Row Index')
    plt.ylabel(column)
    _save_plot(filename)

def plot_bar_chart(df: pd.DataFrame, cat_col: str, num_col: str, filename='plot_2_bar_chart.png'):
    """Generate a bar chart comparing the mean of a numeric column across categories."""
    plt.figure(figsize=(10, 6))
    agg_df = df.groupby(cat_col)[num_col].mean().reset_index()
    sns.barplot(x=cat_col, y=num_col, data=agg_df, palette='viridis')
    plt.title(f'Mean {num_col} by {cat_col}')
    plt.xlabel(cat_col)
    plt.ylabel(f'Mean {num_col}')
    _save_plot(filename)

def plot_histogram(df: pd.DataFrame, column: str, bins=20, filename='plot_3_histogram.png'):
    """Generate a histogram of a numeric column."""
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column].dropna(), bins=bins, kde=True, color='teal', edgecolor='black')
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    _save_plot(filename)

def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str, filename='plot_4_scatter.png'):
    """Generate a scatter plot between two numeric columns."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=x_col, y=y_col, data=df, alpha=0.6, color='darkblue')
    plt.title(f'Scatter Plot: {x_col} vs {y_col}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    _save_plot(filename)

def plot_boxplot(df: pd.DataFrame, cat_col: str, num_col: str, filename='plot_5_boxplot.png'):
    """Generate a box plot of a numeric column split by a categorical column."""
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=cat_col, y=num_col, data=df, palette='Set2')
    plt.title(f'Box Plot of {num_col} by {cat_col}')
    plt.xlabel(cat_col)
    plt.ylabel(num_col)
    _save_plot(filename)

def plot_correlation_heatmap(corr_matrix: pd.DataFrame, filename='plot_6_heatmap.png'):
    """Generate a correlation heatmap."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', fmt=".2f", vmin=-1, vmax=1, square=True)
    plt.title('Correlation Heatmap of Numeric Features', pad=20)
    _save_plot(filename)
