import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import os

def setup_plot_directory(base_dir = "plots", subdirectory = None):
    """
    Create and return path to plot directory.
    
    Args:
        base_dir (str): Base directory for plots
        subdirectory (str, optional): Subdirectory name. If None, uses current date and time
    
    Returns:
        Path: Directory path where plots will be saved
    """
    if subdirectory is None:
        # Use current date and time as subdirectory
        subdirectory = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    plot_dir = Path(base_dir) / subdirectory
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir

def save_plot(plt, filename, plot_dir, dpi = 300, bbox_inches = "tight"):
    """
    Save plot to specified directory.
    
    Args:
        plt: Matplotlib pyplot object
        filename (str): Name of the plot file
        plot_dir (Path): Directory to save the plot
        dpi (int): DPI for the saved image
        bbox_inches (str): bbox_inches parameter for saving
    """
    filepath = plot_dir / filename
    if hasattr(plt, 'savefig'):
        plt.savefig(filepath, dpi = dpi, bbox_inches = bbox_inches)
    else:
        plt.figure.savefig(filepath, dpi = dpi, bbox_inches = bbox_inches)
    
    # Only clear and close using plt
    #plt.clf()

def create_timestamp_suffix():
    """Create a timestamp suffix for unique filenames."""
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')


def save_plot_with_timestamp(plt, filename, plot_dir, dpi = 300, bbox_inches = "tight"):
    """Save plot with timestamp in filename."""
    name, ext = os.path.splitext(filename)
    timestamped_filename = f"{name}_{create_timestamp_suffix()}{ext}"
    save_plot(plt, timestamped_filename, plot_dir, dpi, bbox_inches)


def save_analysis_to_file(content, filename = "analysis_output.txt", base_dir = "analysis_outputs"):
    """
    Save analysis output to a text file.
    
    Args:
        content (str): The content to save
        filename (str): Name of the output file
        base_dir (str): Directory to save the analysis file
    """
    
    output_dir = Path(base_dir)
    output_dir.mkdir(parents = True, exist_ok = True)
    
    name, ext = os.path.splitext(filename)
    timestamped_filename = f"{name}_{create_timestamp_suffix()}{ext}"
    
    # Save the content
    filepath = output_dir / timestamped_filename
    with open(filepath, "w") as f:
        f.write(content)