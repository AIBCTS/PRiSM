import re
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt


@contextmanager
def capture_and_save_figures(
    base_dir=None,
    model_prefix=None,
    formats=['svg', 'png', 'pdf'],
    dpi=300,
    include_timestamp=True,
    descriptive_names=True,
    close_after_save=False,
    prefix=None,
    auto_increment=True,
    namespace=None,
):
    """
    Context manager to automatically capture and save figures generated during execution.

    This context manager monitors matplotlib figure creation and automatically saves
    any new figures created within the context block. It works with any function
    that generates matplotlib figures, even if the figure object is not returned.

    Parameters
    ----------
    base_dir : str or Path, optional
        Base directory to save figures. Defaults to MODELS_DIR / 'figures'
    model_prefix : str, optional
        Model prefix for directory structure. Auto-detected from namespace if None
    formats : list of str, default=['svg', 'png', 'pdf']
        File formats to save. Options: 'svg', 'png', 'pdf', 'jpg'
    dpi : int, default=300
        Resolution for raster formats (png, jpg)
    include_timestamp : bool, default=True
        Whether to include timestamp in filenames
    descriptive_names : bool, default=True
        Whether to use descriptive names from figure titles
    close_after_save : bool, default=False
        Whether to close figures after saving
    prefix : str, optional
        Custom prefix for saved filenames
    auto_increment : bool, default=True
        Whether to auto-increment figure numbers for multiple figures
    namespace : dict, optional
        Namespace to search for variables (e.g., globals() from notebook)

    Returns
    -------
    list
        List of saved file paths

    Examples
    --------
    >>> # Save all figures generated in a cell
    >>> with capture_and_save_figures(prefix="model_evaluation"):
    ...     evaluate_model_performance(y_true, y_pred)
    ...     compare_model_performance(y_true, y_pred1, y_pred2)

    >>> # Use in a loop to save figures with different prefixes
    >>> for model_name in ['xgb', 'logreg', 'mlp']:
    ...     with capture_and_save_figures(prefix=f"{model_name}_results"):
    ...         plot_model_results(model_name)
    """
    # Use provided namespace or fall back to globals
    if namespace is None:
        namespace = globals()

    # Store initial figure numbers
    initial_fig_nums = set(plt.get_fignums())

    # Auto-detect model_prefix if not provided
    if model_prefix is None:
        if 'model_prefix' in namespace:
            model_prefix = namespace['model_prefix']
        else:
            model_prefix = 'unknown_model'

    # Set base directory
    if base_dir is None:
        if 'MODELS_DIR' in namespace:
            base_dir = namespace['MODELS_DIR'] / 'figures' / model_prefix
        else:
            base_dir = Path('./figures') / model_prefix
    else:
        base_dir = Path(base_dir) / model_prefix

    # Create directory
    base_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if include_timestamp else ""

    saved_files = []

    try:
        # Yield control to the user's code
        yield saved_files

        # After user code executes, find new figures
        current_fig_nums = set(plt.get_fignums())
        new_fig_nums = current_fig_nums - initial_fig_nums

        if not new_fig_nums:
            print("No new figures found to save.")
            return

        print(f"Found {len(new_fig_nums)} new figure(s) to save...")

        # Sort figure numbers for consistent ordering
        new_fig_nums = sorted(new_fig_nums)

        for i, fig_num in enumerate(new_fig_nums):
            fig = plt.figure(fig_num)

            # Generate base filename
            if prefix:
                if auto_increment and len(new_fig_nums) > 1:
                    base_name = f"{prefix}_{i+1:02d}"
                else:
                    base_name = prefix
            elif descriptive_names and hasattr(fig, '_suptitle') and fig._suptitle:
                # Use figure title if available
                title = fig._suptitle.get_text()
                clean_title = re.sub(r'[^\w\s-]', '', title)
                clean_title = re.sub(r'[-\s]+', '_', clean_title)
                base_name = f"{clean_title}_fig{fig_num}"
            else:
                # Check axes titles
                axes_titles = []
                for ax in fig.get_axes():
                    if ax.get_title():
                        axes_titles.append(ax.get_title())

                if axes_titles:
                    # Use first axes title
                    title = axes_titles[0]
                    clean_title = re.sub(r'[^\w\s-]', '', title)
                    clean_title = re.sub(r'[-\s]+', '_', clean_title)
                    base_name = f"{clean_title}_fig{fig_num}"
                else:
                    # Fallback to generic name
                    if auto_increment and len(new_fig_nums) > 1:
                        base_name = f"figure_{i+1:02d}"
                    else:
                        base_name = f"figure_{fig_num}"

            # Add timestamp if requested
            if include_timestamp:
                base_name = f"{base_name}_{timestamp}"

            # Save in each requested format
            for fmt in formats:
                filename = f"{base_name}.{fmt}"
                filepath = base_dir / filename

                try:
                    # Set DPI for raster formats
                    save_dpi = dpi if fmt.lower() in ['png', 'jpg', 'jpeg'] else None

                    fig.savefig(
                        filepath,
                        format=fmt,
                        dpi=save_dpi,
                        bbox_inches='tight',
                        facecolor='white',
                        edgecolor='none',
                    )
                    saved_files.append(str(filepath))
                    print(f"  Saved: {filename}")

                except Exception as e:
                    print(f"  Error saving {filename}: {e}")

            # Close figure if requested
            if close_after_save:
                plt.close(fig)

        print(f"\nSaved {len(saved_files)} file(s) to: {base_dir}")

    except Exception as e:
        print(f"Error in figure capture context: {e}")
        raise


class FigureSaver:
    """
    A reusable figure saver class that can be configured once and used multiple times.

    This class provides a more object-oriented approach to figure saving, allowing
    you to set up configuration once and reuse it throughout a notebook.

    Parameters
    ----------
    base_dir : str or Path, optional
        Base directory to save figures
    model_prefix : str, optional
        Model prefix for directory structure
    formats : list of str, default=['svg', 'png', 'pdf']
        File formats to save
    dpi : int, default=300
        Resolution for raster formats
    include_timestamp : bool, default=True
        Whether to include timestamp in filenames
    descriptive_names : bool, default=True
        Whether to use descriptive names from figure titles
    close_after_save : bool, default=False
        Whether to close figures after saving

    Examples
    --------
    >>> # Set up figure saver once
    >>> saver = FigureSaver(model_prefix='xgb', formats=['png', 'pdf'])

    >>> # Use throughout notebook
    >>> with saver.capture(prefix="training_loss"):
    ...     plot_training_loss()

    >>> with saver.capture(prefix="model_performance"):
    ...     evaluate_model_performance(y_true, y_pred)
    """

    def __init__(
        self,
        base_dir=None,
        model_prefix=None,
        formats=['svg', 'png', 'pdf'],
        dpi=300,
        include_timestamp=True,
        descriptive_names=True,
        close_after_save=False,
        namespace=None,
    ):
        self.base_dir = base_dir
        self.model_prefix = model_prefix
        self.formats = formats
        self.dpi = dpi
        self.include_timestamp = include_timestamp
        self.descriptive_names = descriptive_names
        self.close_after_save = close_after_save
        self.namespace = namespace

    def capture(self, prefix=None, auto_increment=True, **kwargs):
        """
        Create a context manager to capture figures with this saver's configuration.

        Parameters
        ----------
        prefix : str, optional
            Custom prefix for saved filenames
        auto_increment : bool, default=True
            Whether to auto-increment figure numbers for multiple figures
        **kwargs
            Override any of the saver's default parameters

        Returns
        -------
        context manager
            Context manager for capturing figures
        """
        # Merge kwargs with instance defaults
        config = {
            'base_dir': kwargs.get('base_dir', self.base_dir),
            'model_prefix': kwargs.get('model_prefix', self.model_prefix),
            'formats': kwargs.get('formats', self.formats),
            'dpi': kwargs.get('dpi', self.dpi),
            'include_timestamp': kwargs.get('include_timestamp', self.include_timestamp),
            'descriptive_names': kwargs.get('descriptive_names', self.descriptive_names),
            'close_after_save': kwargs.get('close_after_save', self.close_after_save),
            'prefix': prefix,
            'auto_increment': auto_increment,
            'namespace': kwargs.get('namespace', self.namespace),
        }

        return capture_and_save_figures(**config)


def save_all_notebook_figures(
    base_dir=None,
    model_prefix=None,
    formats=['svg', 'png', 'pdf'],
    dpi=300,
    include_timestamp=True,
    descriptive_names=True,
    close_after_save=False,
    namespace=None,
):
    """
    Save all currently open matplotlib figures from the notebook.

    Parameters
    ----------
    base_dir : str or Path, optional
        Base directory to save figures. Defaults to MODELS_DIR / 'figures'
    model_prefix : str, optional
        Model prefix for directory structure. Auto-detected from namespace if None
    formats : list of str, default=['svg', 'png', 'pdf']
        File formats to save. Options: 'svg', 'png', 'pdf', 'jpg'
    dpi : int, default=300
        Resolution for raster formats (png, jpg)
    include_timestamp : bool, default=True
        Whether to include timestamp in filenames
    descriptive_names : bool, default=True
        Whether to use descriptive names from figure titles
    close_after_save : bool, default=False
        Whether to close figures after saving
    namespace : dict, optional
        Namespace to search for variables (e.g., globals() from notebook)

    Returns
    -------
    list
        List of saved file paths
    """
    import re
    from datetime import datetime
    from pathlib import Path

    import matplotlib.pyplot as plt

    # Use provided namespace or fall back to globals
    if namespace is None:
        namespace = globals()

    # Auto-detect model_prefix if not provided
    if model_prefix is None:
        if 'model_prefix' in namespace:
            model_prefix = namespace['model_prefix']
        else:
            model_prefix = 'unknown_model'

    # Set base directory
    if base_dir is None:
        if 'MODELS_DIR' in namespace:
            base_dir = namespace['MODELS_DIR'] / 'figures' / model_prefix
        else:
            base_dir = Path('./figures') / model_prefix
    else:
        base_dir = Path(base_dir) / model_prefix

    # Create directory
    base_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if include_timestamp else ""

    # Get all open figures
    fig_nums = plt.get_fignums()
    saved_files = []

    if not fig_nums:
        print("No open figures found to save.")
        return saved_files

    print(f"Found {len(fig_nums)} figure(s) to save...")

    for fig_num in fig_nums:
        fig = plt.figure(fig_num)

        # Generate filename
        if descriptive_names and hasattr(fig, '_suptitle') and fig._suptitle:
            # Use figure title if available
            title = fig._suptitle.get_text()
            # Clean title for filename
            clean_title = re.sub(r'[^\w\s-]', '', title)
            clean_title = re.sub(r'[-\s]+', '_', clean_title)
            base_name = f"{clean_title}_fig{fig_num}"
        else:
            # Check axes titles
            axes_titles = []
            for ax in fig.get_axes():
                if ax.get_title():
                    axes_titles.append(ax.get_title())

            if axes_titles:
                # Use first axes title
                title = axes_titles[0]
                clean_title = re.sub(r'[^\w\s-]', '', title)
                clean_title = re.sub(r'[-\s]+', '_', clean_title)
                base_name = f"{clean_title}_fig{fig_num}"
            else:
                # Fallback to generic name
                base_name = f"figure_{fig_num}"

        # Add timestamp if requested
        if include_timestamp:
            base_name = f"{base_name}_{timestamp}"

        # Save in each requested format
        for fmt in formats:
            filename = f"{base_name}.{fmt}"
            filepath = base_dir / filename

            try:
                # Set DPI for raster formats
                save_dpi = dpi if fmt.lower() in ['png', 'jpg', 'jpeg'] else None

                fig.savefig(
                    filepath,
                    format=fmt,
                    dpi=save_dpi,
                    bbox_inches='tight',
                    facecolor='white',
                    edgecolor='none',
                )
                saved_files.append(str(filepath))
                print(f"  Saved: {filename}")

            except Exception as e:
                print(f"  Error saving {filename}: {e}")

        # Close figure if requested
        if close_after_save:
            plt.close(fig)

    print(f"\nSaved {len(saved_files)} file(s) to: {base_dir}")
    return saved_files


def save_specific_figures(figure_vars=None, namespace=None, **kwargs):
    """
    Save specific figures that were assigned to variables in the notebook.

    Parameters
    ----------
    figure_vars : list of str, optional
        List of variable names containing figure objects.
        If None, will search for common figure variable patterns.
    namespace : dict, optional
        Namespace to search for variables (e.g., globals() from notebook)
    **kwargs
        Additional arguments passed to save_all_notebook_figures

    Returns
    -------
    list
        List of saved file paths
    """
    from pathlib import Path

    # Use provided namespace or fall back to globals
    if namespace is None:
        namespace = globals()

    if figure_vars is None:
        # Auto-detect figure variables from namespace
        figure_vars = []
        for var_name, var_value in namespace.items():
            if (
                var_name.startswith('fig')
                or 'figure' in var_name.lower()
                or var_name.startswith('nomogram')
                or var_name.endswith('_fig')
            ):
                if hasattr(var_value, 'savefig'):  # Check if it's a figure object
                    figure_vars.append(var_name)

    saved_files = []

    # Get parameters from save_all_notebook_figures defaults
    base_dir = kwargs.get('base_dir')
    model_prefix = kwargs.get('model_prefix')
    formats = kwargs.get('formats', ['svg', 'png', 'pdf'])
    dpi = kwargs.get('dpi', 300)
    include_timestamp = kwargs.get('include_timestamp', True)

    # Auto-detect model_prefix if not provided
    if model_prefix is None:
        if 'model_prefix' in namespace:
            model_prefix = namespace['model_prefix']
        else:
            model_prefix = 'unknown_model'

    # Set base directory
    if base_dir is None:
        if 'MODELS_DIR' in namespace:
            base_dir = namespace['MODELS_DIR'] / 'figures' / model_prefix
        else:
            base_dir = Path('./figures') / model_prefix
    else:
        base_dir = Path(base_dir) / model_prefix

    base_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if include_timestamp else ""

    print(f"Found {len(figure_vars)} figure variable(s): {figure_vars}")

    for var_name in figure_vars:
        if var_name in namespace:
            fig = namespace[var_name]
            if hasattr(fig, 'savefig'):
                base_name = var_name
                if include_timestamp:
                    base_name = f"{base_name}_{timestamp}"

                for fmt in formats:
                    filename = f"{base_name}.{fmt}"
                    filepath = base_dir / filename

                    try:
                        save_dpi = dpi if fmt.lower() in ['png', 'jpg', 'jpeg'] else None
                        fig.savefig(
                            filepath,
                            format=fmt,
                            dpi=save_dpi,
                            bbox_inches='tight',
                            facecolor='white',
                            edgecolor='none',
                        )
                        saved_files.append(str(filepath))
                        print(f"  Saved: {filename}")
                    except Exception as e:
                        print(f"  Error saving {filename}: {e}")

    return saved_files


def save_results_figures(namespace=None, **kwargs):
    """
    Save figures from evaluation and comparison results stored in variables.

    This function specifically looks for results dictionaries that contain 'figure' keys,
    which are returned by evaluate_model_performance() and compare_model_performance().

    Parameters
    ----------
    namespace : dict, optional
        Namespace to search for variables (e.g., globals() from notebook)
    **kwargs
        Additional arguments passed to save functions

    Returns
    -------
    list
        List of saved file paths
    """
    from datetime import datetime
    from pathlib import Path

    # Use provided namespace or fall back to globals
    if namespace is None:
        namespace = globals()

    # Get parameters
    base_dir = kwargs.get('base_dir')
    model_prefix = kwargs.get('model_prefix')
    formats = kwargs.get('formats', ['svg', 'png', 'pdf'])
    dpi = kwargs.get('dpi', 300)
    include_timestamp = kwargs.get('include_timestamp', True)

    # Auto-detect model_prefix if not provided
    if model_prefix is None:
        if 'model_prefix' in namespace:
            model_prefix = namespace['model_prefix']
        else:
            model_prefix = 'unknown_model'

    # Set base directory
    if base_dir is None:
        if 'MODELS_DIR' in namespace:
            base_dir = namespace['MODELS_DIR'] / 'figures' / model_prefix
        else:
            base_dir = Path('./figures') / model_prefix
    else:
        base_dir = Path(base_dir) / model_prefix

    base_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if include_timestamp else ""

    saved_files = []
    results_figures = []

    # Search for results variables that contain figures
    for var_name, var_value in namespace.items():
        if (
            var_name.startswith('results_')
            or var_name.endswith('_results')
            or 'results' in var_name.lower()
        ):
            if isinstance(var_value, dict) and 'figure' in var_value:
                results_figures.append((var_name, var_value['figure']))

    print(
        f"Found {len(results_figures)} results figure(s): {[name for name, _ in results_figures]}"
    )

    for var_name, fig in results_figures:
        if hasattr(fig, 'savefig'):
            base_name = f"{var_name}_results"
            if include_timestamp:
                base_name = f"{base_name}_{timestamp}"

            for fmt in formats:
                filename = f"{base_name}.{fmt}"
                filepath = base_dir / filename

                try:
                    save_dpi = dpi if fmt.lower() in ['png', 'jpg', 'jpeg'] else None
                    fig.savefig(
                        filepath,
                        format=fmt,
                        dpi=save_dpi,
                        bbox_inches='tight',
                        facecolor='white',
                        edgecolor='none',
                    )
                    saved_files.append(str(filepath))
                    print(f"  Saved: {filename}")
                except Exception as e:
                    print(f"  Error saving {filename}: {e}")

    return saved_files


# Example usage function that can be called at the end of notebook
def save_all_figures(namespace=None):
    """
    Comprehensive figure saving that combines all methods.
    Call this function at the end of your notebook.

    Parameters
    ----------
    namespace : dict, optional
        Namespace to search for variables (e.g., globals() from notebook).
        If None, will use the calling module's globals().

    Returns
    -------
    list
        List of all saved file paths
    """
    # If no namespace provided, try to get the caller's globals
    if namespace is None:
        import inspect

        frame = inspect.currentframe()
        try:
            # Get the caller's frame (the notebook cell)
            caller_frame = frame.f_back
            if caller_frame:
                namespace = caller_frame.f_globals
            else:
                namespace = globals()
        finally:
            del frame

    print("=" * 60)
    print("SAVING ALL FIGURES FROM NOTEBOOK")
    print("=" * 60)

    # Method 1: Save all currently open matplotlib figures
    print("\n1. Saving all open matplotlib figures...")
    open_figs = save_all_notebook_figures(
        formats=['svg', 'png', 'pdf'],
        dpi=300,
        descriptive_names=True,
        close_after_save=False,
        namespace=namespace,
    )

    # Method 2: Save specific figure variables
    print("\n2. Saving specific figure variables...")
    var_figs = save_specific_figures(formats=['svg', 'png', 'pdf'], dpi=300, namespace=namespace)

    # Method 3: Save results figures from evaluation/comparison functions
    print("\n3. Saving results figures from evaluation/comparison functions...")
    results_figs = save_results_figures(
        formats=['svg', 'png', 'pdf'], dpi=300, namespace=namespace
    )

    # Summary
    all_saved = set(open_figs + var_figs + results_figs)
    print(f"\n{'='*60}")
    print(f"SUMMARY: Saved {len(all_saved)} unique figure files")
    print(f"{'='*60}")

    return list(all_saved)
