from pathlib import Path

import streamlit.components.v1 as components

# Points to build folder after frontend build
_build_dir = Path(__file__).parent / "frontend" / "build"

# Declare the Streamlit component using the build directory
_st_canvas = components.declare_component("st_canvas", path=str(_build_dir))


def st_canvas(**kwargs):
    """
    Wrapper function to invoke the custom Streamlit canvas component.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments to be passed directly to the underlying
        Streamlit component.

    Returns
    -------
    Any
        The value returned by the Streamlit component invocation.
    """
    return _st_canvas(**kwargs)
