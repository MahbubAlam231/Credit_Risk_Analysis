"""
#!/usr/bin/python3
Author      : Mahbub Alam
File        : utils.py
Created     : 2025-08
Description : Helper functions for Credit Risk Analysis # {{{

# }}}
"""

import textwrap

def wrap_labels_(ax, width=15, rotation=0, ha="right", pad=5):# {{{
    """Wrap long labels on x-axis"""
    ticks = ax.get_xticks()
    labels = [label.get_text() for label in ax.get_xticklabels()]

    # Wrap long labels
    wrapped_labels = ["\n".join(textwrap.wrap(l, width=width)) for l in labels]

    # Set ticks + labels explicitly
    ax.set_xticks(ticks)
    if ha == "at_tick":
        ax.set_xticklabels(wrapped_labels, rotation=rotation)
        for label in ax.get_xticklabels():
            label.set_x(label.get_position()[0])
    else:
        ax.set_xticklabels(wrapped_labels, rotation=rotation, ha=ha)

    # Adjust padding
    ax.tick_params(axis="x", pad=pad)

# }}}


