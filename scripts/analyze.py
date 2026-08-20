"""Rebuild lightweight error, drift, association, intervention, and plots."""

import _bootstrap  # noqa: F401

from association_analysis import main as association_main
from churn_explainer import main as explainer_main
from research_analysis import main as research_main


if __name__ == "__main__":
    research_main()
    association_main()
    explainer_main()
