"""Evolution package.

Keep package import side effects minimal so lightweight commands such as
`review-external --help` do not require every optional runtime dependency.
"""

__all__ = [
    "external_evolution",
    "gan_evolution",
    "genetic_evolution",
    "mutation_operators",
    "reflective_evolution",
]
