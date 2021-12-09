import collections

__all__ = ['Model']

Model = collections.namedtuple("Model", ('mesh', 'template', 'bcs', 'analysis_strategy'))

