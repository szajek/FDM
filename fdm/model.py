import collections

__all__ = ['Model']

Model = collections.namedtuple("Model", ('template', 'mesh', 'analysis_strategy'))

