class Immutable(type):
    def __new__(msc, name, bases, nmspc):
        new_class = type(name, bases, nmspc)

        original_initialization = new_class.__init__

        def set_attribute(self, key, value):
            raise AttributeError("Attributes are immutable")

        def wrapper(self, *args, **kwargs):
            new_class.__setattr__ = object.__setattr__
            original_initialization(self, *args, **kwargs)
            new_class.__setattr__ = set_attribute

        new_class.__init__ = wrapper

        return new_class


def create_cached_factory(factory, id_builder):
    cache = {}

    def create(point):
        _id = id_builder(point)
        try:
            return cache[_id]
        except KeyError:
            return cache.setdefault(_id, factory(point))

    return create
