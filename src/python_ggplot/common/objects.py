from copy import deepcopy


class Freezable:
    _frozen: bool = False

    def __post_init__(self):
        object.__setattr__(self, "_frozen", True)

    def freeze(self):
        object.__setattr__(self, "_frozen", True)

    def __setattr__(self, key, value):  # type: ignore
        if getattr(self, "_frozen", False) and key != "_frozen":
            raise AttributeError(f"Cannot modify '{key}' - instance is frozen.")
        super().__setattr__(key, value)  # type: ignore

    def update_with_copy(self, **kwargs):
        new_obj = deepcopy(self)
        new_obj._frozen = False
        for key, value in kwargs.items():
            setattr(new_obj, key, value)
        new_obj._frozen = True
        return new_obj
