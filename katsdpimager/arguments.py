"""Utilities for processing and reconstructing command-line arguments."""

import argparse
from typing import List, Iterable, Set, Mapping, Callable, Any

import astropy.units as u


class SmartNamespace(argparse.Namespace):
    """Namespace that tracks whether arguments are default or not.

    There are a few requirements for this to work:
    - All arguments must have a default, even if it is ``None``. In other
      words, do not set the default to ``argparse.SUPPRESS``.
    - The default must have a meaningful equality comparison with the
      argument type.
    - It will break down if the default is a string but the argument type is
      not a string (in this case, argparse first sets it to the string value,
      then replaces it with the conversion of that string value).

    Access to the information is provided by :func:`argument_changed` rather
    than via methods to avoid potential conflicts with command-line option
    names.
    """

    def __init__(self, **kwargs) -> None:
        self._is_changed: Set[str] = set()
        super().__init__(**kwargs)

    def __setattr__(self, name: str, value: object) -> None:
        if not name.startswith('_') and name in self and getattr(self, name) != value:
            self._is_changed.add(name)
        super().__setattr__(name, value)


def argument_changed(namespace: SmartNamespace, name: str) -> bool:
    """True if the name has been set to more than one value.

    This typically indicates that it was first set to a default, then set
    to a different value on the command line.
    """
    return name in namespace._is_changed


def _normalize_name(name: str) -> str:
    return name.replace('_', '-')


def _bool_handler(name: str, value: object) -> List[str]:
    norm = _normalize_name(name)
    return [f'--{norm}'] if value else [f'--no-{norm}']


def _to_str_handler(name: str, value: object) -> List[str]:
    norm = _normalize_name(name)
    return [f'--{norm}={value}']


DEFAULT_TYPE_HANDLERS = {
    bool: _bool_handler,
    str: _to_str_handler,
    int: _to_str_handler,
    float: _to_str_handler,
    u.Quantity: _to_str_handler
}


def unparse_args(
        args: SmartNamespace,
        exclude: Iterable[str] = (),
        *,
        arg_handlers: Mapping[str, Callable[[str, Any], List[str]]] = {},
        type_handlers: Mapping[type, Callable[[str, Any], List[str]]] = {}) -> List[str]:
    """Reconstruct an equivalent command line from parsed arguments.

    Parameters
    ----------
    args
        Parsed arguments
    exclude
        Elements of `args` to skip. This should usually include at least the
        positional arguments.
    arg_handlers
        Special-case handling for particular arguments, mapping the name in `args`
        to a callable that takes the name and value and returns command-line
        arguments.
    type_handlers
        Like `arg_handlers`, but based on the type of the value rather than the
        destination name. There are standard type handlers for int, float, str,
        :class:`~astropy.units.Quantity` and bool. This is only used if the name
        is not in `arg_handlers`.

    Raises
    ------
    TypeError
        If an argument has no handler.
    """
    skip = set(exclude)
    out: List[str] = []
    type_handlers = {**DEFAULT_TYPE_HANDLERS, **type_handlers}
    for name, value in vars(args).items():
        if name not in skip and argument_changed(args, name):
            try:
                handler = arg_handlers[name]
            except KeyError:
                arg_type = type(value)
                try:
                    handler = type_handlers[arg_type]
                except KeyError:
                    raise TypeError(f'No handler for {name} (type {type})') from None
            out.extend(handler(name, value))
    return out
