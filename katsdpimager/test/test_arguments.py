"""Tests for :mod:`katsdpimager.arguments`"""

import argparse

import astropy.units as u
from nose.tools import assert_equal

from .. import arguments


class TestUnparse:
    def setup(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--int', type=int)
        parser.add_argument('--int-default', type=int, default=3)
        parser.add_argument('--float', type=float)
        parser.add_argument('--float-default', type=float, default=2.5)
        parser.add_argument('--str', type=str)
        parser.add_argument('--str-default', type=str, default='')
        parser.add_argument('--positive', action='store_true')
        parser.add_argument('--no-negative', dest='negative', action='store_false')
        parser.add_argument('--quantity', type=u.Quantity)
        self.parser = parser

    def test_no_args(self):
        """Pass no arguments."""
        args = self.parser.parse_args([], namespace=arguments.SmartNamespace())
        assert_equal(arguments.unparse_args(args), [])

    def test_defaults(self):
        """Pass arguments with default values."""
        argv = ['--int-default=3', '--float-default=2.5', '--str-default=']
        args = self.parser.parse_args(argv, namespace=arguments.SmartNamespace())
        assert_equal(arguments.unparse_args(args), [])

    def test_all_args(self):
        """Pass all arguments."""
        argv = [
            '--int=1',
            '--int-default=2',
            '--float=3.1',
            '--float-default=4.5',
            '--str=hello',
            '--str-default=world',
            '--positive',
            '--no-negative',
            '--quantity=12.0 rad'
        ]
        args = self.parser.parse_args(argv, namespace=arguments.SmartNamespace())
        assert_equal(arguments.unparse_args(args), argv)

    def test_exclude(self):
        """Test the exclude argument."""
        self.parser.add_argument('positional', type=str)
        argv = ['positional', '--int=1', '--int-default=2']
        args = self.parser.parse_args(argv, namespace=arguments.SmartNamespace())
        unparsed = arguments.unparse_args(args, exclude=['positional', 'int'])
        assert_equal(unparsed, ['--int-default=2'])

    def test_arg_handler(self):
        """Test the `arg_handlers` option."""
        self.parser.add_argument('positional', type=str)
        argv = ['--int=1', 'positional_arg']
        args = self.parser.parse_args(argv, namespace=arguments.SmartNamespace())
        unparsed = arguments.unparse_args(
            args, arg_handlers={'positional': lambda name, value: [value]})
        assert_equal(unparsed, argv)

    def test_type_handler(self):
        """Test the type_handlers option."""
        self.parser.add_argument('--list', type=str, action='append')
        argv = ['--list=1', '--list=2']
        args = self.parser.parse_args(argv, namespace=arguments.SmartNamespace())
        unparsed = arguments.unparse_args(
            args, type_handlers={list: lambda name, value: [f'--{name}={x}' for x in value]})
        assert_equal(unparsed, argv)
