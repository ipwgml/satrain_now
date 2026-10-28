"""
satrain_now.cli
===============

The 'satrain_now' command line interface.
"""
import click

from satrain_now.obs import goes

@click.group()
@click.version_option(version="0.0.1")
def main():
    """
    Command line interface for the 'satrain_now' package.
    """
    pass

main.add_command(goes.cli, name="extract_goes_obs")
