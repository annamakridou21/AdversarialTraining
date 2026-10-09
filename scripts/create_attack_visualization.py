"""Compatibility entry point for PGD image visualization."""

if __package__ in (None, ''):
    from visualize_attacks import main, parser
else:
    from .visualize_attacks import main, parser

if __name__ == '__main__':
    main(parser().parse_args())
