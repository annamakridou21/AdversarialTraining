"""Compatibility entry point for measured model comparisons."""

if __package__ in (None, ''):
    from create_comparison_plots import main, parser
else:
    from .create_comparison_plots import main, parser

if __name__ == '__main__':
    main(parser().parse_args())
