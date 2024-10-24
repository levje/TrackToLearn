import logging

root_logger = logging.getLogger('TTL')

def add_logging_args(parser):
    parser.add_argument('--log_file', type=str, default=None,
                        help='File to log to. If not set, logs to stdout.')
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Log level. One of: DEBUG, INFO, WARNING, ERROR, CRITICAL.')
    return parser

def setup_logging(args):
    # Set up logging configuration for all the loggers in the project
    # I don't want the loggers from other libraries to be affected by this
    # configuration.

    if args.log_file is not None:
        logging.basicConfig(filename=args.log_file)

    root_logger.setLevel(args.log_level)

def get_logger(name):
    return root_logger.getChild(name)
