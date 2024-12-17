import sys
import argparse
# from error_handler import ErrorHandler

class CommandLineParser:
	def __init__(self, argument_config):
		# self.error_handler = ErrorHandler()
		self.parser = argparse.ArgumentParser(
			formatter_class=argparse.RawTextHelpFormatter
		)
		self.add_arguments(argument_config)


	def add_arguments(self, argument_config):
		for arg in argument_config:
			self.parser.add_argument(
				arg['name'], 
				type=arg['type'],
				default=arg['default'], 
				choices= arg['choices'],
				help=arg['help'])
	

	def parse_arguments(self):
		if len(sys.argv) > 2:
			self.parser.print_help(sys.stderr)
			sys.exit(1)
		try:
			args = self.parser.parse_args()
			args_dict = vars(args)

			for key, value in args_dict.items():
				arg_name = key
				arg_value = value.lower()

			arg = arg_value == 'true' #check if this is true or something else rubbish?
			
		except ValueError as e:
			print('Wrong amount of arguments, expected 1. Or the argument was incorrect. True or False is expected.')
			# self.error_handler.log_message(str(e))

		return arg
