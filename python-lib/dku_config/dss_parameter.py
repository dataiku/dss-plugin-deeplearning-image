from .custom_check import CustomCheck, CustomCheckError
from typing import Any, List

import logging
logger = logging.getLogger(__name__)


class DSSParameterError(Exception):
    """Exception raised when at least one CustomCheck fails.
    """
    pass


class DSSParameter:
    """Object related to one parameter. It is mainly used for checks to run in backend for custom forms.

    Attributes:
        name(str): Name of the parameter
        value(Any): Value of the parameter
        checks(list[dict], optional): Checks to run on provided value
        required(bool, optional): Whether the value can be None
    """
    def __init__(self, name: str, value: Any, checks: List[dict] = None, required: bool = False, cast_to: type = None, default: Any = None):
        """Initialization method for the DSSParameter class

        Args:
            name(str): Name of the parameter
            value(Any): Value of the parameter
            checks(list[dict], optional): Checks to run on provided value
            required(bool, optional): Whether the value can be None
        """
        if checks is None:
            checks = []
        self.name = name
        self.value = value if value is not None else default
        value_exists = self.run_checks([CustomCheck(type='exists')], raise_error=required)
        self.cast_to = cast_to
        self.checks = [CustomCheck(**check) for check in checks]
        if value_exists:
            if self.cast_to:
                self.run_checks([CustomCheck(type='is_castable', op=cast_to)])
                self.cast_value()
            self.run_checks(self.checks)

    def cast_value(self):
        self.value = self.cast_to(self.value) if self.cast_to else self.value

    def run_checks(self, checks, raise_error=True):
        """Runs all checks provided for this parameter
        """
        for check in checks:
            try:
                check.run(self.value)
            except CustomCheckError as err:
                if raise_error:
                    self.handle_failure(err)
                return False
        return True

    def handle_failure(self, error: CustomCheckError):
        """Is called when at least one test fails. It will raise an Exception with understandable text
        Args:
            error(CustomCheckError: Errors met when running checks
        Raises:
            DSSParameterError: Raises if at least on check fails
        """
        raise DSSParameterError(self.format_failure_message(error))

    def format_failure_message(self, error: CustomCheckError) -> str:
        """Format failure text
        Args:
            error(CustomCheckError: Error met when running check
        Returns:
            str: Formatted error message
        """
        return """
        Error for parameter \"{name}\" :
        {error}
        Please check your settings and fix the error.
        """.format(
            name=self.name,
            error=error
        )

    def handle_success(self):
        """Called if all checks are successful. Prints a success message
        """
        self.print_success_message()

    def print_success_message(self):
        """Formats the success message
        """
        logger.debug('All checks have been successfully done for {}.'.format(self.name))

    def __repr__(self):
        return str({"value": self.value})

    def __str__(self):
        return str({"value": self.value})
