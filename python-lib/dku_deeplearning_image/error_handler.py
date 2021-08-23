from dataikuapi.utils import DataikuException
import re


class DataikuPluginException(DataikuException):
    pass


ERROR_MAPPING = [
    {
        "type": ValueError,
        "err_pattern": "Shapes (.*) and (.*) are incompatible",
        "error_msg": "One of the category has probably too few samples. Check the input data."
    }
]


def build_full_error_msg(error_type, error_msg, original_err):
    return f"{error_msg}\nOriginal error:\n{error_type} {original_err})"


def raise_plugin_error(original_err):
    for e in ERROR_MAPPING:
        if isinstance(original_err, e["type"]) and re.match(e["err_pattern"], str(original_err)):
            full_error_msg = e["error_msg"]
        else:
            full_error_msg = "Unknown error"
        raise DataikuPluginException(build_full_error_msg(str(e["type"]), full_error_msg, original_err))
