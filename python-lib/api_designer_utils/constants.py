import os

PYTHON_LIB_DIR = 'python-lib'
PLUGIN_LIB_PATH = os.path.join('plugins', 'installed', 'deeplearning-image-v2', 'python-lib')
API_DESIGNER_UTILS_DIR = os.path.join(PLUGIN_LIB_PATH, 'api_designer_utils')
SPEC_PATH = os.path.join(PLUGIN_LIB_PATH, 'api_designer_utils', 'requirements.txt')
TEST_IMG_DIR = os.path.join(PLUGIN_LIB_PATH, 'api_designer_utils', 'test_images')
TEMPLATE_FILENAME = 'api_service_template.pyt'
TEMPLATE_PATH = os.path.join(API_DESIGNER_UTILS_DIR, TEMPLATE_FILENAME)
TEST_QUERIES = [{
    "name": "Score lion image",
    "img_filename": 'test_lion_1.jpg'
}]
PY_FILES_DEST_DIR = os.path.join('api_deployer', 'python-lib') + '.zip'
ENDPOINT_SETTINGS_BASE = {
    "type": "PY_FUNCTION",
    "userFunctionName": "api_py_function",
    "envSelection": {
        "envMode": "EXPLICIT_ENV"
    }
}
HTML_RESPONSE_TEMPLATE = """
    <div> Model succesfully deployed to API designer </div>
    <div>Model folder : {model_folder_id}</div>
    <div>API service : {service_id}</div>
    <div>Endpoint : {endpoint_id}</div>
    <a href="/projects/{project_key}/api-designer/{service_id}/endpoints/">See Service in API designer</a>
"""

