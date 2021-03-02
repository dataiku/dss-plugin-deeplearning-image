import os
from dku_deeplearning_image.dku_constants import PLUGIN_ID

# API DESIGNER UTILS

## Plugin paths
PYTHON_LIB_DIR = 'python-lib'
DKU_DL_IMAGE_DIR = 'dku_deeplearning_image'
TEMPLATE_FILENAME = 'api_service_template.pyt'
API_DESIGNER_DIR = 'api_designer'

PLUGIN_INSTALLED_PATH = os.path.join('plugins', 'installed', PLUGIN_ID)
PLUGIN_DEV_PATH = os.path.join('plugins', 'dev', PLUGIN_ID)
PLUGIN_LIB_PATH = os.path.join(PLUGIN_INSTALLED_PATH, PYTHON_LIB_DIR)
PLUGIN_DEV_LIB_PATH = os.path.join(PLUGIN_DEV_PATH, PYTHON_LIB_DIR)
API_DESIGNER_UTILS_DIR = os.path.join(DKU_DL_IMAGE_DIR, API_DESIGNER_DIR)

SPEC_PATH = os.path.join(PYTHON_LIB_DIR, API_DESIGNER_UTILS_DIR, 'requirements.txt')
TEST_IMG_PATH = os.path.join(PYTHON_LIB_DIR, API_DESIGNER_UTILS_DIR, 'test_images')
TEMPLATE_PATH = os.path.join(PYTHON_LIB_DIR, API_DESIGNER_UTILS_DIR, TEMPLATE_FILENAME)
PY_FILES_DEST_DIR = os.path.join('api_deployer', PYTHON_LIB_DIR) + '.zip'

## Templates
TEST_QUERIES = [{
    "name": "Score lion image",
    "img_filename": 'test_lion_1.jpg'
}]
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