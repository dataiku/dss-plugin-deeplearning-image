import os

PYTHON_LIB_DIR = 'python-lib'
API_DESIGNER_UTILS_DIR = os.path.join(PYTHON_LIB_DIR, 'api_designer_utils')
TEMPLATE_FILENAME = 'api_service_template.pyt'
TEMPLATE_PATH = os.path.join(API_DESIGNER_UTILS_DIR, TEMPLATE_FILENAME)
TEST_IMG_DIR = os.path.join(API_DESIGNER_UTILS_DIR, 'test_images')
TEST_QUERIES = [{
    "name": "Score lion image",
    "img_filename": 'test_lion_1.jpg'
}]
PY_FILES_DEST_DIR = os.path.join('api_deployer', 'python-lib') + '.zip'
SPEC_PATH = 'code-env/python/spec/requirements.txt'
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
    <a href="/projects/{project_key}/api-designer/">See Service in API designer</a>
"""