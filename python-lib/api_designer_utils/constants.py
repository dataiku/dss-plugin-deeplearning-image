TEST_IMG_DIR = 'test_images'
TEST_QUERIES = [{
    "name": "Score lion image",
    "img_filename": 'test_lion_1.png'
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
<a href="/projects/{plugin_id}/api-designer/">See Service in API designer</a>
"""