import dataiku
import api_designer_utils.constants as constants
import os
import logging
import base64
import zipfile
from io import BytesIO

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,  # avoid getting log from 3rd party module
                    format='deeplearning-image-macro %(levelname)s - %(message)s')


def list_all_paths_rec(d, paths):
    for folder in d:
        if not 'children' in folder:
            paths.append(folder['path'])
        else:
            list_all_paths_rec(folder['children'], paths)

def list_all_paths(folder):
    paths = []
    list_all_paths_rec(folder, paths)
    return paths

def copy_plugin_to_dss_folder(plugin, folder_id, project_key):
    """
    Copy python-lib from a plugin to a managed folder
    """
    python_lib_dir = list(filter(lambda x: x['name'] == constants.PYTHON_LIB_DIR, plugin.list_files()))
    required_files = list_all_paths(python_lib_dir)
    DEST_DIR = constants.PY_FILES_DEST_DIR
    zip_file = BytesIO()
    folder = dataiku.Folder(folder_id, project_key=project_key)
    with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipper:
        for path in required_files:
            dest_path = '/'.join(path.split('/')[1:])
            file_content = plugin.get_file(path).read()
            zipper.writestr(dest_path, data=file_content)
    folder.upload_stream(DEST_DIR, zip_file.getvalue())


def get_api_service(project, create_new_service, service_id=None):
    """
    Create or get an api service dss object and return it
    """
    return (project.create_api_service if create_new_service else project.get_api_service)(service_id)


def create_api_code_env(plugin, client, env_name):
    already_exist = len([env.get('envName') for env in client.list_code_envs() if env == env_name])
    if not already_exist:
        _ = client.create_code_env(env_lang='PYTHON', env_name=env_name, deployment_mode='DESIGN_MANAGED')
    my_env = client.get_code_env('PYTHON', env_name)
    env_def = my_env.get_definition()
    libraries_to_install = plugin.get_file(constants.SPEC_PATH).read()
    env_def['specPackageList'] = libraries_to_install
    env_def['desc']['installCorePackages'] = True
    my_env.set_definition(env_def)
    my_env.update_packages()


def get_test_queries(plugin):
    formatted_queries = []
    for query in constants.TEST_QUERIES:
        test_img_path = os.path.join(constants.TEST_IMG_DIR, query['img_filename'])
        test_img = plugin.get_file(test_img_path).read()
        test_img_64 = base64.encodebytes(test_img).decode('utf-8')
        formatted_queries.append({
            "name": query["name"],
            "q": {"img_b64": test_img_64}
        })
    return formatted_queries


def build_model_endpoint_settings(plugin, endpoint_id, code_env_name, model_folder_id, max_nb_labels, min_threshold):
    """
    Create a endpoints dict that will be added to a list of endpoints of an api service
    """
    endpoint_settings = constants.ENDPOINT_SETTINGS_BASE
    endpoint_settings["id"] = endpoint_id
    endpoint_settings['envSelection']['envName'] = code_env_name
    endpoint_settings["inputFolderRefs"] = [{"ref": model_folder_id}]
    endpoint_settings["testQueries"] = get_test_queries(plugin)
    endpoint_settings["code"] = format_code_template(
        plugin=plugin,
        max_nb_labels=max_nb_labels,
        min_threshold=min_threshold,
        python_lib_dir=constants.PY_FILES_DEST_DIR)
    return endpoint_settings


def format_code_template(plugin, **kwargs):
    template_content = plugin.get_file(constants.TEMPLATE_PATH).read().decode('utf-8')
    template_content_formatted = template_content.format(**kwargs)
    return template_content_formatted


def create_python_endpoint(api_service, setting_dict):
    """
    Create or update an endpoint to the API service DSS object api_service.
    """
    api_setting = api_service.get_settings()
    api_setting_details = api_setting.get_raw()

    endpoints = [endpoint for endpoint in api_setting_details['endpoints'] if endpoint['id'] != setting_dict['id']]
    endpoints.append(setting_dict)
    api_setting_details['endpoints'] = endpoints
    api_setting.save()


def get_html_result(project_key, model_folder_id, service_id, endpoint_id):
    """
    Get the result html string of the macro
    """
    return constants.HTML_RESPONSE_TEMPLATE.format(
        project_key=project_key,
        model_folder_id=model_folder_id,
        service_id=service_id,
        endpoint_id=endpoint_id)
