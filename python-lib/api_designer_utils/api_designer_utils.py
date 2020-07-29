import dataiku
import api_designer_utils.constants as constants
import os
import logging
import base64
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,  # avoid getting log from 3rd party module
                    format='deeplearning-image-macro %(levelname)s - %(message)s')


def copy_plugin_to_dss_folder(plugin, folder_id, project_key):
    """
    Copy python-lib from a plugin to a managed folder
    """
    required_files = {
        'config': ['__init__.py', 'api_deployer_config.py', 'dku_config.py'],
        'dku_deeplearning_image': ['__init__.py', 'utils.py', 'constants.py', 'applications.py'],
        'recipe': ['__init__.py', 'score_recipe.py'],
        'utils_objects': ['__init__.py', 'dku_model.py', 'dku_application.py'],
    }
    SOURCE_DIR = 'python-lib'
    DEST_DIR = os.path.join('api_deployer', 'python-lib')
    folder = dataiku.Folder(folder_id, project_key=project_key)
    for foldername, files in required_files.items():
        for filename in files:
            source_path = os.path.join(SOURCE_DIR, foldername, filename)
            dest_path = os.path.join(DEST_DIR, foldername, filename)
            file_content = plugin.get_file(source_path).read()
            folder.upload_stream(dest_path, file_content)

    
def get_api_service(service_id, create_service, project):
    """
    Create or get an api service dss object and return it
    """
    return (project.create_api_service if create_service else project.get_api_service)(service_id)


def create_api_code_env(plugin, client, env_name):
    SPEC_PATH = 'code-env/python/spec/requirements.txt'
    already_exist = len([env.get('envName') for env in client.list_code_envs() if env == env_name])
    if not already_exist:
        _ = client.create_code_env(env_lang='PYTHON', env_name=env_name, deployment_mode='DESIGN_MANAGED')
    my_env = client.get_code_env('PYTHON', env_name)
    env_def = my_env.get_definition()
    libraries_to_install = plugin.get_file(SPEC_PATH).read()
    env_def['specPackageList'] = libraries_to_install
    env_def['desc']['installCorePackages'] = True
    my_env.set_definition(env_def)
    my_env.update_packages()

def get_test_queries():
    formatted_queries = []
    for query in constants.TEST_QUERIES:
        test_img_path = os.path.join(constants.TEST_IMG_FILENAME, query['img_filename'])
        test_img_64 = base64.encodebytes(open(test_img_path, 'rb').read())
        formatted_queries.append({
                "name": query["name"],
                "q": {"img_b64": test_img_64}
            })
    return formatted_queries


def build_model_endpoint_settings(endpoint_id, code_env_name, model_folder_id, max_nb_labels, min_threshold):
    """
    Create a endpoints dict that will be added to a list of endpoints of an api service
    """
    endpoint_settings = constants.ENDPOINT_SETTINGS_BASE
    endpoint_settings["id"] = endpoint_id
    endpoint_settings['envSelection']['envName'] = code_env_name
    endpoint_settings["inputFolderRefs"] = [{"ref": model_folder_id}]
    endpoint_settings["testQueries"] = get_test_queries()
    endpoint_settings["code"] = format_code_template(
        max_nb_labels=max_nb_labels,
        min_threshold=min_threshold)
    return endpoint_settings


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


def get_html_result(plugin_id, model_folder_id, service_id, endpoint_id):
    """
    Get the result html string of the macro
    """
    return constants.HTML_RESPONSE_TEMPLATE.format(
        plugin_id=plugin_id,
        model_folder_id=model_folder_id,
        service_id=service_id,
        endpoint_id=endpoint_id)
