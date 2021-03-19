import dataiku
from dku_deeplearning_image.keras_applications import APPLICATIONS
import dku_deeplearning_image.api_designer.api_designer_utils as utils

api_client = dataiku.api_client()


def get_code_options():
    choices = [{
        "label": "Select an environment",
        "value": "existing"
    }]
    if utils.is_user_admin():
        choices.append({
            "label": "Create new environment",
            "value": "new"
        })
    return choices


def get_output_managed_folder(project_key):
    project_managed_folders = api_client.get_project(project_key).list_managed_folders()

    choices = [{
        'label': '{} ({})'.format(mf['name'], mf['type']),
        'value': mf['id']
    } for mf in project_managed_folders]
    choices.append({'label': 'Create new Filesystem folder...', 'value': 'create_new_folder'})
    return choices


def get_model_choice():
    choices = [{
        'label': '{app[label]} trained on {ds}'.format(app=app, ds=ds.capitalize()),
        'value': '{}::{}'.format(app['name'].value, ds)
    } for app in APPLICATIONS for ds in list(app['weights'].keys())]
    return choices


def get_services_id(project_key):
    api_services = api_client.get_project(project_key).list_api_services()
    choices = [{
        'label': '{api_service[id]}'.format(api_service=api_service),
        'value': '{api_service[id]}'.format(api_service=api_service)
    } for api_service in api_services]
    choices.append({'label': 'Create new service...', 'value': 'create_new_service'})
    return choices


def get_code_envs():
    code_envs = api_client.list_code_envs()
    choices = [{
        'label': '{code_env[envName]}'.format(code_env=code_env),
        'value': '{code_env[envName]}'.format(code_env=code_env)
    } for code_env in code_envs]
    choices.append({'label': 'Create new code env...', 'value': 'create_new_code_env'})
    return choices


def do(payload, config, plugin_config, inputs):
    project_key = dataiku.default_project_key()
    if payload.get('parameterName') == 'output_managed_folder':
        choices = get_output_managed_folder(project_key)
    elif payload.get('parameterName') == 'model_choice':
        choices = get_model_choice()
    elif payload.get('parameterName') == 'service_id':
        choices = get_services_id(project_key)
    elif payload.get('parameterName') == 'code_env_name':
        choices = get_code_envs()
    elif payload.get('parameterName') == 'code_env_options':
        choices = get_code_options()
    else:
        choices = []
    return {"choices": choices}
