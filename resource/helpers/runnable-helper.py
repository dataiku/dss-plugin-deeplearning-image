import dataiku
from dku_deeplearning_image.applications import APPLICATIONS

api_client = dataiku.api_client()

def do(payload, config, plugin_config, inputs):
    project_key = dataiku.default_project_key()
    if payload.get('parameterName') == 'output_managed_folder':
        project_managed_folders = api_client.get_project(project_key).list_managed_folders()

        choices = [{
            'label': '{} ({})'.format(mf['name'], mf['type']),
            'value': mf['id']
        } for mf in project_managed_folders]
        choices.append({'label': 'Create new Filesystem folder...', 'value': 'create_new_folder'})
    elif payload.get('parameterName') == 'model_choice':
        choices = [{
            'label': '{app[label]} trained on {ds}'.format(app=app, ds=ds.capitalize()),
            'value': '{}::{}'.format(app['name'], ds)
        } for app in APPLICATIONS for ds in list(app['weights'].keys())]
    elif payload.get('parameterName') == 'service_id':
        api_services = api_client.get_project(project_key).list_api_services()
        choices = [{
            'label': '{api_service[id]}'.format(api_service=api_service),
            'value': '{api_service[id]}'.format(api_service=api_service)
        } for api_service in api_services]
        choices.append({'label': 'Create new service...', 'value': 'create_new_service'})
    elif payload.get('parameterName') == 'code_env_name':
        code_envs = api_client.list_code_envs()
        choices = [{
            'label': '{code_env[envName]}'.format(code_env=code_env),
            'value': '{code_env[envName]}'.format(code_env=code_env)
        } for code_env in code_envs]
        choices.append({'label': 'Create new code env...', 'value': 'create_new_code_env'})
    else:
        choices = []
    return {"choices": choices}
