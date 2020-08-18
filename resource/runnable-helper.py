import dataiku
from dku_deeplearning_image.applications import APPLICATIONS

api_client = dataiku.api_client()

def do(payload, config, plugin_config, inputs):
    if payload.get('parameterName') == 'output_managed_folder':
        project_key = dataiku.default_project_key()
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
    else:
        choices = []
    return {"choices": choices}
