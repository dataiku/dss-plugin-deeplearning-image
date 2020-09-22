from .dku_config import DkuConfig


class ApiDeployerConfig(DkuConfig):
    def __init__(self, config, project, client):
        self.name = 'api_deployer'
        self.output_role = 'create_api_service'
        self.project = project
        self.client = client
        super(ApiDeployerConfig, self).__init__(config)

    def _load_recipe_param(self, config):
        super(ApiDeployerConfig, self)._load_recipe_param(config)
        self.add_param(
            name='model_folder_id',
            value=self.config.get("model_folder_id"),
            checks=[{
                'type': 'in',
                'op': [folder.get("id") for folder in self.project.list_managed_folders()],
                'err_msg': "Folder ID {value} must be the id of a managed folder containing a model trained with the deeplearning-image plugin. The folder must belong to the project in which is executed the macro"
            }],
            required=True)

        ##########################################
        # API Service handling
        ##########################################
        service_id = config.get("service_id")

        self.add_param(
            name='create_new_service',
            value=(service_id == "create_new_service"),
            checks=[
                {'type': 'is_type', 'op': bool, 'err_msg': "create_new_service is not bool: {value}"}
            ])

        list_service = [service.get("id") for service in self.project.list_api_services()]
        if self.create_new_service:
            service_id = config.get("new_service_id")
            check = {'type': 'not_in', 'op': list_service, 'err_msg': "Service ID {value} already in use."}
        else:
            check = {'type': 'in', 'op': list_service, 'err_msg': "Service ID : {value} not found."}
        self.add_param(
            name='service_id',
            value=service_id,
            required=True,
            checks=[check])

        self.add_param(
            name='endpoint_id',
            value=config.get("endpoint_id"),
            required=True)

        ##########################################
        # Code env handling
        ##########################################
        code_env_name = config.get("code_env_name")

        self.add_param(
            name='code_env_name',
            value=code_env_name,
            required=True,
            checks=[check])
        ##########################################
        # Classification parameters
        ##########################################

        self.add_param(
            name='max_nb_labels',
            value=int(self.config.get("max_nb_labels")),
            checks=[
                {'type': 'exists', 'err_msg': "Max number of labels is empty"},
                {'type': 'sup', 'op': 0, 'err_msg': "Max number of labels must be strictly greater than 0"}
            ])

        self.add_param(
            name='min_threshold',
            value=float(self.config.get("min_threshold")),
            checks=[
                {'type': 'exists', 'err_msg': "Min threshold is empty"},
                {'type': 'between', 'op': [0, 1], 'err_msg': "Min threshold must be between 0 and 1"}
            ])
