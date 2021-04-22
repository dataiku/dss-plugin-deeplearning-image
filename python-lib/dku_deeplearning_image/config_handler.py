from dku_config import DkuConfig
import dku_deeplearning_image.dku_constants as constants


def add_gpu_config(dku_config, config):
    dku_config.add_param(
        name='should_use_gpu',
        value=config.get('should_use_gpu'),
        default=False
    )
    dku_config.add_param(
        name='gpu_usage',
        value=config.get('gpu_usage'))
    gpu_list = config.get('gpu_list') if dku_config.gpu_usage == 'custom' else []
    dku_config.add_param(
        name='gpu_list',
        value=gpu_list,
        checks=[{
            "type": "custom",
            "op": (not dku_config.should_use_gpu) or gpu_list or dku_config.gpu_usage != "custom",
            "err_msg": 'You have to select at least one GPU, or uncheck "Use GPU" checkbox.'
        }])
    dku_config.add_param(
        name='gpu_memory_allocation_mode',
        value=config.get('gpu_memory_allocation_mode'))
    dku_config.add_param(
        name='gpu_memory_limit',
        value=config.get('gpu_memory_limit') if dku_config.gpu_memory_allocation_mode == 'memory_limit' else 0)


def add_score_recipe_config(dku_config, config):
    dku_config.add_param(
        name='max_nb_labels',
        value=config.get('max_nb_labels'),
        required=True,
        cast_to=int)
    dku_config.add_param(
        name='min_threshold',
        value=config.get('min_threshold'),
        required=True,
        cast_to=float)


def add_extract_recipe_config(dku_config, config):
    dku_config.add_param(
        name='extract_layer_index',
        value=config.get('extract_layer_index'),
        cast_to=int,
        required=True,
        default=-2)


def add_retrain_recipe_config(dku_config, config):
    dku_config.add_param(name='col_filename', value=config.get("col_filename"), required=True)
    dku_config.add_param(name='col_label', value=config.get("col_label"), required=True)
    dku_config.add_param(name='train_ratio', value=config.get("train_ratio"), cast_to=float)
    dku_config.add_param(
        name='input_shape',
        value=(config.get("image_height"), config.get("image_width"), 3),
        cast_to=(lambda el: tuple(map(int, el))))
    dku_config.add_param(name='batch_size', value=config.get("batch_size"), cast_to=int, required=True)
    dku_config.add_param(name='model_pooling', value=config.get("model_pooling"), required=True)
    dku_config.add_param(name='model_reg', value=config.get("model_reg"))
    dku_config.add_param(name='model_dropout', value=config.get("model_dropout"), cast_to=float)
    dku_config.add_param(name='layer_to_retrain', value=config.get("layer_to_retrain"), required=True)
    dku_config.add_param(name='layer_to_retrain_n', value=config.get("layer_to_retrain_n"), cast_to=int)
    dku_config.add_param(name='optimizer', value=config.get("model_optimizer"), required=True)
    dku_config.add_param(name='learning_rate', value=config.get("model_learning_rate"), required=True)
    model_custom_params_opti = config.get("model_custom_params_opti", [])
    dku_config.add_param(
        name='custom_params_opti',
        value=model_custom_params_opti,
        checks=[
            {
                "type": "custom",
                "op": all([el and el.get('name') and 'value' in el for el in model_custom_params_opti]),
                "err_msg": "Names or values are missing in the Optimization custom parameters"
            }
        ]
    )
    dku_config.add_param(name='nb_epochs', value=config.get("nb_epochs"), cast_to=int, required=True)
    dku_config.add_param(name='nb_steps_per_epoch', value=config.get("nb_steps_per_epoch"), cast_to=int, required=True)
    dku_config.add_param(name='nb_validation_steps', value=config.get("nb_validation_steps"), cast_to=int, required=True)
    dku_config.add_param(name='data_augmentation', value=config.get("data_augmentation"), required=True)
    n_augmentation = config.get("n_augmentation") if dku_config.data_augmentation else 0
    dku_config.add_param(
        name='n_augmentation',
        value=n_augmentation,
        checks=[
            {
                'type': 'inf_eq',
                'op': dku_config.batch_size,
                'err_msg': "The number of augmentations must be lower than the batch size. Aborting."
            }
        ],
        cast_to=int
    )
    custom_params_data_augment = config.get("custom_params_data_augment", [])
    dku_config.add_param(
        name='custom_params_data_augment',
        value=config.get("model_custom_params_data_augmentation"),
        checks=[
            {
                "type": "custom",
                "op": all([el and el.get('name') and 'value' in el for el in custom_params_data_augment]),
                "err_msg": "Names or values are missing in the Data augmentation custom parameters"
            }
        ]
    )
    dku_config.add_param(name='use_tensorboard', value=config.get("tensorboard"))
    dku_config.add_param(name='random_seed', value=config.get("random_seed"), cast_to=int)


def add_api_deployer_config(dku_config, config, project):

    dku_config.add_param(
        name='model_folder_id',
        value=config.get("model_folder_id"),
        checks=[{
            'type': 'in',
            'op': [folder.get("id") for folder in project.list_managed_folders()],
            'err_msg': "Folder ID {value} must be the id of a managed folder containing a model trained with the "
                       "deeplearning-image-v2 plugin. The folder must belong to the project in which is executed the macro"
        }],
        required=True)

    ##########################################
    # API Service handling
    ##########################################
    service_id = config.get("service_id")

    dku_config.add_param(
        name='create_new_service',
        value=(service_id == "create_new_service"),
        checks=[
            {'type': 'is_type', 'op': bool, 'err_msg': "create_new_service is not bool: {value}"}
        ])

    list_service = [service.get("id") for service in project.list_api_services()]
    if dku_config.create_new_service:
        service_id = config.get("new_service_id")
        check = {'type': 'not_in', 'op': list_service, 'err_msg': "Service ID {value} already in use."}
    else:
        check = {'type': 'in', 'op': list_service, 'err_msg': "Service ID : {value} not found."}

    dku_config.add_param(
        name='service_id',
        value=service_id,
        required=True,
        checks=[check])

    dku_config.add_param(
        name='endpoint_id',
        value=config.get("endpoint_id"),
        required=True)

    ##########################################
    # Code env handling
    ##########################################
    dku_config.add_param(
        name='create_new_code_env',
        value=(config.get("code_env_options") == "new"),
        checks=[
            {'type': 'is_type', 'op': bool, 'err_msg': "create_new_code_env is not bool: {value}"}
        ])

    dku_config.add_param(
        name='code_env_name',
        value=config.get("code_env_name" if not dku_config.create_new_code_env else "new_code_env_name"),
        required=True)

    dku_config.add_param(
        name='python_interpreter',
        value=config.get("python_interpreter"),
        required=dku_config.create_new_code_env)

    dku_config.add_param(
        name='custom_interpreter',
        value=config.get("custom_interpreter"),
        required=dku_config.create_new_code_env and dku_config.python_interpreter == 'CUSTOM')

    ##########################################
    # Classification parameters
    ##########################################

    dku_config.add_param(
        name='max_nb_labels',
        value=config.get("max_nb_labels"),
        checks=[
            {'type': 'exists', 'err_msg': "Max number of labels is empty"},
            {'type': 'sup', 'op': 0, 'err_msg': "Max number of labels must be strictly greater than 0"}
        ],
        cast_to=int,
        required=True,
        default=2)

    dku_config.add_param(
        name='min_threshold',
        value=config.get("min_threshold"),
        checks=[
            {'type': 'exists', 'err_msg': "Min threshold is empty"},
            {'type': 'between', 'op': [0, 1], 'err_msg': "Min threshold must be between 0 and 1"}
        ],
        cast_to=float)


def create_dku_config(config, goal, project=None):
    dku_config = DkuConfig()
    add_gpu_config(dku_config, config)
    if goal == constants.GOAL.SCORE:
        add_score_recipe_config(dku_config, config)
    elif goal == constants.GOAL.RETRAIN:
        add_retrain_recipe_config(dku_config, config)
    elif goal == constants.GOAL.EXTRACT:
        add_extract_recipe_config(dku_config, config)
    elif goal == constants.GOAL.API_DESIGNER:
        add_api_deployer_config(dku_config, config, project)
    return dku_config
