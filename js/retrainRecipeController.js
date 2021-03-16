const app = angular.module('deepLearningImageTools.recipe');

app.controller('retrainRecipeController', function($scope, utils) {
    $scope.getShowHideAdvancedParamsMessage = function() {
        return utils.getShowHideAdvancedParamsMessage($scope.showAdvancedParams)
    };

    $scope.toggleAdvancedParams = function() {
        $scope.showAdvancedParams = !$scope.showAdvancedParams;
    };

    $scope.addCustomParam = function(paramsName) {
        $scope.config[paramsName].push({});
    };

    $scope.removeCustomParam = function(index, paramsName) {
        $scope.config[paramsName].splice(index, 1);
    };

    const updateScopeData = function(data) {
        $scope.poolingOptions = [
            ["No pooling", "None"],
            ["Average", "avg"],
            ["Maximum", "max"]
        ];

        $scope.layersOptions = [
            ["Last layer", "last"],
            ["All layers", "all"],
            ["N last layers", "n_last"]
        ];

        $scope.optimizerOptions = [
            ["Adam", "adam"],
            ["Adagrad", "adagrad"],
            ["SGD", "sgd"]
        ];
        $scope.gpuInfo = data.gpu_info;
        $scope.styleSheetUrl = utils.getStylesheetUrl(data.pluginId);
        $scope.utils = utils;
        $scope.labelColumns = data.columns;
        $scope.modelSummary = data.summary;
        initPotentiallyBlockedVariables(data.model_config);
    };

    const initPotentiallyBlockedVariables = function(modelConfig) {
        $scope.retrained = modelConfig.retrained || false;
        let poolingDefault = "avg";
        let imageWidthDefault = 197;
        let imageHeightDefault = 197;
        if ($scope.retrained) {
            poolingDefault = modelConfig.top_params.pooling;
            imageWidthDefault = modelConfig.top_params.input_shape[0];
            imageHeightDefault = modelConfig.top_params.input_shape[1];
        }
        utils.initVariable($scope, 'model_pooling', poolingDefault);
        utils.initVariable($scope, 'image_width', imageWidthDefault);
        utils.initVariable($scope, 'image_height', imageHeightDefault);
    };

    const initVariables = function() {
        utils.initVariable($scope, "random_seed", 1337);
        utils.initVariable($scope, "train_ratio", 0.8);
        utils.initVariable($scope, "gpu_usage", 'all');
        utils.initVariable($scope, "gpu_memory_allocation_mode", 'all');
        utils.initVariable($scope, "gpu_memory_limit", 100);
        utils.initVariable($scope, 'layer_to_retrain', 'last');
        utils.initVariable($scope, 'layer_to_retrain_n', 2);
        utils.initVariable($scope, 'model_dropout', 0);
        utils.initVariable($scope, 'model_reg', {"l1": 0, "l2": 0});
        utils.initVariable($scope, 'model_optimizer', "adam");
        utils.initVariable($scope, 'model_learning_rate', 0.001);
        utils.initVariable($scope, 'batch_size', 32);
        utils.initVariable($scope, 'nb_epochs', 10);
        utils.initVariable($scope, 'nb_steps_per_epoch', 50);
        utils.initVariable($scope, 'nb_validation_steps', 25);
        utils.initVariable($scope, 'model_custom_params_opti', []);
        utils.initVariable($scope, 'n_augmentation', 2);
        utils.initVariable($scope, 'model_custom_params_data_augmentation', []);
        utils.initVariable($scope, 'data_augmentation', false);
        utils.initVariable($scope, 'tensorboard', false);
    };

    const init = function() {
        $scope.finishedLoading = false;
        $scope.showAdvancedParams = false;
        initVariables();
        utils.retrieveInfoBackend($scope, "get-info-retrain", updateScopeData);
    };

    init();
});