const app = angular.module('deepLearningImageTools.recipe');

app.controller('retrainRecipeController', function ($scope, utils) {
    $scope.getShowHideAdvancedParamsMessage = function () {
        return utils.getShowHideAdvancedParamsMessage($scope.showAdvancedParams)
    };

    $scope.toggleAdvancedParams = function () {
        $scope.showAdvancedParams = !$scope.showAdvancedParams;
    };

    $scope.addCustomParam = function (paramsName) {
        $scope.config[paramsName].push({});
    };

    $scope.removeCustomParam = function (index, paramsName) {
        $scope.config[paramsName].splice(index, 1);
    };

    const updateCommonScopeData = function (data) {
        $scope.gpuInfo = data.gpu_info;
        $scope.styleSheetUrl = utils.getStylesheetUrl(data.pluginId);
    }

    const updateScopeData = function (data) {
        updateCommonScopeData(data)
        $scope.poolingOptions = data.pooling_options;
        $scope.layersOptions = data.layers_options;
        $scope.optimizerOptions = data.optimizer_options;
        $scope.labelColumns = data.columns;
        $scope.modelSummary = data.model_summary;
        initPotentiallyBlockedVariables(data.model_config);
    };

    const initPotentiallyBlockedVariables = function (modelConfig) {
        $scope.retrained = modelConfig.retrained || false;
        let poolingDefault = $scope.retrained ? modelConfig.top_params.pooling : "avg";
        let imageWidthDefault = $scope.retrained ? modelConfig.top_params.input_shape[0] : 197;
        let imageHeightDefault = $scope.retrained ? modelConfig.top_params.input_shape[1] : 197;
        utils.initVariable($scope, 'model_pooling', poolingDefault);
        utils.initVariable($scope, 'image_width', imageWidthDefault);
        utils.initVariable($scope, 'image_height', imageHeightDefault);
    };

    const initVariables = function () {
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

    const init = function () {
        $scope.finishedLoading = false;
        $scope.showAdvancedParams = false;
        initVariables();
        utils.retrieveInfoBackend($scope, "get-info-retrain", updateScopeData);
    };

    init();
});