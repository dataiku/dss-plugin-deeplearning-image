const app = angular.module('deepLearningImageTools.recipe', []);

app.controller('scoringRecipeController', function ($scope, utils) {

    const updateCommonScopeData = function (data) {
        $scope.gpuInfo = data.gpu_info;
        $scope.styleSheetUrl = utils.getStylesheetUrl(data.pluginId);
    }

    const updateScopeData = function (data) {
        updateCommonScopeData(data)
    };

    const initVariables = function () {
        utils.initVariable($scope, "max_nb_labels", 5);
        utils.initVariable($scope, "min_threshold", 0);
    };

    const init = function () {
        $scope.finishedLoading = false;
        initVariables();
        utils.retrieveInfoBackend($scope, "get-info-scoring", updateScopeData);
    };

    init();
});

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

app.controller('extractRecipeController', function ($scope, utils) {
    $scope.getShowHideAdvancedParamsMessage = function () {
        return utils.getShowHideAdvancedParamsMessage($scope.showAdvancedParams)
    };

    $scope.toggleAdvancedParams = function () {
        $scope.showAdvancedParams = !$scope.showAdvancedParams;
    };

    const preprocessLayers = function (layers) {
        return layers.reverse().map(function (layer, i) {
            let index = -(i + 1);
            return {
                name: layer + " (" + index + ")",
                index: index
            };
        });
    };

    const updateCommonScopeData = function (data) {
        $scope.gpuInfo = data.gpu_info;
        $scope.styleSheetUrl = utils.getStylesheetUrl(data.pluginId);
    }

    const updateScopeData = function (data) {
        updateCommonScopeData(data);
        $scope.layers = preprocessLayers(data.layers);
        $scope.modelSummary = data.summary;
        $scope.config.extract_layer_index = $scope.config.extract_layer_index || data.default_layer_index;
    };

    const init = function () {
        $scope.finishedLoading = false;
        $scope.showAdvancedParams = false;
        utils.retrieveInfoBackend($scope, "get-info-about-model", updateScopeData);
    };

    init();
});

app.controller('gpuController', function ($scope, utils) {
    const initVariables = function () {
        utils.initVariable($scope, "gpu_usage", 'all');
        utils.initVariable($scope, "gpu_memory_allocation_mode", 'all');
        utils.initVariable($scope, "gpu_memory_limit", 100);
        utils.initVariable($scope, "should_use_gpu", $scope.gpuInfo.can_use_gpu);
    };

    $scope.$watch('gpuInfo', function () {
        if ($scope.gpuInfo) {
            initVariables();
        }
    });
})

app.directive('gpuForm', function () {
    return {
        templateUrl: '/plugins/deeplearning-image-v2/resource/templates/gpu-form-template.html'
    };
});

app.service("utils", function () {
    this.initVariable = function ($scope, varName, initValue) {
        $scope.config[varName] = $scope.config[varName] || initValue;
    };

    this.retrieveInfoBackend = function ($scope, method, updateScopeData) {
        $scope.callPythonDo({method}).then(function (data) {
            updateScopeData(data);
            $scope.finishedLoading = true;
        }, function (data) {
            $scope.finishedLoading = true;
        });
    };

    this.getStylesheetUrl = function (pluginId) {
        return `/plugins/${pluginId}/resource/stylesheets/dl-image-toolbox.css`;
    };

    this.getShowHideAdvancedParamsMessage = function (showAdvancedParams) {
        return showAdvancedParams ? "Hide Model Summary" : "Show Model Summary";
    };
})