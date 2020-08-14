var app = angular.module('deepLearningImageTools.retrain', []);

app.controller('retrainRecipeController', function($scope) {

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

    $scope.addCustomParam = function(paramsName) {
        $scope.config[paramsName].push({});
    };

    $scope.removeCustomParam = function(index, paramsName) {
        $scope.config[paramsName].splice(index, 1);
    };

    $scope.getShowHideAdvancedParamsMessage = function() {
        return $scope.showAdvancedParams ? "Hide Model Summary" : "Show Model Summary";
    };

    $scope.showHideAdvancedParams = function() {
        $scope.showAdvancedParams = !$scope.showAdvancedParams;
    };


    var retrieveInfoRetrain = function() {
        $scope.callPythonDo({method: "get-info-retrain"}).then(function(data) {
            handleGPU(data);
            $scope.labelColumns = data["columns"];
            $scope.modelSummary = data["summary"];
            initPotentiallyBlockedVariables(data["model_config"]);
            $scope.styleSheetUrl = getStylesheetUrl(data.pluginId);
            $scope.finishedLoading = true;
        }, function(data) {
            $scope.finishedLoading = true;
        });
    };

    var initVariable = function(varName, initValue) {
        $scope.config[varName] = $scope.config[varName] || initValue;
    };

    var getStylesheetUrl = function(pluginId) {
        return `/plugins/${pluginId}/resource/stylesheets/dl-image-toolbox.css`
    }

    var initPotentiallyBlockedVariables = function(modelConfig) {
        $scope.retrained = modelConfig.retrained || false;
        var poolingDefault = "avg";
        var imageWidthDefault = 197;
        var imageHeightDefault = 197;
        if ($scope.retrained) {
            poolingDefault = modelConfig.top_params.pooling;
            imageWidthDefault = modelConfig.top_params.input_shape[0];
            imageHeightDefault = modelConfig.top_params.input_shape[1];
        }
        initVariable('model_pooling', poolingDefault);
        initVariable('image_width', imageWidthDefault);
        initVariable('image_height', imageHeightDefault);
    };

    var initVariables = function() {
        initVariable("random_seed", 1337);
        initVariable("train_ratio", 0.8);
        initVariable("gpu_usage", 'all');
        initVariable("gpu_memory", 'all');
        initVariable('layer_to_retrain', 'last');
        initVariable('layer_to_retrain_n', 2);
        initVariable('model_dropout', 0);
        initVariable('model_reg', {"l1": 0, "l2": 0});
        initVariable('model_optimizer', "adam");
        initVariable('model_learning_rate', 0.001);
        initVariable('batch_size', 32);
        initVariable('nb_epochs', 10);
            initVariable('nb_steps_per_epoch', 50);
        initVariable('nb_validation_steps', 25);
        initVariable('model_custom_params_opti', []);
        initVariable('n_augmentation', 2);
        initVariable('model_custom_params_data_augmentation', []);
        initVariable('data_augmentation', false);
        initVariable('tensorboard', false);
    };
    
    var handleGPU = function(data) {
        $scope.gpuList = data["gpu_list"];
        $scope.canUseGPU = data["can_use_gpu"];
        $scope.gpuUsage = data["gpu_usage_choices"];
        $scope.gpuMemory = data["gpu_memory_choices"];
        initVariable("should_use_gpu", data["can_use_gpu"]);
    }

    var init = function() {
        $scope.finishedLoading = false;
        $scope.showAdvancedParams = false;
        initVariables();
        retrieveInfoRetrain();
    };

    init();
});