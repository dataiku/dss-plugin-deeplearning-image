var app = angular.module('deepLearningImageTools.extract', []);

app.controller('extractRecipeController', function($scope) {

    $scope.getShowHideAdvancedParamsMessage = function() {
        if ($scope.showAdvancedParams) {
            return "Hide Model Summary";
        } else {
            return "Show Model Summary";
        }
    };

    $scope.showHideAdvancedParams = function() {
        $scope.showAdvancedParams = !$scope.showAdvancedParams;
    };

    var preprocessLayers = function(layers) {
        return layers.reverse().map(function(layer, i) {
            var index = - ( i + 1);
            return {
                name: layer + " (" + index + ")",
                index: index
            };
        });
    };

    var initVariable = function(varName, initValue) {
        if ($scope.config[varName] == undefined) {
            $scope.config[varName] = initValue;
        }
    };

    var initVariables = function() {
        initVariable("gpu_usage", 'all');
        initVariable("gpu_memory", 'all');
    };
    
    var retrieveInfoOnModel = function() {

        $scope.callPythonDo({method: "get-info-about-model"}).then(function(data) {
            handleGPU(data);
            var defaultLayerIndex = data["default_layer_index"];
            $scope.layers = preprocessLayers(data.layers);
            $scope.modelSummary = data.summary;

            if ($scope.config.extract_layer_index == undefined) {
                $scope.config.extract_layer_index = defaultLayerIndex
            }
            $scope.finishedLoading = true;
        }, function(data) {
            // TODO : Deal when failing to retrieve info
            $scope.finishedLoading = true;
        });
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
        retrieveInfoOnModel();
    };

    init();
});
