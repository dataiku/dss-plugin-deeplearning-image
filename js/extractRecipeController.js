var app = angular.module('deepLearningImageTools.extract', []);

app.controller('extractRecipeController', function($scope) {

    $scope.getShowHideAdvancedParamsMessage = function() {
        return $scope.showAdvancedParams ? "Hide Model Summary" : "Show Model Summary";
    };

    $scope.showHideAdvancedParams = function() {
        $scope.showAdvancedParams = !$scope.showAdvancedParams;
    };

    let preprocessLayers = function(layers) {
        return layers.reverse().map(function(layer, i) {
            let index = - ( i + 1);
            return {
                name: layer + " (" + index + ")",
                index: index
            };
        });
    };

    let initVariable = function(varName, initValue) {
        $scope.config[varName] = $scope.config[varName] || initValue;
    };

    let getStylesheetUrl = function(pluginId) {
        return `/plugins/${pluginId}/resource/stylesheets/dl-image-toolbox.css`
    }

    let initVariables = function() {
        initVariable("gpu_usage", 'all');
        initVariable("gpu_memory_allocation_mode", 'all');
        initVariable("gpu_memory_limit", 100);
    };
    
    let retrieveInfoOnModel = function() {

        $scope.callPythonDo({method: "get-info-about-model"}).then(function(data) {
            handleGPU(data);
            var defaultLayerIndex = data["default_layer_index"];
            $scope.layers = preprocessLayers(data.layers);
            $scope.modelSummary = data.summary;
            $scope.config.extract_layer_index = $scope.config.extract_layer_index || defaultLayerIndex
            $scope.styleSheetUrl = getStylesheetUrl(data.pluginId);
            $scope.finishedLoading = true;
        }, function(data) {
            $scope.finishedLoading = true;
        });
    };
    
    let handleGPU = function(data) {
        $scope.gpuList = data["gpu_list"];
        $scope.canUseGPU = data["can_use_gpu"];
        $scope.gpuUsage = data["gpu_usage_choices"];
        $scope.gpuMemoryAllocationMode = data["gpu_memory_allocation_mode"];
        initVariable("should_use_gpu", data["can_use_gpu"]);
    }
    
    let init = function() {
        $scope.finishedLoading = false;
        $scope.showAdvancedParams = false;
        initVariables();
        retrieveInfoOnModel();
    };

    init();
});
