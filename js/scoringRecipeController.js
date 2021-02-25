var app = angular.module('deepLearningImageTools.scoring', []);

app.controller('scoringRecipeController', function($scope) {


    let retrieveCanUseGPU = function() {

        $scope.callPythonDo({method: "get-info-scoring"}).then(function(data) {
            handleGPU(data);
            $scope.styleSheetUrl = getStylesheetUrl(data.pluginId);
            $scope.finishedLoading = true;
        }, function(data) {
            $scope.finishedLoading = true;
        });
    };

    let initVariable = function(varName, initValue) {
        $scope.config[varName] = $scope.config[varName] || initValue;
    };

    let getStylesheetUrl = function(pluginId) {
        return `/plugins/${pluginId}/resource/stylesheets/dl-image-toolbox.css`
    }

    let initVariables = function() {
        initVariable("max_nb_labels", 5);
        initVariable("min_threshold", 0);
        initVariable("gpu_usage", 'all');
        initVariable("gpu_memory_allocation_mode", 'all');
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
        initVariables();
        retrieveCanUseGPU();
    };

    init();
});
