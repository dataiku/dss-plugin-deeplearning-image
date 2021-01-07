var app = angular.module('deepLearningImageTools.scoring', []);

app.controller('scoringRecipeController', function($scope) {


    var retrieveCanUseGPU = function() {

        $scope.callPythonDo({method: "get-info-scoring"}).then(function(data) {
            handleGPU(data);
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

    var initVariables = function() {
        initVariable("max_nb_labels", 5);
        initVariable("min_threshold", 0);
        initVariable("gpu_usage", 'all');
        initVariable("gpu_memory", 'all');
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
        initVariables();
        retrieveCanUseGPU();
    };

    init();
});
