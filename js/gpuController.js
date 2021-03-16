const app = angular.module('deepLearningImageTools.recipe');

app.controller('gpuController', function($scope, utils) {
    const initVariables = function() {
        utils.initVariable($scope, "gpu_usage", 'all');
        utils.initVariable($scope, "gpu_memory_allocation_mode", 'all');
        utils.initVariable($scope, "gpu_memory_limit", 100);
        utils.initVariable($scope, "should_use_gpu", $scope.canUseGPU);
    };

    const updateScopeData = function() {
        $scope.gpuList = $scope.gpuInfo.gpu_list;
        $scope.canUseGPU = $scope.gpuInfo.can_use_gpu;
        $scope.gpuUsage = $scope.gpuInfo.gpu_usage_choices;
        $scope.gpuMemoryAllocationMode = $scope.gpuInfo.gpu_memory_allocation_mode;
    }

    const init = function() {
        initVariables();
        updateScopeData();
    };

    $scope.$watch('gpuInfo', function() {
        if ($scope.gpuInfo) {
            init();
        }
    });
})

app.directive('gpuForm', function() {
  return {
    templateUrl: '/plugins/deeplearning-image-v2/resource/templates/gpu-form-template.html'
  };
});