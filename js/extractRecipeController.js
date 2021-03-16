const app = angular.module('deepLearningImageTools.recipe');

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