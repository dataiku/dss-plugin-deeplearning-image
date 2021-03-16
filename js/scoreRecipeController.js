const app = angular.module('deepLearningImageTools.recipe');

app.controller('scoringRecipeController', function($scope, utils) {

    const updateScopeData = function(data) {
        $scope.gpuInfo = data.gpu_info;
        $scope.styleSheetUrl = utils.getStylesheetUrl(data.pluginId);
    };

    const initVariables = function() {
        utils.initVariable($scope, "max_nb_labels", 5);
        utils.initVariable($scope, "min_threshold", 0);
    };

    const init = function() {
        $scope.finishedLoading = false;
        initVariables();
        utils.retrieveInfoBackend($scope, "get-info-scoring", updateScopeData);
    };

    init();
});