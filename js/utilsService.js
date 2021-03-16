const app = angular.module('deepLearningImageTools.recipe');

app.service("utils", function() {
    this.initVariable = function($scope, varName, initValue) {
        $scope.config[varName] = $scope.config[varName] || initValue;
    };

    this.retrieveInfoBackend = function($scope, method, updateScopeData) {
        $scope.callPythonDo({ method }).then(function(data) {
            updateScopeData(data);
            $scope.finishedLoading = true;
        }, function(data) {
            $scope.finishedLoading = true;
        });
    };

    this.getStylesheetUrl = function(pluginId) {
        return `/plugins/${pluginId}/resource/stylesheets/dl-image-toolbox.css`;
    };

    this.getShowHideAdvancedParamsMessage = function(showAdvancedParams) {
        return showAdvancedParams ? "Hide Model Summary" : "Show Model Summary";
    };
})