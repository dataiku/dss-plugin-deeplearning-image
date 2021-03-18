# -*- coding: utf-8 -*-
from dku_plugin_test_utils import dss_scenario

TEST_PROJECT_KEY = "PLUGINTESTDEEPLEARNINGIMAGE"


def add_integration_test(user_dss_clients, scenario_id):
    dss_scenario.run(user_dss_clients, project_key=TEST_PROJECT_KEY, scenario_id=scenario_id)


def test_run_tensorboard(user_dss_clients):
    """This test only checks if the webapp can be started without any error. There are no frontend checks as it
    is complicated to implement"""
    add_integration_test(user_dss_clients, "Webapp_-_Tensorboard")


def test_extract_features(user_dss_clients):
    """Runs extract recipe for each available model. It verifies that there are no errors, and that we did not
    loose any data in the process"""
    add_integration_test(user_dss_clients, "Recipe_-_Extract_features")


def test_classify_images(user_dss_clients):
    """Runs classify recipe for each available model. It verifies that there are no errors, and that we did not
        loose any data in the process. Moreover, it checks the accuracy of the prediction, and makes sure it is not
        below a certain threshold"""
    add_integration_test(user_dss_clients, "Recipe_-_Classify_images")


def test_retrain_model(user_dss_clients):
    """Runs retrain recipe for each available model. It verifies that there are no errors, and that all required
    files are present in output folder. Moreover, it runs a score recipe to check the accuracy of the new model"""
    add_integration_test(user_dss_clients, "Recipe_-_Retrain_model")


def test_advanced_retrain(user_dss_clients):
    """Runs retrain recipe for one model with a bunch of different settings"""
    add_integration_test(user_dss_clients, "Recipe_-_Advanced_Retrain")


def test_model_download(user_dss_clients):
    """Runs the macro to download model for each model available. It then verifies that we got all required files
    in output folder"""
    add_integration_test(user_dss_clients, "Macro_-_Model_Download")


def test_cloud_integration(user_dss_clients):
    """Runs an entire workflow on folders from the cloud"""
    add_integration_test(user_dss_clients, "Recipe_-_Cloud_Integration")


def test_edge_cases(user_dss_clients):
    """Runs score recipe on edge cases (folders without any files, invalid files, etc.)"""
    add_integration_test(user_dss_clients, "Recipe_-_Edge_cases")


def test_api_endpoint_deployment(user_dss_clients):
    """Creates an API endpoint eand checks it has been created properly"""
    add_integration_test(user_dss_clients, "Macro_-_Deploy_API_Service")
