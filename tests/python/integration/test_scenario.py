# -*- coding: utf-8 -*-
from dku_plugin_test_utils import dss_scenario

TEST_PROJECT_KEY = "PLUGINTESTDEEPLEARNINGIMAGE"


def add_integration_test(user_dss_clients, scenario_id):
    dss_scenario.run(user_dss_clients, project_key=TEST_PROJECT_KEY, scenario_id=scenario_id)


def test_model_download(user_dss_clients):
    """Runs the macro to download model for each model available. It then verifies that we got all required files
    in output folder"""
    add_integration_test(user_dss_clients, "Macro_-_Model_Download")


def test_extract_features(user_dss_clients):
    """Runs extract recipe for each available model. It verifies that there are no errors, and that we did not
    loose any data in the process"""
    add_integration_test(user_dss_clients, "Recipe_-_Extract")


def test_classify_images(user_dss_clients):
    """Runs classify recipe for each available model. It verifies that there are no errors, and that we did not
        loose any data in the process. Moreover, it checks the accuracy of the prediction, and makes sure it is not
        below a certain threshold"""
    add_integration_test(user_dss_clients, "Recipe_-_Classify")


def test_retrain_model(user_dss_clients):
    """Runs retrain recipe for each available model. It verifies that there are no errors, and that all required
    files are present in output folder. Moreover, it runs a score recipe to check the accuracy of the new model"""
    add_integration_test(user_dss_clients, "Recipe_-_Retrain")


def test_run_tensorboard(user_dss_clients):
    """This test only checks if the webapp can be started without any error. There are no frontend checks as it
    is complicated to implement"""
    add_integration_test(user_dss_clients, "Webapp_-_Tensorboard")


def test_cloud_integration(user_dss_clients):
    """Runs an entire workflow on folders from the cloud"""
    add_integration_test(user_dss_clients, "Recipe_-_Cloud_Integration")


def test_api_endpoint_deployment(user_dss_clients):
    """Creates an API endpoint eand checks it has been created properly"""
    add_integration_test(user_dss_clients, "Macro_-_Deploy_API_Service")


def test_model_resnet(user_dss_clients):
    """Tests a specific model"""
    add_integration_test(user_dss_clients, "Model_-_Resnet")


def test_model_inception_v3(user_dss_clients):
    """Tests a specific model"""
    add_integration_test(user_dss_clients, "Model_-_InceptionV3")


def test_model_xception(user_dss_clients):
    """Tests a specific model"""
    add_integration_test(user_dss_clients, "Model_-_Xception")


def test_model_vgg16(user_dss_clients):
    """Tests a specific model"""
    add_integration_test(user_dss_clients, "Model_-_VGG16")
