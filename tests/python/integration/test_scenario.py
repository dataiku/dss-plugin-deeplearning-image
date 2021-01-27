import pytest
import logging

from dku_plugin_test_utils import dss_scenario


pytestmark = pytest.mark.usefixtures("plugin", "dss_target")


test_kwargs = {
    "user": "user1",
    "project_key": "PLUGINTESTDEEPLEARNINGIMAGE",
    "logger": logging.getLogger("dss-plugin-test.deeplearning-image.test_scenario"),
}


def add_integration_test(user_clients, scenario_id):
    test_kwargs["client"] = user_clients[test_kwargs["user"]]
    dss_scenario.run(scenario_id=scenario_id, **test_kwargs)


def test_run_tensorboard_start(user_clients):
    add_integration_test(user_clients, "Webapp_-_Tensorboard")


def test_extract_recipe(user_clients):
    add_integration_test(user_clients, "Recipe_-_Extract_features")


def test_score_recipe(user_clients):
    add_integration_test(user_clients, "Recipe_-_Classify_images")


def test_retrain_recipe(user_clients):
    add_integration_test(user_clients, "Recipe_-_Retrain_model")


def test_advanced_retrain(user_clients):
    add_integration_test(user_clients, "Recipe_-_Advanced_Retrain")


def test_model_download(user_clients):
    add_integration_test(user_clients, "Macro_-_Model_Download")


def test_cloud_integration(user_clients):
    add_integration_test(user_clients, "Recipe_-_Cloud_Integration")


def test_edge_cases(user_clients):
    add_integration_test(user_clients, "Recipe_-_Edge_cases")


def test_api_endpoint_deployment(user_clients):
    add_integration_test(user_clients, "Macro_-_Deploy_API_Service")
