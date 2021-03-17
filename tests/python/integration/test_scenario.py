# -*- coding: utf-8 -*-
from dku_plugin_test_utils import dss_scenario

TEST_PROJECT_KEY = "PLUGINTESTDEEPLEARNINGIMAGE"


def add_integration_test(user_dss_clients, scenario_id):
    dss_scenario.run(user_dss_clients, project_key=TEST_PROJECT_KEY, scenario_id=scenario_id)


def test_run_tensorboard(user_dss_clients):
    add_integration_test(user_dss_clients, "Webapp_-_Tensorboard")


def test_extract_features(user_dss_clients):
    add_integration_test(user_dss_clients, "Recipe_-_Extract_features")


def test_classify_images(user_dss_clients):
    add_integration_test(user_dss_clients, "Recipe_-_Classify_images")


def test_retrain_model(user_dss_clients):
    add_integration_test(user_dss_clients, "Recipe_-_Retrain_model")


def test_advanced_retrain(user_dss_clients):
    add_integration_test(user_dss_clients, "Recipe_-_Advanced_Retrain")


def test_model_download(user_dss_clients):
    add_integration_test(user_dss_clients, "Macro_-_Model_Download")


def test_cloud_integration(user_dss_clients):
    add_integration_test(user_dss_clients, "Recipe_-_Cloud_Integration")


def test_edge_cases(user_dss_clients):
    add_integration_test(user_dss_clients, "Recipe_-_Edge_cases")


def test_api_endpoint_deployment(user_dss_clients):
    add_integration_test(user_dss_clients, "Macro_-_Deploy_API_Service")
