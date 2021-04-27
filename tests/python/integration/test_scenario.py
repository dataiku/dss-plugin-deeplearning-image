# -*- coding: utf-8 -*-
from dku_plugin_test_utils import dss_scenario

TEST_PROJECT_KEY = "PLUGINTESTDEEPLEARNINGIMAGE"


def add_integration_test(user_dss_clients, scenario_id):
    dss_scenario.run(user_dss_clients, project_key=TEST_PROJECT_KEY, scenario_id=scenario_id)


def test_run_tensorboard(user_dss_clients):
    """This test only checks if the webapp can be started without any error. There are no frontend checks as it
    is complicated to implement"""
    add_integration_test(user_dss_clients, "Webapp_-_Tensorboard")
