import pytest
import logging

from dku_plugin_test_utils import dss_scenario


pytestmark = pytest.mark.usefixtures("plugin", "dss_target")


test_kwargs = {
    "user": "user1",
    "project_key": "PLUGINTESTDEEPLEARNINGIMAGE",
    "logger": logging.getLogger("dss-plugin-test.deeplearning-image.test_scenario"),
}


def test_run_tensorboard_start(user_clients):
    test_kwargs["client"] = user_clients[test_kwargs["user"]]
    dss_scenario.run(scenario_id="Webapp_-_Tensorboard", **test_kwargs)


def test_extract_recipe(user_clients):
    test_kwargs["client"] = user_clients[test_kwargs["user"]]
    dss_scenario.run(scenario_id="Recipe_-_Extract_features", **test_kwargs)
