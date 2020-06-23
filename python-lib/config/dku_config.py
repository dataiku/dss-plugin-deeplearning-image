from dataiku.customrecipe import get_input_names_for_role, get_output_names_for_role, get_recipe_config
import dl_image_toolbox_utils as utils
import dataiku


class DkuConfig:
    @utils.log_func(txt='config loading')
    def load(self):
        self._load_input_output()
        self._load_recipe_param()
        utils.display_gpu_device()

    def _load_recipe_param(self):
        self.recipe_config = get_recipe_config()

        should_use_gpu = self.recipe_config.get('should_use_gpu', False)
        self.gpu_options = utils.load_gpu_options(
            should_use_gpu, self.recipe_config['list_gpu'], self.recipe_config['gpu_allocation'])
        self.n_gpu = self.gpu_options.get("n_gpu", 0)
        self.use_gpu = should_use_gpu and self.n_gpu > 1

    def _load_input(self):
        image_folder_input_name = get_input_names_for_role('image_folder')[0]
        self.image_folder = dataiku.Folder(image_folder_input_name)

        model_folder_input_name = get_input_names_for_role('model_folder')[0]
        self.model_folder = dataiku.Folder(model_folder_input_name)

    def _load_output(self):
        output_type = dataiku.Folder if isinstance(self, RetrainConfig) else dataiku.Dataset
        output_name = get_output_names_for_role(self.output_role)[0]
        self.output_model_folder = output_type(output_name)

    def _load_input_output(self):
        self._load_input()
        self._load_output()



