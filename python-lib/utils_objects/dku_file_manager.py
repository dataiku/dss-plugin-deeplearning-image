from dataiku.customrecipe import get_input_names_for_role, get_output_names_for_role
import dataiku

class DkuFileManager:
    def get_file(self, side, type_, role):
        dku_func = get_input_names_for_role if side == 'input' else get_output_names_for_role
        dku_type = dataiku.Folder if type_ == 'folder' else dataiku.Dataset
        return dku_type(dku_func(role)[0])

    def get_input_folder(self, role):
        return self.get_file('input', 'folder', role)

    def get_output_folder(self, role):
        return self.get_file('output', 'folder', role)

    def get_input_dataset(self, role):
        return self.get_file('input', 'dataset', role)

    def get_output_dataset(self, role):
        return self.get_file('output', 'dataset', role)
