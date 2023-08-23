from LoadMatFileData import *
from SaveConfigurationFile import *
from ResultsDataVisualization import *

PATH_END_MEMBERS_RESULT = '../results'
configuration_file = 'configuration_visualization'


def results_visualization():
    parameters = SaveConfigurationFile(PATH_END_MEMBERS_RESULT, configuration_file).load_configurations_file()
    abundances_maps = LoadDataMatlabFile(PATH_END_MEMBERS_RESULT, parameters["abundances_maps"]).\
        load_matlab_file_extension()
    end_members = LoadDataMatlabFile(PATH_END_MEMBERS_RESULT, parameters["end_members"]).load_matlab_file_extension()
    visualization_results = VisualizationResults(end_members, abundances_maps)
    visualization_results.end_members_visualization()
    visualization_results.abundances_visualization(parameters["image_width"], parameters["image_height"])


results_visualization()