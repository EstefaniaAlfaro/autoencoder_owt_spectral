import os

EXTENSION = '.txt'
PATH_RESULTS = '../results'


class ShowMetricsResults:
    def __init__(self, results, title_file:str, case_option: str):
        self.results = results
        self.title_file = title_file
        self.case_option = case_option

    def selector_option(self):
        case = ''
        try:
            if self.case_option == 1:
                case = 'Predicted-abundance-map'
            elif self.case_option == 2:
                case = 'Predicted-end_members'
        except NameError:
            print('----Select a valid number----')
        return case

    def save_metrics_results_abundances(self):
        length_results = len(self.results)
        full_path = os.path.join(PATH_RESULTS, self.title_file)
        case_selector = self.selector_option()
        with open(full_path + EXTENSION, 'w') as filename:
            for index in range(length_results):
                filename.write('-----------*-----------')
                filename.write('\n')
                filename.write('Best root-mean-square-error:' + str(self.results[index][0]))
                filename.write('\n')
                filename.write('Ground-truth match:' + str(self.results[index][1][0]) + '\t' +
                               case_selector + ':' + str(self.results[index][1][-1]))
                filename.write('\n')
            filename.write('-----------*-----------')
        filename.close()

