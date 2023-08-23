import pysptools.eea as eea
import pysptools.abundance_maps as amap


class EndMembersExtractionGeometrical:
    def __init__(self, preprocessed_hypercube, end_members_number, case_option_abundances_maps):
        self.preprocessed_hypercube = preprocessed_hypercube
        self.end_members_number = end_members_number
        self.case_option_abundances_maps = case_option_abundances_maps

    def abundances_maps_unconstrained_least_squared(self, end_member_extraction):
        abundances_map = amap.UCLS()
        abundances_map_ucls = abundances_map.map(self.preprocessed_hypercube, end_member_extraction, normalize=True)
        return abundances_map_ucls

    def abundances_maps_non_negative_constrained_least_squared(self, end_member_extraction):
        abundances_map = amap.NNLS()
        abundances_map_nnls = abundances_map.map(self.preprocessed_hypercube, end_member_extraction, normalize=True)
        return abundances_map_nnls

    def abundances_maps_fully_constrained_least_square(self, end_member_extraction):
        abundances_map = amap.FCLS()
        abundances_map_fcls = abundances_map.map(self.preprocessed_hypercube, end_member_extraction, normalize=True)
        return abundances_map_fcls

    def switch_selector_abundances_maps(self, end_member_extraction):
        abundances_estimation = []
        try:
            if self.case_option_abundances_maps == 0:
                abundances_estimation = self.abundances_maps_unconstrained_least_squared(end_member_extraction)
            elif self.case_option_abundances_maps == 1:
                abundances_estimation = self.abundances_maps_non_negative_constrained_least_squared(
                    end_member_extraction)
            elif self.case_option_abundances_maps == 2:
                abundances_estimation = self.abundances_maps_fully_constrained_least_square(end_member_extraction)
        except NameError:
            print('Please enter a valid option')
        return abundances_estimation

    def end_members_extraction_n_findr(self):
        end_member_extraction_n_findr = eea.NFINDR()
        end_member_extraction = end_member_extraction_n_findr.extract(self.preprocessed_hypercube,
                                                                      self.end_members_number,
                                                                      normalize=False, ATGP_init=True)
        abundances_estimation = self.switch_selector_abundances_maps(end_member_extraction)
        return end_member_extraction, abundances_estimation

    def end_member_extraction_fast_iterative_pure_pixel_index(self):
        end_member_extraction_fippi = eea.FIPPI()
        end_member_extraction = end_member_extraction_fippi.extract(self.preprocessed_hypercube,
                                                                    self.end_members_number,
                                                                    normalize=False)
        abundances_estimation = self.switch_selector_abundances_maps(end_member_extraction)
        return end_member_extraction, abundances_estimation

    def end_member_extraction_pixel_purity_index(self):
        end_member_extraction_ppi = eea.PPI()
        end_member_extraction = end_member_extraction_ppi.extract(self.preprocessed_hypercube,
                                                                  self.end_members_number,
                                                                  normalize=False)
        abundances_estimation = self.switch_selector_abundances_maps(end_member_extraction)
        return end_member_extraction, abundances_estimation

