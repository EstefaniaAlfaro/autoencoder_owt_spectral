from GeometricalsEndMembersMethods import *
from PatchesDataset import *
from EnhancementInitializer import *


def preprocessing_image_hsi2(parameters, hypercube_data, number_patches, batch_size_model):
    end_member_extraction_hsi2 = EndMembersExtractionGeometrical(hypercube_data, parameters["end_members_number"],
                                                                 parameters["case_option_abundances"])
    end_member_extraction_n_findr_hsi2, abundances_estimation_hsi2 = end_member_extraction_hsi2 \
        .end_members_extraction_n_findr()
    end_member_extraction_fippi_hsi2, abundances_estimation_fippi_hsi2 = end_member_extraction_hsi2. \
        end_member_extraction_fast_iterative_pure_pixel_index()
    hypercube_data_patches = TrainingPatchesData(hypercube_data, parameters["patch_size_model"], number_patches,
                                                 batch_size_model).data_patches()
    enhancement_hsi2_initializer_energy = EnhancementInitializer(hypercube_data_patches).enhancement_energy()
    enhancement_hsi2_initializer_mean = EnhancementInitializer(hypercube_data_patches).enhancement_mean()
    enhancement_hsi2_initializer_standard_deviation = EnhancementInitializer(hypercube_data_patches)\
        .enhancement_standard_deviation()
    return hypercube_data_patches, end_member_extraction_n_findr_hsi2, abundances_estimation_hsi2, \
           end_member_extraction_fippi_hsi2, abundances_estimation_fippi_hsi2, enhancement_hsi2_initializer_energy,\
           enhancement_hsi2_initializer_mean, enhancement_hsi2_initializer_standard_deviation

