from LoadData import LoadDataset
from HypercubeRepresentation import HypercubeDataRepresentation
from VectorReshape import ImageReshapeTensor
from SplitDataset import *
from ExtractPatches import *
from StackDataPatches import *
from EnhancementInitializer import *
from ScalingDataset import *
from GeometricalsEndMembersMethods import *
from ReshapeHypercube import *
from PatchesDataset import *
from LoadGroundTruth import *


def preprocessing_stage_spectral(data_path, ground_truth_path, data_target, ground_truth_target,
                                 ground_truth_abundances_label, image_width, image_height, end_members_number,
                                 case_option_abundances_maps, batch_size, rows_number, columns_number,
                                 batch_size_depth, patch_size, image_size, training_percentage, testing_percentage,
                                 validation_percentage, patch_size_model,
                                     number_patches, batch_size_model):
    load_data = LoadDataset(data_path, ground_truth_path, data_target, ground_truth_target,
                            ground_truth_abundances_label)
    dataset, ground_truth_data, ground_truth_abundances, bands_number = load_data.data_number_images()
    abundances_ground_truth = LoadGroundTruth(image_width, image_height).\
        load_abundances_maps_ground_truth(ground_truth_abundances)
    dataset_scaling = ScalingDataset(dataset).normalization_max_abs()
    hypercube_dataset = HypercubeDataRepresentation(dataset)
    # hypercube_dataset = HypercubeDataRepresentation(dataset_scaling)
    hypercube_data = hypercube_dataset.reshape_hypercube(image_width, image_height)
    # hypercube_data = hypercube_data.astype('float32') / 255
    hypercube_data_patches = TrainingPatchesData(hypercube_data, patch_size_model, number_patches,
                                                 batch_size_model).data_patches()
    end_member_extraction = EndMembersExtractionGeometrical(hypercube_data, end_members_number,
                                                            case_option_abundances_maps)
    end_member_extraction_n_findr, abundances_estimation = end_member_extraction.end_members_extraction_n_findr()
    end_member_extraction_fippi, abundances_estimation_fippi = end_member_extraction. \
        end_member_extraction_fast_iterative_pure_pixel_index()
    tensor_image = ImageReshapeTensor(hypercube_data, batch_size, rows_number, columns_number, batch_size_depth)
    tensor_image_reshape = tensor_image.batch_size_selection()
    patches = ExtractPatches(patch_size)(tf.convert_to_tensor([hypercube_data]))
    patches_abundances_maps = ExtractPatches(patch_size)(tf.convert_to_tensor([abundances_estimation]))
    list_patch_image_stack, stack_patches_data = StackPatches(patches, patch_size, batch_size_depth). \
        stack_patches_dataset()
    list_patch_abundances_stack, stack_patches_abundances = StackPatches(patches_abundances_maps, patch_size,
                                                                         end_members_number).stack_patches_dataset()
    reshape_abundances_maps = ReshapeHypercube(stack_patches_abundances).reshape_hypercube_dataframe()
    reshape_abundances_maps_tensor = tf.convert_to_tensor(reshape_abundances_maps.values)
    training_data, testing_data, validation_data, random_position = SplitData(stack_patches_data, training_percentage,
                                                                              testing_percentage,
                                                                              validation_percentage).split_dataset()
    list_patch_image_stack_transpose = tf.transpose(list_patch_image_stack[0])
    training_data_patches, testing_data_patches, validation_data_patches, random_position_patches = SplitData(
        list_patch_image_stack_transpose, training_percentage, testing_percentage, validation_percentage). \
        split_dataset()
    enhancement_initializer_energy = EnhancementInitializer(training_data).enhancement_energy()
    enhancement_initializer_mean = EnhancementInitializer(training_data_patches).enhancement_mean()
    enhancement_initializer_standard_deviation = EnhancementInitializer(training_data_patches). \
        enhancement_standard_deviation()
    return stack_patches_data, enhancement_initializer_energy, enhancement_initializer_mean, \
           enhancement_initializer_standard_deviation, reshape_abundances_maps_tensor, \
           training_data, testing_data, validation_data, random_position, hypercube_data_patches,\
           abundances_ground_truth, ground_truth_data, hypercube_data
