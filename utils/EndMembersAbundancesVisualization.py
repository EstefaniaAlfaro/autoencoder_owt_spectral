from matplotlib import pyplot as plt


class EndMembersAbundancesVisualization:
    def __init__(self, end_members, abundances_maps):
        self.end_members = end_members
        self.abundances_maps = abundances_maps

    def end_members_visualization(self):
        plt.figure(figsize=(8, 5))
        colors = ['r', 'g', 'b', 'y', 'k', 'm', 'k']
        length_end_members = len(self.end_members)
        for index in range(length_end_members):
            plt.plot(self.end_members[index, :], colors[index] + '--')
            plt.grid(color='k', linestyle='-', linewidth=0.6)
            plt.title('EndMembers')
            plt.xlabel('Wavelengths')
            plt.ylabel('Brightness')

    def abundances_visualization(self):
        plt.figure(figsize=(5, 5))
        length_abundances = self.abundances_maps.shape[-1]
        rows_number = round(length_abundances/2)
        for index in range(length_abundances):
            plt.subplot(rows_number, rows_number, index + 1)
            plt.imshow(self.abundances_maps[:, :, index], cmap='jet')
        plt.suptitle('Abundances maps')
        plt.show()


