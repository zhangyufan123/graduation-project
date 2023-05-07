import numpy as np
import collections

class Hungarian():
    """
    """

    def __init__(self, input_matrix=None, is_profit_matrix=False):

        if input_matrix is not None:
            my_matrix = np.array(input_matrix)
            self._input_matrix = np.array(input_matrix)
            self._maxColumn = my_matrix.shape[1]
            self._maxRow = my_matrix.shape[0]

            matrix_size = max(self._maxColumn, self._maxRow)
            pad_columns = matrix_size - self._maxRow
            pad_rows = matrix_size - self._maxColumn
            my_matrix = np.pad(my_matrix, ((0, pad_columns), (0, pad_rows)), 'constant', constant_values=(0))

            if is_profit_matrix:
                my_matrix = self.make_cost_matrix(my_matrix)

            self._cost_matrix = my_matrix
            self._size = len(my_matrix)
            self._shape = my_matrix.shape

            self._results = []
            self._totalPotential = 0
        else:
            self._cost_matrix = None

    def make_cost_matrix(self, profit_matrix):

        matrix_shape = profit_matrix.shape
        offset_matrix = np.ones(matrix_shape, dtype=int) * profit_matrix.max()
        cost_matrix = offset_matrix - profit_matrix
        return cost_matrix

    def get_results(self):

        return self._results

    def calculate(self):

        result_matrix = self._cost_matrix.copy()

        for index, row in enumerate(result_matrix):
            result_matrix[index] -= row.min()

        for index, column in enumerate(result_matrix.T):
            result_matrix[:, index] -= column.min()

        total_covered = 0
        while total_covered < self._size:

            cover_zeros = CoverZeros(result_matrix)
            single_zero_pos_list = cover_zeros.calculate()
            covered_rows = cover_zeros.get_covered_rows()
            covered_columns = cover_zeros.get_covered_columns()
            total_covered = len(covered_rows) + len(covered_columns)

            if total_covered < self._size:
                result_matrix = self._adjust_matrix_by_min_uncovered_num(result_matrix, covered_rows, covered_columns)

        self._results = single_zero_pos_list
        value = 0
        #print(single_zero_pos_list)
        for row, column in single_zero_pos_list:
            #print(row, column)
            value += self._input_matrix[row, column]
        self._totalPotential = value

    def get_total_potential(self):
        return self._totalPotential

    def _adjust_matrix_by_min_uncovered_num(self, result_matrix, covered_rows, covered_columns):
        adjusted_matrix = result_matrix
        elements = []
        for row_index, row in enumerate(result_matrix):
            if row_index not in covered_rows:
                for index, element in enumerate(row):
                    if index not in covered_columns:
                        elements.append(element)
        min_uncovered_num = min(elements)
        for row_index, row in enumerate(result_matrix):
            if row_index not in covered_rows:
                for index, element in enumerate(row):
                    if index not in covered_columns:
                        adjusted_matrix[row_index, index] -= min_uncovered_num

        for row_ in covered_rows:
            for col_ in covered_columns:
                # print((row_,col_))
                adjusted_matrix[row_, col_] += min_uncovered_num

        return adjusted_matrix


class CoverZeros():

    def __init__(self, matrix):
        self._zero_locations = (matrix == 0)
        self._zero_locations_copy = self._zero_locations.copy()
        self._shape = matrix.shape

        self._covered_rows = []
        self._covered_columns = []

    def get_covered_rows(self):
        return self._covered_rows

    def get_covered_columns(self):
        return self._covered_columns

    def row_scan(self, marked_zeros):
        min_row_zero_nums = [9999999, -1]
        for index, row in enumerate(self._zero_locations_copy):  # index为行号
            row_zero_nums = collections.Counter(row)[True]
            if row_zero_nums < min_row_zero_nums[0] and row_zero_nums != 0:

                min_row_zero_nums = [row_zero_nums, index]

        row_min = self._zero_locations_copy[min_row_zero_nums[1], :]
        row_indices, = np.where(row_min)
        marked_zeros.append((min_row_zero_nums[1], row_indices[0]))
        self._zero_locations_copy[:, row_indices[0]] = np.array([False for _ in range(self._shape[0])])
        self._zero_locations_copy[min_row_zero_nums[1], :] = np.array([False for _ in range(self._shape[0])])

    def calculate(self):
        ticked_row = []
        ticked_col = []
        marked_zeros = []
        while True:
            if True not in self._zero_locations_copy:
                break
            self.row_scan(marked_zeros)

        independent_zero_row_list = [pos[0] for pos in marked_zeros]
        ticked_row = list(set(range(self._shape[0])) - set(independent_zero_row_list))
        TICK_FLAG = True
        while TICK_FLAG:
            TICK_FLAG = False
            for row in ticked_row:
                row_array = self._zero_locations[row, :]
                for i in range(len(row_array)):
                    if row_array[i] == True and i not in ticked_col:
                        ticked_col.append(i)
                        TICK_FLAG = True

            for row, col in marked_zeros:
                if col in ticked_col and row not in ticked_row:
                    ticked_row.append(row)
                    FLAG = True
        self._covered_rows = list(set(range(self._shape[0])) - set(ticked_row))
        self._covered_columns = ticked_col

        return marked_zeros


if __name__ == '__main__':
    # 以下为3个测试用例
    '''
    cost_matrix = [
        [4, 2, 8],
        [4, 3, 7],
        [3, 1, 6]]
    '''
    cost_matrix = [[99999, 0.32999352783032476, 0.020961390057103846], [0.3941583562578619, 0.001594596227283529, 0.33062591515178674]]
    hungarian = Hungarian(cost_matrix)
    print('calculating...')
    hungarian.calculate()
    print("Expected value:\t\t12")
    print("Calculated value:\t", hungarian.get_total_potential())  # = 12
    print("Expected results:\n\t[(0, 1), (1, 0), (2, 2)]")
    print("Results:\n\t", hungarian.get_results())
    print("-" * 80)

    profit_matrix = [
        [62, 75, 80, 93, 95, 97],
        [75, 80, 82, 85, 71, 97],
        [80, 75, 81, 98, 90, 97],
        [78, 82, 84, 80, 50, 98],
        [90, 85, 85, 80, 85, 99],
        [65, 75, 80, 75, 68, 96]]

    hungarian = Hungarian(profit_matrix, is_profit_matrix=True)
    hungarian.calculate()
    print("Expected value:\t\t543")
    print("Calculated value:\t", hungarian.get_total_potential())  # = 543
    print("Expected results:\n\t[(0, 4), (2, 3), (5, 5), (4, 0), (1, 1), (3, 2)]")
    print("Results:\n\t", hungarian.get_results())
    print("-" * 80)
    profit_matrix = [
        [62, 75, 80, 93, 0, 97],
        [75, 0, 82, 85, 71, 97],
        [80, 75, 81, 0, 90, 97],
        [78, 82, 0, 80, 50, 98],
        [0, 85, 85, 80, 85, 99],
        [65, 75, 80, 75, 68, 0]]
    hungarian = Hungarian(profit_matrix, is_profit_matrix=True)
    hungarian.calculate()
    print("Expected value:\t\t523")
    print("Calculated value:\t", hungarian.get_total_potential())  # = 523
    print("Expected results:\n\t[(0, 3), (2, 4), (3, 0), (5, 2), (1, 5), (4, 1)]")
    print("Results:\n\t", hungarian.get_results())
    print("-" * 80)
