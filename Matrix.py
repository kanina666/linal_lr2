from typing import List

class Matrix:
    def __init__(self, data: List[List[float]]):
        self.data = data
        self.n = len(data)
        self.m = len(data[0]) if data else 0


    def __getitem__(self, idx):
        return self.data[idx]

    def __setitem__(self, idx, value):
        self.data[idx] = value

    def copy(self):
        return Matrix([row[:] for row in self.data])

    def shape(self):
        return self.n, self.m

    def transpose(self) -> 'Matrix':
        transposed_data = [[self.data[j][i] for j in range(self.n)] for i in range(self.m)]
        return Matrix(transposed_data)

    def __matmul__(self, other: 'Matrix') -> 'Matrix':
        if self.m != other.n:
            raise ValueError("Матрицы не подходят для умножения")

        result = []
        for i in range(self.n):
            row = []
            for j in range(other.m):
                s = sum(self.data[i][k] * other.data[k][j] for k in range(self.m))
                row.append(s)
            result.append(row)
        return Matrix(result)