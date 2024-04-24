#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
using namespace std;

void printMatrix(const vector<vector<double>> &matrix)
{
    for (int i = 0; i < matrix.size(); ++i)
    {
        const auto &row = matrix[i];
        for (double elem : row)
        {
            cout << elem << " ";
        }
        cout << endl;
    }
    cout << endl;
}

vector<vector<double>> gaussJordanInverse(const vector<vector<double>> &matrix)
{
	int n = matrix.size();
	vector<vector<double>> augmented(n, vector<double>(2 * n));
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            augmented[i][j] = matrix[i][j];
        }
        augmented[i][i + n] = 1.0;
    }
    {
        for (int i = 0; i < n; i++)
        {
            int pivot = i;
            {
                for (int j = i + 1; j < n; j++)
                {
                    if (fabs(augmented[j][i]) > fabs(augmented[pivot][i]))
                    {
                        pivot = j;
                    }
                }
                swap(augmented[i], augmented[pivot]);
            }
            double pivotValue = augmented[i][i];
            for (int j = 0; j < 2 * n; j++)
            {
                augmented[i][j] /= pivotValue;
            }
            {
                for (int j = 0; j < n; j++)
                {
                    if (j != i)
                    {
                        double factor = augmented[j][i];
                        for (int k = 0; k < 2 * n; k++)
                        {
                            augmented[j][k] -= factor * augmented[i][k];
                        }
                    }
                }
            }
        }
    }
	vector<vector<double>> inverse(n, vector<double>(n));
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			inverse[i][j] = augmented[i][j + n];
		}
	}

	return inverse;
}

int main()
{
	int n;
	cin >> n;

	vector<vector<double>> matrix(n, vector<double>(n));
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			cin >> matrix[i][j];
		}
	}
	double start_time = omp_get_wtime();
	vector<vector<double>> inverse = gaussJordanInverse(matrix);
	double end_time = omp_get_wtime();
	double cpu_time = omp_get_wtick() * (end_time - start_time);
	cout << "Execution time: " << end_time - start_time << " seconds" << endl;
	printMatrix(inverse);

	return 0;
}
