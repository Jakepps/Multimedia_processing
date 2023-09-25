#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
#include <string>
#include <map>
#include <numeric>
#include <fstream>
typedef unsigned long long ll;
using namespace std;

vector<vector<int>> MyGaussianBlur(vector<vector<int>> img, int kernel_size, double standard_deviation) {
    vector<vector<double>> kernel = vector<vector<double>>(kernel_size, vector<double>(kernel_size));
    int a = (kernel_size + 1);
    int b = a;

    for (int i = 0; i < kernel_size; i++) {
        for (int j = 0; j < kernel_size; j++) {
            kernel[i][j] = gauss(i, j, standard_deviation, a, b);
        }
    }
    double sum = 0;
    for (int i = 0; i < kernel_size; i++) {
        for (int j = 0; j < kernel_size; j++) {
            sum += kernel[i][j];
        }
    }
    for (int i = 0; i < kernel_size; i++) {
        for (int j = 0; j < kernel_size; j++) {
            kernel[i][j] /= sum;
        }
    }

    int x_start = kernel_size;
    int y_start = kernel_size;
    for (int i = x_start; i < img.size() - x_start; i++) {
        for (int j = y_start; j < img[i].size() - y_start; j++) {
            double val = 0;
            for (int k = -(kernel_size/2); k < kernel_size/2; k++) {
                for (int l = -(kernel_size / 2); l < kernel_size / 2; l++) {
                    val += img[i + k][j + l] * kernel[k + (kernel_size / 2)][l + (kernel_size / 2)];
                }
            }
            img[i][j] = val;
        }
    }
    return img;
}

double gauss(double x, double  y, double  omega, double  a, double  b) {
    double omegaIn2 = 2 * pow(omega, 2);
    double m1 = 1 / (3.14 * omegaIn2);
    double m2 = exp(-(pow((x - a),2) + pow((y - b), 2)) / omegaIn2);
    return m1 * m2;
}



int main() {
    MyGaussianBlur({ {50, 50, 50, 60, 60, 10, 60, 60, 50, 50, 50},{50, 50, 50, 60, 60, 10, 60, 60, 50, 50, 50},{50, 50, 50, 60, 60, 10, 60, 60, 50, 50, 50},{50, 50, 50, 60, 60, 10, 60, 60, 50, 50, 50},{50, 50, 50, 60, 60, 10, 60, 60, 50, 50, 50},{50, 50, 50, 60, 60, 10, 60, 60, 50, 50, 50},{50, 50, 50, 60, 60, 10, 60, 60, 50, 50, 50},{50, 50, 50, 60, 60, 10, 60, 60, 50, 50, 50}, }, 3, 100);
}
