#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>

cv::Mat svertka(const cv::Mat& img, const std::vector<std::vector<int>>& kernel) {
    int kernel_size = kernel.size();
    int x_start = kernel_size / 2;
    int y_start = kernel_size / 2;

    cv::Mat matr = cv::Mat::zeros(img.size(), img.type());

    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            matr.at<uchar>(i, j) = img.at<uchar>(i, j);
        }
    }

    cv::Mat result = matr.clone();

    for (int i = x_start; i < matr.rows - x_start; ++i) {
        for (int j = y_start; j < matr.cols - y_start; ++j) {
            int val = 0;
            for (int k = -(kernel_size / 2); k <= kernel_size / 2; ++k) {
                for (int l = -(kernel_size / 2); l <= kernel_size / 2; ++l) {
                    val += img.at<uchar>(i + k, j + l) * kernel[k + (kernel_size / 2)][l + (kernel_size / 2)];
                }
            }
            result.at<uchar>(i, j) = static_cast<uchar>(val);
        }
    }

    return result;
}

int get_angle_number(int x, int y) {
    double tg = (x != 0) ? static_cast<double>(y) / x : 999;

    if (x < 0) {
        if (y < 0) {
            if (tg > 2.414) {
                return 0;
            }
            else if (tg < 0.414) {
                return 6;
            }
            else if (tg <= 2.414) {
                return 7;
            }
        }
        else {
            if (tg < -2.414) {
                return 4;
            }
            else if (tg < -0.414) {
                return 5;
            }
            else if (tg >= -0.414) {
                return 6;
            }
        }
    }
    else {
        if (y < 0) {
            if (tg < -2.414) {
                return 0;
            }
            else if (tg < -0.414) {
                return 1;
            }
            else if (tg >= -0.414) {
                return 2;
            }
        }
        else {
            if (tg < 0.414) {
                return 2;
            }
            else if (tg < 2.414) {
                return 3;
            }
            else if (tg >= 2.414) {
                return 4;
            }
        }
    }

    return 0; // Default case
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <image_path> <standard_deviation> <kernel_size> <bound_path>" << std::endl;
        return 1;
    }

    std::string path = argv[1];
    double standard_deviation = std::stod(argv[2]);
    int kernel_size = std::stoi(argv[3]);
    int bound_path = std::stoi(argv[4]);

    cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);

    if (img.empty()) {
        std::cerr << "Error: Unable to read image." << std::endl;
        return 1;
    }

    // 1
    cv::Mat imgBlurByCV2;
    cv::GaussianBlur(img, imgBlurByCV2, cv::Size(kernel_size, kernel_size), standard_deviation);

    cv::imshow(path, imgBlurByCV2);

    // 2
    std::vector<std::vector<int>> Gx = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
    std::vector<std::vector<int>> Gy = { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1} };

    cv::Mat img_Gx = svertka(img, Gx);
    cv::Mat img_Gy = svertka(img, Gy);

    cv::Mat matr_gradient = cv::Mat::zeros(img.size(), img.type());

    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            matr_gradient.at<uchar>(i, j) = img.at<uchar>(i, j);
        }
    }

    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            matr_gradient.at<uchar>(i, j) = static_cast<uchar>(std::sqrt(std::pow(img_Gx.at<uchar>(i, j), 2) +
                std::pow(img_Gy.at<uchar>(i, j), 2)));
        }
    }

    // 3
    cv::Mat img_angles = img.clone();
    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            img_angles.at<uchar>(i, j) = get_angle_number(img_Gx.at<uchar>(i, j), img_Gy.at<uchar>(i, j));
        }
    }

    // Rest of your code...

    while (true) {
        if (cv::waitKey(1) & 0xFF == 'q') {
            break;
        }
    }

    return 0;
}
