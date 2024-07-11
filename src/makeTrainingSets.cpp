#include <cmath>
#include <iostream>
#include <random>

int main(int argc, char* argv[]) {
    std::cout << "topology: " << argv[1] << " " << argv[2] << " " << argv[3]
              << std::endl;
    for (int i = 2000; i >= 0; --i) {
        int n1 = (int)(2.0 * random() / double(RAND_MAX));
        int n2 = (int)(2.0 * random() / double(RAND_MAX));
        int t = n1 ^ n2;
        std::cout << "in: " << n1 << ".0 " << n2 << ".0 " << std::endl;
        std::cout << "out: " << t << ".0 " << std::endl;
    }
}