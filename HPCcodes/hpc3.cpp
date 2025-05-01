//%%writefile MinMaxAvgSum.cpp
#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

int main() {
    vector<double> arr(10);
    omp_set_num_threads(4);

    for (int i = 0; i < 10; ++i) arr[i] = 2.0 + i;

    // Print active threads
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        cout << "Thread " << tid << " is active\n";
    }

    double max_val = 0.0, min_val = 1000.0, sum = 0.0;

    #pragma omp parallel for reduction(max:max_val) reduction(min:min_val) reduction(+:sum)
    for (int i = 0; i < 10; ++i) {
        int tid = omp_get_thread_num();
        cout << "Thread " << tid << " working on i = " << i << "\n";
        if (arr[i] > max_val) max_val = arr[i];
        if (arr[i] < min_val) min_val = arr[i];
        sum += arr[i];
    }

    double avg = sum / arr.size();
    cout << "\nMax = " << max_val << "\nMin = " << min_val
              << "\nSum = " << sum << "\nAvg = " << avg << endl;

    return 0;
}

//!g++ -fopenmp  MinMaxAvgSum.cpp -o MinMaxAvgSum
//!./MinMaxAvgSum
