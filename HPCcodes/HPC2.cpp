%%writefile BubbleMerge.cpp

#include <iostream>
#include <vector>
#include <omp.h>
#include <cstdlib>
using namespace std;

// Sort implementations
void bubbleSort(vector<int>& data, bool parallel) {
    for (int i = 0; i < data.size() - 1; i++) {
        #pragma omp parallel for if(parallel)
        for (int j = 0; j < data.size() - i - 1; j++) {
            if (data[j] > data[j + 1]) swap(data[j], data[j + 1]);
        }
    }
}

void mergeSort(vector<int>& data, int left, int right, bool parallel) {
    if (left >= right) return;

    int mid = (left + right) / 2;

    if (parallel) {
        #pragma omp parallel sections
        {
            #pragma omp section
            mergeSort(data, left, mid, true);
            #pragma omp section
            mergeSort(data, mid + 1, right, true);
        }
    } else {
        mergeSort(data, left, mid, false);
        mergeSort(data, mid + 1, right, false);
    }

    // Merge
    vector<int> temp(right - left + 1);
    int i = left, j = mid + 1, k = 0;
    while (i <= mid && j <= right)
        temp[k++] = (data[i] <= data[j]) ? data[i++] : data[j++];
    while (i <= mid) temp[k++] = data[i++];
    while (j <= right) temp[k++] = data[j++];
    for (k = 0; k < temp.size(); k++)
        data[left + k] = temp[k];
}

// Run benchmark test
void benchmark(string name, vector<int> data, bool isBubble) {
    vector<int> seqData = data, parData = data;
    double seqTime, parTime;

    // Sequential sort
    seqTime = omp_get_wtime();
    if (isBubble) bubbleSort(seqData, false);
    else mergeSort(seqData, 0, seqData.size() - 1, false);
    seqTime = omp_get_wtime() - seqTime;

    // Parallel sort
    parTime = omp_get_wtime();
    if (isBubble) bubbleSort(parData, true);
    else mergeSort(parData, 0, parData.size() - 1, true);
    parTime = omp_get_wtime() - parTime;

    cout << name << " Sort (seq): " << seqTime << "s, (par): " << parTime << "s\n";
}

int main() {
    const int SIZE = 10000;

    // Generate random data
    vector<int> data(SIZE);
    for (int& element : data) element = rand() % 10000;

    // Run benchmarks
    benchmark("Bubble", data, true);

    // Generate new data for merge sort
    for (int& element : data) element = rand() % 10000;
    benchmark("Merge", data, false);

    return 0;
}

//!g++ -fopenmp BubbleMerge.cpp -o run
//!./run
