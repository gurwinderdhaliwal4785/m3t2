#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

// Function to perform quick sort
void quick_sort(int *array, int left, int right) {
    int i, j, pivot, temp;

    if (left < right) {
        pivot = left;
        i = left;
        j = right;

        // Partitioning step of quicksort
        while (i < j) {
            while (array[i] <= array[pivot] && i <= right)
                i++;
            while (array[j] > array[pivot])
                j--;
            if (i < j) {
                temp = array[i];
                array[i] = array[j];
                array[j] = temp;
            }
        }

        // Swap pivot with element at position j
        temp = array[pivot];
        array[pivot] = array[j];
        array[j] = temp;

        // Recursive calls for left and right sub-arrays
        quick_sort(array, left, j - 1);
        quick_sort(array, j + 1, right);
    }
}

int main(int argc, char **argv) {
    int rank, size, *array, *chunk, a, x;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Process 0 handles input and sorting
    if (rank == 0) {
        printf("Enter the number of elements in the array: ");
        fflush(stdout);
        scanf("%d", &a);

        array = (int *) malloc(a * sizeof(int));
        printf("Enter the elements of the array: ");
        fflush(stdout);
        for (x = 0; x < a; x++)
            scanf("%d", &array[x]);

        // Perform initial quicksort
        quick_sort(array, 0, a - 1);
    }

    // Broadcast the array size to all processes
    MPI_Bcast(&a, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate memory for chunk to store sub-array
    chunk = (int *) malloc(a / size * sizeof(int));

    // Scatter the array to all processes
    MPI_Scatter(array, a / size, MPI_INT, chunk, a / size, MPI_INT, 0, MPI_COMM_WORLD);

    // Perform quicksort on the chunk
    clock_t start_time = clock();
    quick_sort(chunk, 0, a / size - 1);
    clock_t end_time = clock();

    // Calculate and print execution time
    double duration = (double)(end_time - start_time) / ((double)CLOCKS_PER_SEC / 1000000);
    if (rank == 0) {
        printf("Execution time is %f microseconds\n", duration);
    }

    // Perform merging of sorted chunks
    for (int order = 1; order < size; order *= 2) {
        if (rank % (2 * order) != 0) {
            // Send chunk to the previous process
            MPI_Send(chunk, a / size, MPI_INT, rank - order, 0, MPI_COMM_WORLD);
            break;
        }

        int recv_size = (rank + order < size) ? a / size : a - (rank + order) * (a / size);
        int *other = (int *) malloc(recv_size * sizeof(int));

        // Receive chunk from the next process
        MPI_Recv(other, recv_size, MPI_INT, rank + order, 0, MPI_COMM_WORLD, &status);

        // Merge the received chunk with the current chunk
        int *temp = (int *) malloc((a / size + recv_size) * sizeof(int));
        int i = 0, j = 0, k = 0;
        while (i < a / size && j < recv_size) {
            if (chunk[i] < other[j])
                temp[k++] = chunk[i++];
            else
                temp[k++] = other[j++];
        }
        while (i < a / size)
            temp[k++] = chunk[i++];
        while (j < recv_size)
            temp[k++] = other[j++];

        // Free memory and update chunk pointer
        free(other);
        free(chunk);
        chunk = temp;
    }

    // Gather sorted chunks to process 0
    if (rank == 0) {
        array = (int *) malloc(a * sizeof(int));
    }
    MPI_Gather(chunk, a / size, MPI_INT, array, a / size, MPI_INT, 0, MPI_COMM_WORLD);

    // Process 0 prints the sorted array
    if (rank == 0) {
        printf("Sorted array: ");
        for (x = 0; x < a; x++) {
            printf("%d ", array[x]);
        }
        printf("\n");
        fflush(stdout);
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
} 