# SE3082 – Parallel Computing
## Lab Sheet 10 – CUDA (Google Colab Edition)

**Student ID:** IT24610803  
**Module:** SE3082 Parallel Computing  
**Platform:** Google Colab (NVIDIA Tesla T4 GPU)  
**Compiler:** NVIDIA CUDA Compiler (`nvcc`)  

---

## 📌 Repository Contents

| File | Description |
| :--- | :--- |
| **`SE3082_Lab10_IT24610803.ipynb`** | Main Google Colab Jupyter Notebook containing all exercises, code cells, compilation/execution commands, and verification. |
| **`SE3082_Lab10_IT24610803_Report.pdf`** | Official Lab 10 Output & Verification Report. |
| **`ex1_timed.cu`** | Exercise 1: Timing a GPU program with `cpuSecond()` and `cudaDeviceSynchronize()`. |
| **`ex2_check.cu`** | Exercise 2: Catching errors with the `CHECK()` macro. |
| **`ex3_picture.cu`** | Exercise 3: Scaling an $800 \times 600$ picture with a 2D grid (`PictureKernel`) and saturation cap at 255. |
| **`ex4_volume.cu`** | Exercise 4: Multiplying two 3D matrices element-by-element ($500 \times 500 \times 500 = 125,000,000$ elements) with 3D grid and boundary guards. |

---

## 🚀 How to Run in Google Colab

1. Open [Google Colab](https://colab.research.google.com).
2. Set Runtime: **Runtime &rarr; Change runtime type &rarr; Hardware accelerator: T4 GPU &rarr; Save**.
3. Open or upload `SE3082_Lab10_IT24610803.ipynb`.
4. Run cells sequentially or click **Runtime &rarr; Run all**.

### Compilation & Execution Pattern
```bash
# Write code to file
%%writefile name.cu
# ... code ...

# Compile using nvcc
!nvcc -arch=native -o name.o name.cu

# Execute binary
!./name.o
```

---

## 📊 Summary of Exercises & Verification

### Exercise 1: Timing a GPU Program
- Benchmarked kernel execution using `cpuSecond()` and `cudaDeviceSynchronize()`.
- Demonstrated that omitting synchronization stops the timer prematurely before GPU kernel completion.
- Evaluated PCIe transfer overhead ($10\times$ to $20\times$ slower than GPU computation for memory-bound tasks).

### Exercise 2: Error Handling with `CHECK()`
- Robust error macro wrapping all CUDA API calls (`cudaMalloc`, `cudaMemcpy`, `cudaFree`, `cudaGetLastError`, `cudaDeviceSynchronize`).
- Validated error detection with deliberate edge cases:
  - Exceeding block limit (2048 threads): `code: 9, reason: invalid configuration argument`
  - Excessive memory allocation: `code: 2, reason: out of memory`

### Exercise 3: 2D Picture Processing (`PictureKernel`)
- **Dimensions:** $800$ columns $\times$ $600$ rows ($480,000$ pixels)
- **Block size:** `dim3 threadsPerBlock(16, 16);` (256 threads/block)
- **Grid size:** `dim3 numBlocks(50, 38);` (1,900 blocks)
- **Saturation cap:** Values exceeding 255 saturated at 255 (240,121 pixels capped).
- **CPU Verification:** `0 mismatches out of 480000`

### Exercise 4: 3D Matrix Multiplication
- **Dimensions:** $500 \times 500 \times 500 = 125,000,000$ elements
- **Block size:** `dim3 threadsPerBlock(8, 8, 8);` (512 threads/block)
- **Grid size:** `dim3 numBlocks(63, 63, 63);` (250,047 blocks)
- **Boundary guards:** `x < nx && y < ny && z < nz` protect all three dimensions.
- **CPU Verification:** `0 mismatches out of 125000000`
