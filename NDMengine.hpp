/*
 * NDMatrix is a generalized N-dimensional matrix class that allows creation of multi-dimensional
 * arrays of any shape using dynamic memory via std::vector. It supports flexible matrix or tensor
 * shapes like 2x2, 100x100, or even 4D shapes like 3x4x2x5. The class internally flattens the N-D
 * structure into a contiguous 1D vector and calculates access offsets using strides, allowing
 * efficient memory access.
 *
 * It is constructed by passing a vector of dimension sizes (e.g., {3.3} for 3x3), and initialized
 * with a value. Indexing is done via `at({i, j, ...})` using initializer lists. The class ensures
 * bounds checking and includes utility methods such as `fill()` and access to shape information.
 *
 */

//
//  Created by Marco Å. Ojeda, NOXIUM RESEARCH -- this project started on 08/07/2025.
//

#include <vector>
#include <stdexcept>
#include <numeric>
#include <initializer_list>

template<typename T>
class NDMatrix{
private:
    std::vector<size_t> shape;      // Stores the size of each dimension, e.g., {3, 4, 5} for a 3x4x5 tensor.
    std::vector<size_t> strides;    // Stores precomputed stride (offset multiplier) for each dimension to flatten indices.
    std::vector<T> data;            // Contiguous 1D storage of all matrix elements in row-major order.
    
    // Computes the linear (flattened) index from a list of N-dimensional indices.
    // Throws if the number of indices does not match the matrix's dimensionality
    // or if any individual  index is out of bounds. This is used internally to map
    // multi-dimensional access to the underlying 1D data vector.
    size_t compute_flat_index(std::initializer_list<size_t> indices) const {
        
        // Check if the number of indices matches the number of dimensions.
        if (indices.size() != shape.size())
            throw std::invalid_argument("([NDMatrix Error - std::invalid_argument]):: Incorrect number of indices.");
        
        size_t idx = 0;
        size_t i = 0;
        
        // Compute flattened index using row-major layout.
        for (size_t val : indices) {
            // Ensure each index is within bounds for its dimension.
            if (val >= shape[i])
                throw std::out_of_range("([NDMatrix Error - std::out_of_range]):: Index out of bounds.");
            
            // Multiply index value by its stride and accumulate.
            idx += val * strides[i];
            ++i;
        }
        return idx;
    }
    
    size_t compute_flat_index(const std::vector<size_t>& indices) const {
        if (indices.size() != shape.size()) {
            throw std::invalid_argument("([NDMatrix Error - std::invalid_argument]):: compute_flat_index(): index dimensionality mismatch.");
        }
        size_t flat_index = 0;
        for (size_t i = 0; i < shape.size(); ++i) {
            flat_index += indices[i] * strides[i];
        }
        return flat_index;
    }
    
    // Computes the stride vector for the NDMatrix based on its shape.
    // Strides represent how many elements to skip in the flattened data vector
    // when moving by one index along each dimension (row-major layout).
    void compute_strides(){
        // Resize the strides vector to match the number of dimensions.
        strides.resize(shape.size());
        size_t stride = 1;
        
        // Compute strides in reverse order (right to left, i.e., row-major).
        for (size_t i = shape.size(); i-- > 0;){
            strides[i] = stride;        // Assign current stride.
            stride *= shape[i];         // Update stride for next (outer) dimension.
        }
    }
    
public:
    // Constructor for NDMatrix.
    // Takes a vector of dimension sizes (`dims`) and an optional initial value (`init`).
    // Initializes the shape, computes the strides for index translation,
    // calculates the total number of elements, and fills the underlying data vector.
    NDMatrix(const std::vector<size_t>& dims, const T& init = T()) :
    shape(dims){
        
        compute_strides(); // Precompute how indices map to flat memory.
        
        // Compute total number of elements: product of all dimensions.
        size_t total_size = std::accumulate(
                                            shape.begin(), shape.end(), size_t(1), std::multiplies<>());
        
        // Initialize data vector with total size, filled with `init`.
        data.resize(total_size, init);
    }
    
    // Returns a mutable reference to the elements at the specified N-dimensional index.
    // Accepts an initializer list of indices (e.g., {i, j,k}) and internally computes
    // the corresponding flat index into the 1D data vector. Throws if index is invalid.
    T& at(std::initializer_list<size_t> indices){
        // Compute the flattened (linear) index and return a reference to the element.
        return data[compute_flat_index(indices)];
    }
    
    // Returns a const reference to the element at the specified N-dimensional index.
    // Accepts an initializer list of indices (e.g., {i, j, k}) and computes the corresponding
    // flat index into the internal 1D data vector. Throws if index is invalid.
    const T& at(std::initializer_list<size_t> indices) const {
        // Compute the flattened (linear) index and return a const reference to the element.
        return data[compute_flat_index(indices)];
    }
    
    // Overload for dynamic vector indexing (used in slicing/view logic)
    T& at(const std::vector<size_t>& indices){
        return data[compute_flat_index(indices)];
    }
    
    const T& at(const std::vector<size_t>& indices) const {
        return data[compute_flat_index(indices)];
    }
    
    // Returns the total number of elements stored in the matrix (i.e., total size)
    size_t size() const {
        return data.size();
    }
    
    // Flat subscript operator for direct access to internal data (row-major)
    T& operator[](size_t i){
        return data[i];
    }
    
    const T& operator[] (size_t i) const {
        return data[i];
    }
    
    // Returns a const reference to the vector containing the size of each dimension.
    // This provides external access to the shape of the NDMatrix, such as {3, 4, 5} for a 3x4x5 tensor.
    const std::vector<size_t>& dimensions() const {
        return shape; // Dimension sizes stored in order of axes
    }
    
    // Fills the entire NDMatrix with the given value. This method overwrites all existing
    // elements in the data vector with the specified value. It is useful, for reinitializing
    // or resetting the matrix contents.
    void fill(const T& value){
        std::fill(data.begin(), data.end(), value); // Fill all elements in the matrix with 'value'.
    }
    
    // Variadic template operator() to allow intuitive multi-index access, e.g., matrix(i, j, k)
    template<typename... Args>
    T& operator()(Args... args){
        // Ensure all provided arguments can be converted to size_t (type-safe indexing)
        static_assert((std::is_convertible_v<Args, size_t> && ...), "All indices must be size_t-convertible.");
        
        // Forward the unpacked indices as an initializer_list to .at(...)
        return at({static_cast<size_t>(args)...});
    }
    
    // Const version of the variadic operator() for read-only access
    template<typename... Args>
    const T& operator()(Args... args) const {
        static_assert((std::is_convertible_v<Args, size_t> && ...), "All indices must be size_t-convertible.");
        return at({static_cast<size_t>(args)...});
    }
    
    
    // reshape() allows changing the dimensional structure of the matrix without reallocating or modifying the
    // underlying data.
    void reshape(const std::vector<size_t>& new_shape){
        // Compute total number of elements in the new shape
        size_t new_total = std::accumulate(new_shape.begin(), new_shape.end(), size_t(1), std::multiplies<>());
        
        // Validity shape compatibility
        if (new_total != data.size())
            throw std::invalid_argument("([NDMatrix Error - std::invalid_argument]):: reshape(): element count mismatch.");
        
        // Apply new shape and update strides
        shape = new_shape;
        compute_strides();
    }
    
    // Iterator support
    auto begin() { return data.begin(); }
    auto end() { return data.end(); }
    
    auto begin() const { return data.begin(); }
    auto end() const { return data.end(); }
    
    auto cbegin() const { return data.cbegin(); }
    auto cend() const { return data.cend(); }
    
    // Reverse iterator support
    auto rbegin() { return data.rbegin(); }
    auto rend() { return data.rend(); }
    
    auto rbegin() const { return data.rbegin(); }
    auto rend() const { return data.rend(); }
    
    auto crbegin() const { return data.crbegin(); }
    auto crend() const { return data.crend(); }
    
    
    // A basic view function that returns a submatrix view by copying elements into a new NDMatrix.
    // This is a first simple version of slicing that does not share data but returns a proper NDMatrix.
    // Example: view({1,0}, {2,2}) returns a 2x2 matrix slice starting at position (1, 0).
    NDMatrix<T> view(const std::vector<size_t>& start, const std::vector<size_t>& extent) const {
        // Ensure dimensionality matches
        if (start.size() != shape.size() || extent.size() != shape.size()){
            throw std::invalid_argument("([NDMatrix Error - std::invalid_argument]):: view(): start and extent must match number of dimensions.");
        }
        
        // Ensure bounds are valid
        for (size_t i = 0; i < shape.size(); ++i){
            if (start[i] + extent[i] > shape[i]){
                throw std::out_of_range("([NDMatrix Error - std::out_of_range]):: view(): requested slice is out of bounds.");
            }
        }
        
        // Create new matrix to hold the sliced values
        NDMatrix<T> result(extent);
        
        // Recursive lambda to copy values from this matrix to the result matrix.
        // This traverses the N-dimensional index space defined by `extent` starting from `start`.
        // For each position, it computes the corresponding index in the source (this) and target
        // (result) matrix.
        std::function<void(size_t, std::vector<size_t>, std::vector<size_t>)> copy;
        copy = [&](size_t dim, std::vector<size_t> idx_this, std::vector<size_t> idx_new){
            if (dim == shape.size()){
                result.at({idx_new.begin(), idx_new.end()}) = at({idx_this.begin(), idx_this.end()});
                return;
            }
            for (size_t i = 0; i < extent[dim]; ++i){
                idx_this[dim] = start[dim] + i;
                idx_new[dim] = i;
                copy(dim + 1, idx_this, idx_new);
            }
        };
        
        copy(0, start, std::vector<size_t>(shape.size(), 0));
        return result;
    }
    
    // ---------------------------------
    // Elementwise Arithmetic Operations
    // --------------------------------
    
    NDMatrix operator+(const NDMatrix& other) const {
        if (shape != other.shape){
            throw std::invalid_argument("([NDMatrix Error - std::invalid_argument]):: operator+: shapes do not match");
        }
        NDMatrix result(shape);
        for (size_t i = 0; i < data.size(); ++i){
            result.data[i] = data[i] + other.data[i];
        }
        return result;
    }
    
    NDMatrix operator-(const NDMatrix& other) const {
        if (shape != other.shape){
            throw std::invalid_argument("([NDMatrix Error - std::invalid_argument]):: operator-: shapes do not match.");
        }
        NDMatrix result(shape);
        for (size_t i = 0; i < data.size(); ++i){
            result.data[i] = data[i] - other.data[i];
        }
        return result;
    }
    
    NDMatrix operator*(const NDMatrix& other) const {
        if (shape != other.shape){
            throw std::invalid_argument("([NDMatrix Error - std::invalid_argument]):: operator*: shapes do not match.");
        }
        NDMatrix result(shape);
        for (size_t i = 0; i < data.size(); ++i){
            result.data[i] = data[i] * other.data[i];
        }
        return result;
    }
    
    NDMatrix operator/(const NDMatrix& other) const {
        if (shape != other.shape){
            throw std::invalid_argument("([NDMatrix Error - std::invalid_argument]):: operator/: shapes do not match.");
        }
        NDMatrix result(shape);
        for (size_t i = 0; i < data.size(); ++i){
            result.data[i] = data[i] / other.data[i];
        }
        return result;
    }
    
    // ----------------------------
    // Scalar Arithmetic Operators
    // ----------------------------
    
    NDMatrix operator+(const T& scalar) const{
        NDMatrix result(shape);
        for (size_t i = 0; i < data.size(); ++i){
            result.data[i] = data[i] + scalar;
        }
        return result;
    }
    
    NDMatrix operator-(const T& scalar) const {
        NDMatrix result(shape);
        for (size_t i = 0; i < data.size(); ++i){
            result.data[i] = data[i] - scalar;
        }
        return result;
    }
    
    NDMatrix operator*(const T& scalar) const {
        NDMatrix result(shape);
        for (size_t i = 0; i < data.size(); ++i){
            result.data[i] = data[i] * scalar;
        }
        return result;
    }
    
    NDMatrix operator/ (const T& scalar) const {
        NDMatrix result(shape);
        for (size_t i = 0; i < data.size(); ++i){
            result.data[i] = data[i] / scalar;
        }
        return result;
    }
    
    // Extracts a (N-1)-dimensional slice by fixing one index along the specified axis.
    // For example -- for a 3D matrixx of shape {3, 4, 5}, calling slice(0,2) returns
    // a 2D matrix of shape {4, 5} corresponding to the third "layer".
    // Throws: std::out_of_range if axis or index is out of bounds.
    NDMatrix slice(size_t axis, size_t index) const {
        // Error checking
        if (axis >= shape.size()){
            throw std::out_of_range("([NDMatrix Error - std::out_of_range]):: slice(): axis is out of range.");
        }
        if (index >= shape[index]){
            throw std::out_of_range("([NDMatrix Error - std::out_of_range]):: slice(): index is out of range for given axis.");
        }
        
        // Build shape of the result (N-1 dimensions)
        std::vector<size_t> result_shape;
        for (size_t i = 0; i < shape.size(); ++i){
            if (i != axis){
                result_shape.push_back(shape[i]);
            }
        }
        
        NDMatrix result(result_shape);
        
        // Recursive traversal to copy data
        std::function<void(size_t, std::vector<size_t>, std::vector<size_t>)> copy;
        copy = [&](size_t dim, std::vector<size_t> src_index, std::vector<size_t> dst_index){
            if (dim == shape.size()){
                result.at(dst_index) = this->at(src_index);
                return;
            }
            
            if (dim == axis){
                src_index[dim] = index;     // fix dimension.
                copy(dim + 1, src_index, dst_index);
            } else {
                size_t dst_dim = (dim < axis) ? dim : dim - 1;
                for (size_t i = 0; i < shape[dim]; ++i){
                    src_index[dim] = i;
                    dst_index[dst_dim] = i;
                    copy(dim + 1, src_index, dst_index);
                }
            }
        };
        copy(0, std::vector<size_t>(shape.size(), 0), std::vector<size_t>(shape.size() - 1, 0));
        return result;
    }
    
    // Computes the broadcast-compatible shape between this matrix and another matrix.
    // Follows broadcasting rules similar to NumPy: compare dimensions from the end toward
    // the front, dimensions must either match or one of them must be 1. Or if incompatible,
    // throws std::invalid_argument.
    //
    // Returns: A new shape vector that represents the shape of the broadcasted result.
    // Throws: std::invalid_argument if shapes are not broadcast-compatible.
    
    std::vector<size_t> broadcast_shape(const std::vector<size_t>& other_shape) const {
        std::vector<size_t> result;
        
        const size_t n = shape.size();
        const size_t m = other_shape.size();
        const size_t max_dim = std::max(n, m);
        
        result.resize(max_dim);
        
        // Iterate from the last dimension backwards
        for (size_t i = 0; i < max_dim; ++i){
            size_t dim_a = (i < n) ? shape[n - 1 - i] : 1;
            size_t dim_b = (i < m) ? other_shape[m - 1 - i] : 1;
            
            if (dim_a != dim_b && dim_a != 1 && dim_b != 1){
                throw std::invalid_argument("([NDMengine Error - std::invalid_argument]):: broadcast_shape(): shapes are not broadcast-compatible.");
            }
            result[max_dim - 1 - i] = std::max(dim_a, dim_b);
        }
        return result;
    }
    
    
    
    /*
     *      Index Conversion Utility - ICU
     *      ------------------------------
     *      These allow easier interoperability and introspection for debugging, slicing, and iteration.
     *      Useful for reverse-mapping flat indices (especially in slicing, broadcasting).
     *      Enables external tools like custom iterators or visualizations.
     */
    
    //  unravel_index -- Converts a flat index into an N-dimensional index vector based on the matrix's shape
    // and strides. Example: For a shape {3, 4} and flat index 5, returns {1, 1}.
    std::vector<size_t> unravel_index(size_t flat_index) const {
        std::vector<size_t> indices(shape.size());
        for (size_t i = 0; i < shape.size(); ++i){
            indices[i] = flat_index / strides[i];
            flat_index %= strides[i];
        }
        return indices;
    }
    
    // ravel_index -- Converts an N-dimensional idnex vector into a flat index using internal stride information.
    // Throws if the number of dimensions does not match or any index is out of bounds.
    // Example: For shape {3, 4} and indices {1, 1}, returns 5.
    
    size_t ravel_index(const std::vector<size_t>& indices) const{
        if (indices.size() != shape.size()){
            throw std::invalid_argument("([NDMengine Error - ravel_index]):: Dimensionality mismatch.");
        }
        size_t flat_index = 0;
        for (size_t i = 0; i < shape.size(); ++i){
            if (indices[i] >= shape[i]){
                throw std::out_of_range("([NDMengine Error - ravel_index]):: Index out of bounds.");
            }
            flat_index += indices[i] * strides[i];
        }
        return flat_index;
    }
    
    // Continue here on the documentation!!
    
    /*
     *      Shape Comparison and Alignment Helpers
     *      ---------------------------------------
     *      Needed in internal logic and unit tests.
     *      Helps with early exits in overloaded operators. Supports future tools like
     *      shape assertions and debugging tools.
     */
    
    // same_shape - Checks whether another NDMatrix instance has the same shape as this matrix.
    // Useful for asserting exact shape equality in operations like assignment or elementwise arithmetic.
    
    bool same_shape(const NDMatrix& other) const {
        return this->shape == other.shape;
    }
    
    // is_broadcastable_with -- Checks if the current matrix shape is broadcast-compatible with another shape vector.
    // From the trailing dimension backward, each dimension must match or be 1 in either.
    // Missing dimensions are treated as 1.
    
    bool is_broadcastable_with (const std::vector<size_t>& smaller_shape) const {
        size_t n = shape.size();
        size_t m = smaller_shape.size();
        
        for (size_t i = 0; i < n; ++i){
            size_t dim_a = shape[n - 1 - i];
            size_t dim_b = (i < m) ? smaller_shape[m - 1 - i] : 1;
            
            if (dim_b != dim_a && dim_b != 1){
                return false;
            }
        }
        return true;
    }
    
    /*
     *      Memory Layout Queries
     *      ---------------------
     *      Allow users to query layout, capacity, and memory characteristics. It enables
     *      advanced users to do raw memory access or wrap GPU tensors around your layout.
     */
    
    // is_contiguous -- Returns true if the matrix occupies a single, contiguous block of memory
    // using standrd row-major layout. This assumes default stride layout.
    // Note: This function is trivial in this engine since the internal data is always stored
    // contiguously in row-major order. However, it may become critical if views or advanced strides
    // are introduced in the future.
    bool is_contiguous() const {
        size_t expected_stride = 1;
        for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i){
            if (strides[i] != expected_stride) return false;
            expected_stride *= shape[i];
        }
        return true;
    }
    
    // stride -- Returns the stride size (memory step) for a given axis. This tells how many
    // elements must be jumped over in the flattened buffer to move one unit along the axis.
    // Throws std::out_of_range if axis is invalid.
    size_t stride(size_t axis) const {
        if (axis >= strides.size()){
            throw std::out_of_range("([NDMatrix Error - stride]):: Axis out of bounds.");
        }
        return strides[axis];
    }
    
    // data_ptr -- Provides direct access to the underlying new raw memory buffer (non-const version).
    // This is useful for GPU data uploads or advanced external memory tools.
    T* data_ptr(){
        return data.data();
    }
    
    // data_ptr (const) -- provides const-qualified direct access to the raw memory buffer.
    const T* data_ptr() const {
        return data.data();
    }
    
    
}; // -------- -------- -------- -------- NDMatrix Class Ending...

// ------------------------------------------
// Left-hand Scalar Operations for NDMatrix
//-------------------------------------------

/*
 * Enables: scalar + matrix
 * Example: auto B = 2.5 + A;
 * Performs elementwise addition: result(i) = scalar + A(i)
 */

template<typename T>
NDMatrix<T> operator+(const T& scalar, const NDMatrix<T>& matrix){
    NDMatrix<T> result(matrix.dimensions());    // same shape.
    for (size_t i = 0; i < result.size(); ++i){
        result[i] = scalar + matrix[i];     // elementwise
    }
    return result;
}


/*
 * Enables: scalar - matrix
 * Example: auto B = 2.5 - A;
 * Performs elementwise subtraction: result(i) = scalar - A(i)
 */

template<typename T>
NDMatrix<T> operator-(const T& scalar, const NDMatrix<T>& matrix){
    NDMatrix<T> result(matrix.dimensions());
    for (size_t i = 0; i < result.size(); ++i){
        result[i] = scalar - matrix[i];
    }
    return result;
}

/*
 * Enables: scalar * matrix
 * Example: auto B = 2.5 * A;
 * Performs elementwise multiplication: result(i) = scalar * A(i)
 */

template<typename T>
NDMatrix<T> operator*(const T& scalar, const NDMatrix<T>& matrix){
    NDMatrix<T> result(matrix.dimensions());
    for (size_t i = 0; i < result.size(); ++i){
        result[i] = scalar * matrix[i];
    }
    return result;
}

/*
 * Enables: scalar / matrix
 * Example: auto B = 2.5 / A;
 * Performs elementwise division: result(i) = scalar / A(i)
 */

template<typename T>
NDMatrix<T> operator/(const T& scalar, const NDMatrix<T>& matrix){
    NDMatrix<T> result(matrix.dimensions());
    for (size_t i = 0; i < result.size(); ++i){
        result[i] = scalar / matrix[i];
    }
    return result;
}
