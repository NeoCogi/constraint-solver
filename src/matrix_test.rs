/*
MIT License

Copyright (c) 2026 Raja Lehtihet & Wael El Oraiby

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#[cfg(test)]
mod additional_matrix_tests {
    use crate::matrix::Matrix;

    #[test]
    fn test_lu_pivoting() {
        // Test case where pivoting is necessary
        let a = Matrix::from_vec(vec![0.0, 1.0, 1.0, 0.0], 2, 2).unwrap();

        let b = Matrix::from_vec(vec![2.0, 1.0], 2, 1).unwrap();

        let x = a.solve_lu(&b).unwrap();

        // Solution should be x = [1, 2]
        assert!((x[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((x[(1, 0)] - 2.0).abs() < 1e-10);

        // Verify
        let ax = &a * &x;
        assert!((ax[(0, 0)] - b[(0, 0)]).abs() < 1e-10);
        assert!((ax[(1, 0)] - b[(1, 0)]).abs() < 1e-10);
    }

    #[test]
    fn test_lu_4x4_system() {
        // The exact system from our constraint solver
        let j = Matrix::from_vec(
            vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, -5.0, -1.0, 5.0, 1.0,
            ],
            4,
            4,
        )
        .unwrap();

        let b = Matrix::from_vec(vec![0.0, 0.0, -0.5, 2.5], 4, 1).unwrap();

        let x = j.solve_lu(&b).unwrap();

        // Expected solution: [0, 0, 0.6, -0.5]
        assert!((x[(0, 0)] - 0.0).abs() < 1e-10);
        assert!((x[(1, 0)] - 0.0).abs() < 1e-10);
        assert!((x[(2, 0)] - 0.6).abs() < 1e-10);
        assert!((x[(3, 0)] - (-0.5)).abs() < 1e-10);

        // Verify J*x = b
        let jx = &j * &x;
        for i in 0..4 {
            assert!((jx[(i, 0)] - b[(i, 0)]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_lu_singular_detection() {
        // Singular matrix
        let a = Matrix::from_vec(vec![1.0, 2.0, 2.0, 4.0], 2, 2).unwrap();

        let b = Matrix::from_vec(vec![1.0, 1.0], 2, 1).unwrap();

        assert!(a.solve_lu(&b).is_err());
    }

    #[test]
    fn test_lu_nearly_singular() {
        // Nearly singular matrix
        let a = Matrix::from_vec(vec![1.0, 1.0, 1.0, 1.0 + 1e-12], 2, 2).unwrap();

        let b = Matrix::from_vec(vec![2.0, 2.0 + 1e-12], 2, 1).unwrap();

        // Should still solve (though with reduced accuracy)
        match a.solve_lu(&b) {
            Ok(x) => {
                // Solution should be approximately [1, 1]
                assert!((x[(0, 0)] - 1.0).abs() < 1e-6);
                assert!((x[(1, 0)] - 1.0).abs() < 1e-6);
            }
            Err(_) => {
                // It's OK if it fails due to being too singular
            }
        }
    }
}
