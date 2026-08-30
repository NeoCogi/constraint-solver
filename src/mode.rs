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

//! Serial and explicitly sized parallel execution modes for solver-owned
//! numerical work.

use rayon::{ThreadPool, ThreadPoolBuilder};

/// Execution strategy used for matrix operations within a solver instance.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum Mode {
    /// Execute all supported numerical operations on the calling thread.
    #[default]
    Serial,
    /// Execute sufficiently large supported operations in a dedicated Rayon
    /// pool with exactly `thread_count` worker threads.
    Parallel {
        /// Positive number of worker threads allocated to this solver.
        thread_count: usize,
    },
}

/// Build the optional dedicated thread pool represented by an execution mode.
pub(crate) fn build_thread_pool(mode: &Mode) -> Result<Option<ThreadPool>, String> {
    // Serial mode needs no pool. Parallel mode validates its public count before
    // delegating platform-specific worker creation to Rayon.
    match mode {
        Mode::Serial => Ok(None),
        Mode::Parallel { thread_count } => {
            if *thread_count == 0 {
                return Err("thread_count must be greater than 0".to_string());
            }
            ThreadPoolBuilder::new()
                .num_threads(*thread_count)
                .build()
                .map(Some)
                .map_err(|err| format!("failed to build thread pool: {err}"))
        }
    }
}
