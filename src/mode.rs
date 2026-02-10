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

use rayon::{ThreadPool, ThreadPoolBuilder};

#[derive(Debug, Clone)]
pub enum Mode {
    Serial,
    Parallel { thread_count: usize },
}

impl Default for Mode {
    fn default() -> Self {
        Mode::Serial
    }
}

impl Mode {
    pub fn serial() -> Self {
        Mode::Serial
    }

    pub fn parallel(thread_count: usize) -> Self {
        Mode::Parallel { thread_count }
    }
}

pub(crate) fn build_thread_pool(mode: &Mode) -> Result<Option<ThreadPool>, String> {
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
