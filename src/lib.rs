//! Tauformer - A GPT implementation
//!
// Copyright [2025] tuned.org.uk, Mec-iS
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

pub mod backend;
pub mod causalattention;
pub mod checkpoint;
pub mod config;
pub mod engine;
pub mod sampling;
pub mod tokenizer;
pub use checkpoint::{load_checkpoint, load_weights, save_checkpoint, save_weights};
pub mod pretraining;
pub mod rope;
pub mod tauattention;
pub mod taugpt;
pub mod taumode;

#[cfg(test)]
mod tests;

pub use backend::{AutoBackend, get_device, print_backend_info};
pub use config::NanoChatConfig;

use std::sync::Once;

static INIT: Once = Once::new();

pub fn init() {
    INIT.call_once(|| {
        // Read RUST_LOG env variable, default to "info" if not set
        let env = env_logger::Env::default().default_filter_or("debug");

        // don't panic if called multiple times across binaries
        let _ = env_logger::Builder::from_env(env)
            .is_test(true) // nicer formatting for tests
            .try_init();
    });
}

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::{
        backend::{AutoBackend, get_device},
        causalattention::GptModel,
        config::NanoChatConfig,
    };
}
