// src/manifold_meta.rs
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct ManifoldMeta {
    pub builder_config: BuilderConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BuilderConfig {
    pub synthesis: Synthesis,
    pub nfeatures: WrappedUsize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Synthesis {
    #[serde(rename = "TauMode")]
    pub tau_mode: String, // "Median", "Mean", ...
}

#[derive(Debug, Clone, Deserialize)]
pub struct WrappedUsize {
    #[serde(rename = "Usize")]
    pub value: usize,
}

impl ManifoldMeta {
    pub fn nfeatures(&self) -> usize {
        self.builder_config.nfeatures.value
    }

    pub fn tau_mode_str(&self) -> &str {
        self.builder_config.synthesis.tau_mode.as_str()
    }
}
