//! Hardware memory presets (VRAM × RAM) for the interactive launcher and CLI.
//!
//! Maps user-selected GPU VRAM and system RAM tiers into the exact environment
//! budgets read by `victory-gpu`, reducing texture/pool/eviction footprints on
//! modest laptops (2–4 GB VRAM / 16–20 GB RAM) or keeping full quality on
//! high-end machines (8–16+ GB VRAM / 32+ GB RAM).

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::Console;

/// Distant-tree wind pipelines skipped in low-spec mode (require diagnostics).
pub const LOW_SPEC_SKIP_PIPELINES: &str = "4e35b816f2f91d68,b3ba0ef9ee352720,5d627391703fcb4c,ec6e4c363cf9ef2b,c89489afd89ee7f0,45f081396c2c59ba,63083c777948927a,0bd4d80ad23b106b";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VramTier {
    Gb2,
    Gb4,
    Gb8,
    Gb16Plus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RamTier {
    Gb16,
    Gb20,
    Gb32Plus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SavedConsole {
    SeriesX,
    SeriesS,
}

impl From<Console> for SavedConsole {
    fn from(c: Console) -> Self {
        match c {
            Console::SeriesX => SavedConsole::SeriesX,
            Console::SeriesS => SavedConsole::SeriesS,
        }
    }
}

impl From<SavedConsole> for Console {
    fn from(c: SavedConsole) -> Self {
        match c {
            SavedConsole::SeriesX => Console::SeriesX,
            SavedConsole::SeriesS => Console::SeriesS,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HardwareProfile {
    pub vram: VramTier,
    pub ram: RamTier,
    pub console: SavedConsole,
    pub exe: Option<PathBuf>,
}

impl Default for HardwareProfile {
    fn default() -> Self {
        Self {
            vram: VramTier::Gb4,
            ram: RamTier::Gb20,
            console: SavedConsole::SeriesS,
            exe: None,
        }
    }
}

/// Detects the closest VRAM and RAM presets from physical byte counts.
pub fn detect_tiers(ram_bytes: u64, vram_bytes: u64) -> (VramTier, RamTier) {
    let vram_mib = vram_bytes >> 20;
    let vram = if vram_mib <= 3072 {
        VramTier::Gb2
    } else if vram_mib <= 6144 {
        VramTier::Gb4
    } else if vram_mib <= 10240 {
        VramTier::Gb8
    } else {
        VramTier::Gb16Plus
    };

    let ram_gib = ram_bytes >> 30;
    let ram = if ram_gib <= 16 {
        RamTier::Gb16
    } else if ram_gib <= 24 {
        RamTier::Gb20
    } else {
        RamTier::Gb32Plus
    };
    (vram, ram)
}

impl VramTier {
    pub fn label(self) -> &'static str {
        match self {
            Self::Gb2 => "2 GB",
            Self::Gb4 => "4 GB",
            Self::Gb8 => "8 GB",
            Self::Gb16Plus => "16 GB+",
        }
    }

    pub fn suggested_console(self) -> SavedConsole {
        match self {
            Self::Gb2 | Self::Gb4 => SavedConsole::SeriesS,
            Self::Gb8 | Self::Gb16Plus => SavedConsole::SeriesX,
        }
    }
}

impl RamTier {
    pub fn label(self) -> &'static str {
        match self {
            Self::Gb16 => "16 GB",
            Self::Gb20 => "20 GB",
            Self::Gb32Plus => "32 GB+",
        }
    }
}

impl HardwareProfile {
    /// Whether this combination activates the low-spec reduction profile.
    pub fn is_low_spec(&self) -> bool {
        matches!(self.vram, VramTier::Gb2 | VramTier::Gb4)
            || matches!(self.ram, RamTier::Gb16 | RamTier::Gb20)
    }

    /// Computes all environment overrides corresponding to this profile.
    pub fn env_vars(&self) -> Vec<(&'static str, String)> {
        let mut vars: Vec<(&'static str, String)> = Vec::with_capacity(10);

        let (tex_max, card_on, card_mb, placed_evict, keep_frames) = match self.vram {
            VramTier::Gb2 => ("256", "0", None, "120", "4"),
            VramTier::Gb4 => ("512", "0", None, "240", "8"),
            VramTier::Gb8 => ("1024", "1", Some("1024"), "600", "16"),
            VramTier::Gb16Plus => ("0", "1", Some("2048"), "0", "32"),
        };

        let (staging_mb, shadow_evict, pool_mb) = match self.ram {
            RamTier::Gb16 => ("96", "120", "64"),
            RamTier::Gb20 => ("128", "180", "128"),
            RamTier::Gb32Plus => (
                "384",
                if matches!(self.vram, VramTier::Gb2 | VramTier::Gb4) {
                    "600"
                } else {
                    "0"
                },
                "512",
            ),
        };

        vars.push(("VICTORY_TEXTURE_MAX", tex_max.to_owned()));
        vars.push(("VICTORY_CARD_BUFFERS", card_on.to_owned()));
        if let Some(mb) = card_mb {
            vars.push(("VICTORY_CARD_BUFFERS_MB", mb.to_owned()));
        }
        vars.push(("VICTORY_PLACED_TEX_EVICT", placed_evict.to_owned()));
        vars.push(("VICTORY_KEEP_FRAMES", keep_frames.to_owned()));
        vars.push(("VICTORY_STAGING_BUDGET_MB", staging_mb.to_owned()));
        vars.push(("VICTORY_SHADOW_EVICT_FRAMES", shadow_evict.to_owned()));
        vars.push(("VICTORY_POOL_MB", pool_mb.to_owned()));
        // Uncap host frame pacing to avoid Windows timer sleep jitter (1–15 ms).
        vars.push(("VICTORY_FPS", "0".to_owned()));

        if self.is_low_spec() {
            vars.push(("VICTORY_LOW_SPEC", "1".to_owned()));
            vars.push(("VICTORY_SKIP_PIPELINES", LOW_SPEC_SKIP_PIPELINES.to_owned()));
        } else {
            vars.push(("VICTORY_LOW_SPEC", "0".to_owned()));
        }

        vars
    }

    /// Human-readable summary displayed live in the launcher UI.
    pub fn summary(&self) -> String {
        let mode = match (self.vram, self.ram) {
            (VramTier::Gb2, _) => "Modo Ultra-Reduzido (Máxima economia de VRAM)",
            (VramTier::Gb4, _) => "Modo Reduzido / Low-Spec (Notebook 4 GB VRAM)",
            (VramTier::Gb8, RamTier::Gb32Plus) => "Modo Balanceado (8 GB VRAM)",
            (VramTier::Gb8, _) => "Modo Balanceado com Economia de RAM",
            (VramTier::Gb16Plus, RamTier::Gb32Plus) => "Modo Normal (Qualidade total do emulador)",
            (VramTier::Gb16Plus, _) => "Modo Normal de GPU com Economia de RAM",
        };
        let tex = match self.vram {
            VramTier::Gb2 => "256 px",
            VramTier::Gb4 => "512 px",
            VramTier::Gb8 => "1024 px",
            VramTier::Gb16Plus => "Original (sem corte)",
        };
        let card = match self.vram {
            VramTier::Gb2 | VramTier::Gb4 => "OFF (salva VRAM)",
            VramTier::Gb8 => "ON (máx. 1 GB)",
            VramTier::Gb16Plus => "ON (máx. 2 GB)",
        };
        let ram_desc = match self.ram {
            RamTier::Gb16 => "Staging 96 MB • Pools 64 MB • LRU 120f",
            RamTier::Gb20 => "Staging 128 MB • Pools 128 MB • LRU 180f",
            RamTier::Gb32Plus => "Staging 384 MB • Pools 512 MB • Padrão",
        };
        let console = match self.console {
            SavedConsole::SeriesS => "Xbox Series S (~7,6 GB)",
            SavedConsole::SeriesX => "Xbox Series X (~13,5 GB)",
        };
        format!(
            "{mode}\r\nGPU ({}): Texturas {tex} | Card VRAM: {card}\r\nRAM ({}): {ram_desc}\r\nConsole: {console}",
            self.vram.label(),
            self.ram.label(),
        )
    }

    /// Applies the selected profile to process environment variables without
    /// overwriting variables explicitly set by the caller.
    ///
    /// Must only be called on the main thread during startup before worker
    /// threads are spawned.
    pub fn apply_env(&self) {
        for (key, val) in self.env_vars() {
            if std::env::var_os(key).is_none() {
                // SAFETY: called on the main thread during CLI/launcher startup
                // before any runtime or title worker threads are spawned.
                unsafe { std::env::set_var(key, val) };
            }
        }
    }

    pub fn load_saved(path: &Path) -> Option<Self> {
        let text = std::fs::read_to_string(path).ok()?;
        toml::from_str(&text).ok()
    }

    pub fn save(&self, path: &Path) {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(text) = toml::to_string_pretty(self) {
            let _ = std::fs::write(path, text);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn find<'a>(vars: &'a [(&'static str, String)], key: &str) -> Option<&'a str> {
        vars.iter().find(|(k, _)| *k == key).map(|(_, v)| v.as_str())
    }

    #[test]
    fn detect_tiers_maps_common_hardware() {
        // Lenovo IdeaPad: 19 GiB RAM, 2 GiB VRAM (MX110)
        assert_eq!(
            detect_tiers(19 << 30, 2 << 30),
            (VramTier::Gb2, RamTier::Gb20)
        );
        // Notebook 4 GiB VRAM, 16 GiB RAM
        assert_eq!(
            detect_tiers(16 << 30, 4 << 30),
            (VramTier::Gb4, RamTier::Gb16)
        );
        // Mid-range 8 GiB VRAM, 32 GiB RAM
        assert_eq!(
            detect_tiers(32 << 30, 8 << 30),
            (VramTier::Gb8, RamTier::Gb32Plus)
        );
        // High-end 16 GiB VRAM, 64 GiB RAM
        assert_eq!(
            detect_tiers(64 << 30, 16 << 30),
            (VramTier::Gb16Plus, RamTier::Gb32Plus)
        );
    }

    #[test]
    fn low_spec_2gb_20gb_sets_aggressive_reduction() {
        let p = HardwareProfile {
            vram: VramTier::Gb2,
            ram: RamTier::Gb20,
            console: SavedConsole::SeriesS,
            exe: None,
        };
        assert!(p.is_low_spec());
        let vars = p.env_vars();
        assert_eq!(find(&vars, "VICTORY_TEXTURE_MAX"), Some("256"));
        assert_eq!(find(&vars, "VICTORY_CARD_BUFFERS"), Some("0"));
        assert_eq!(find(&vars, "VICTORY_PLACED_TEX_EVICT"), Some("120"));
        assert_eq!(find(&vars, "VICTORY_KEEP_FRAMES"), Some("4"));
        assert_eq!(find(&vars, "VICTORY_STAGING_BUDGET_MB"), Some("128"));
        assert_eq!(find(&vars, "VICTORY_SHADOW_EVICT_FRAMES"), Some("180"));
        assert_eq!(find(&vars, "VICTORY_POOL_MB"), Some("128"));
        assert_eq!(find(&vars, "VICTORY_LOW_SPEC"), Some("1"));
    }

    #[test]
    fn normal_16gb_32gb_runs_full_quality() {
        let p = HardwareProfile {
            vram: VramTier::Gb16Plus,
            ram: RamTier::Gb32Plus,
            console: SavedConsole::SeriesX,
            exe: None,
        };
        assert!(!p.is_low_spec());
        let vars = p.env_vars();
        assert_eq!(find(&vars, "VICTORY_TEXTURE_MAX"), Some("0"));
        assert_eq!(find(&vars, "VICTORY_CARD_BUFFERS"), Some("1"));
        assert_eq!(find(&vars, "VICTORY_CARD_BUFFERS_MB"), Some("2048"));
        assert_eq!(find(&vars, "VICTORY_PLACED_TEX_EVICT"), Some("0"));
        assert_eq!(find(&vars, "VICTORY_KEEP_FRAMES"), Some("32"));
        assert_eq!(find(&vars, "VICTORY_STAGING_BUDGET_MB"), Some("384"));
        assert_eq!(find(&vars, "VICTORY_SHADOW_EVICT_FRAMES"), Some("0"));
        assert_eq!(find(&vars, "VICTORY_POOL_MB"), Some("512"));
        assert_eq!(find(&vars, "VICTORY_LOW_SPEC"), Some("0"));
        assert_eq!(find(&vars, "VICTORY_SKIP_PIPELINES"), None);
    }
}
